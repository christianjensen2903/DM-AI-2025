# passage_ltr_retriever.py

import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import rank_bm25
from ltr_passages import precompute_passage_tokens, split_by_sections, split_by_words
from tqdm import tqdm

from langchain.schema import Document

from bm25_retriever import (
    preprocess_func,
    preprocess_statements,
    normalize_medical_text,
    get_retrieval_predictions,
)
from utils import load_cleaned_documents, load_topics

# ---------- constants ----------
FEATURE_NAMES = ["bm25", "tfidf_cosine", "jaccard", "query_len", "passage_len"]


# ---------- config ----------
@dataclass
class PassageLTRConfig:
    train_source: str = "synthetic"
    val_source: str = "train"
    k1: float = 1.5
    b: float = 0.75
    normalize: bool = False
    remove_stopwords: bool = False
    use_stemming: bool = False
    negative_per_query: int = 5
    random_state: int = 42
    early_stop_frac: float = 0.1
    passage_size: int = 150  # words per passage
    passage_overlap: int = 50
    use_section_splitting: bool = True
    candidate_pool: int = 100  # top BM25 to consider during inference


# ---------- logging ----------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def split_document_into_passages(
    doc: Document, config: PassageLTRConfig
) -> List[Document]:
    text = doc.page_content
    passages: List[Document] = []

    if config.use_section_splitting:
        sections = split_by_sections(text)
        for si, section in enumerate(sections):
            if len(section.split()) > config.passage_size:
                word_passages = split_by_words(
                    section, config.passage_size, config.passage_overlap
                )
                for pi, passage in enumerate(word_passages):
                    passages.append(
                        Document(
                            page_content=passage,
                            metadata={
                                **doc.metadata,
                                "passage_type": "section_word",
                                "section_id": si,
                                "passage_id": f"{si}_{pi}",
                                "parent_doc": doc.metadata.get("source", "unknown"),
                                "topic_id": doc.metadata.get("topic_id", -1),
                            },
                        )
                    )
            else:
                passages.append(
                    Document(
                        page_content=section,
                        metadata={
                            **doc.metadata,
                            "passage_type": "section",
                            "section_id": si,
                            "passage_id": str(si),
                            "parent_doc": doc.metadata.get("source", "unknown"),
                            "topic_id": doc.metadata.get("topic_id", -1),
                        },
                    )
                )
    else:
        word_passages = split_by_words(
            text, config.passage_size, config.passage_overlap
        )
        for pi, passage in enumerate(word_passages):
            passages.append(
                Document(
                    page_content=passage,
                    metadata={
                        **doc.metadata,
                        "passage_type": "word",
                        "passage_id": str(pi),
                        "parent_doc": doc.metadata.get("source", "unknown"),
                        "topic_id": doc.metadata.get("topic_id", -1),
                    },
                )
            )
    return passages


def create_passage_corpus(
    docs: List[Document], config: PassageLTRConfig
) -> List[Document]:
    all_passages = []
    logger.info(f"Splitting {len(docs)} documents into passages...")
    for doc in tqdm(docs, desc="Creating passages"):
        all_passages.extend(split_document_into_passages(doc, config))
    logger.info(f"Created {len(all_passages)} passages from {len(docs)} documents")
    return all_passages


# ---------- main class ----------
class PassageLTRRetriever:
    def __init__(self, config: PassageLTRConfig):
        self.config = config
        self.passages: List[Document] = []
        self.passage_token_lists: List[List[str]] = []
        self.bm25: Optional[rank_bm25.BM25Okapi] = None
        self.tfidf: Optional[TfidfVectorizer] = None
        self.passage_tfidf = None
        self.scaler: Optional[StandardScaler] = None
        self.ranker: Optional[lgb.LGBMRanker] = None
        self.topic2id: Dict[str, int] = {}
        self.id2topic: Dict[int, str] = {}

    def _prep_text(self, text: str) -> List[str]:
        if self.config.normalize:
            return preprocess_func(
                normalize_medical_text(text),
                remove_stopwords=self.config.remove_stopwords,
                use_stemming=self.config.use_stemming,
            )
        else:
            return preprocess_func(
                text,
                remove_stopwords=self.config.remove_stopwords,
                use_stemming=self.config.use_stemming,
            )

    def _build_tfidf(self, passages: List[Document], statements: List[List[str]]):
        passage_texts = [" ".join(self._prep_text(p.page_content)) for p in passages]
        statement_texts = [" ".join(tokens) for tokens in statements]
        self.tfidf = TfidfVectorizer()
        self.tfidf.fit(passage_texts + statement_texts)
        self.passage_tfidf = self.tfidf.transform(passage_texts)

    @staticmethod
    def _ndcg_at_k(relevances: List[int], k: int) -> float:
        def dcg(rs):
            return sum((2**r - 1) / np.log2(idx + 2) for idx, r in enumerate(rs[:k]))

        ideal = sorted(relevances, reverse=True)
        ideal_dcg = dcg(ideal)
        if ideal_dcg == 0:
            return 0.0
        return dcg(relevances) / ideal_dcg

    def _build_candidate_pairs_passages(
        self,
        passages: List[Document],
        statement_tokens: List[str],
        true_topic: int,
    ) -> Tuple[List[List[float]], List[int]]:
        rng = np.random.default_rng(self.config.random_state)

        positives = [
            (pi, passage)
            for pi, passage in enumerate(passages)
            if passage.metadata.get("topic_id", -1) == true_topic
        ]
        negatives = [
            (pi, passage)
            for pi, passage in enumerate(passages)
            if passage.metadata.get("topic_id", -1) != true_topic
        ]
        if not positives:
            return [], []

        num_neg = min(self.config.negative_per_query, len(negatives))
        neg_indices = rng.choice(len(negatives), num_neg, replace=False)

        pairs = []
        for pi, passage in positives:
            pairs.append((pi, passage, 1))
        for idx in neg_indices:
            pi, passage = negatives[idx]
            pairs.append((pi, passage, 0))

        all_bm25_scores = self.bm25.get_scores(statement_tokens)
        stmt_text = " ".join(statement_tokens)
        stmt_vec = self.tfidf.transform([stmt_text])[0]
        stmt_set = set(statement_tokens)

        passage_indices = [pi for pi, _, _ in pairs]
        relevant_passage_vecs = self.passage_tfidf[passage_indices]
        cos_sims = cosine_similarity(stmt_vec, relevant_passage_vecs)[0]

        feature_vecs = []
        labels = []

        for idx, (pi, passage, label) in enumerate(pairs):
            feats = []
            bm25_score = all_bm25_scores[pi]
            feats.append(bm25_score)
            cos_sim = cos_sims[idx]
            feats.append(cos_sim)

            passage_tokens = self.passage_token_lists[pi]
            passage_set = set(passage_tokens)
            jaccard = (
                (len(stmt_set & passage_set) / len(stmt_set | passage_set))
                if (stmt_set | passage_set)
                else 0.0
            )
            feats.append(jaccard)
            feats.append(len(statement_tokens))
            feats.append(len(passage_tokens))
            feature_vecs.append(feats)
            labels.append(label)

        return feature_vecs, labels

    def fit(self) -> None:
        base = Path("data")
        topics_json = base / "topics.json"
        cleaned_root = base / "cleaned_topics"

        # Load topics and base documents
        self.topic2id, _ = load_topics(topics_json)
        self.id2topic = {v: k for k, v in self.topic2id.items()}
        docs = load_cleaned_documents(cleaned_root, self.topic2id, normalize=False)
        logger.info(f"Loaded {len(docs)} base documents.")

        # Create passage corpus
        self.passages = create_passage_corpus(docs, self.config)

        # Preprocess for BM25
        if self.config.normalize:
            texts_processed = [
                preprocess_func(
                    normalize_medical_text(p.page_content),
                    remove_stopwords=self.config.remove_stopwords,
                    use_stemming=self.config.use_stemming,
                )
                for p in tqdm(self.passages, desc="Processing passages for BM25")
            ]
        else:
            texts_processed = [
                preprocess_func(
                    p.page_content,
                    remove_stopwords=self.config.remove_stopwords,
                    use_stemming=self.config.use_stemming,
                )
                for p in tqdm(self.passages, desc="Processing passages for BM25")
            ]

        # Load & preprocess statements
        train_statements_dir = base / self.config.train_source / "statements"
        train_answers_dir = base / self.config.train_source / "answers"
        val_statements_dir = base / self.config.val_source / "statements"
        val_answers_dir = base / self.config.val_source / "answers"

        processed_statements_train, true_labels_train, _ = preprocess_statements(
            train_statements_dir,
            train_answers_dir,
            self.config.normalize,
            self.config.remove_stopwords,
            self.config.use_stemming,
        )
        processed_statements_val, true_labels_val, _ = preprocess_statements(
            val_statements_dir,
            val_answers_dir,
            self.config.normalize,
            self.config.remove_stopwords,
            self.config.use_stemming,
        )

        # BM25 baseline (passages)
        self.bm25 = rank_bm25.BM25Okapi(
            texts_processed, k1=self.config.k1, b=self.config.b
        )
        y_true_bm25_val, y_pred_bm25_val = get_retrieval_predictions(
            self.bm25, self.passages, processed_statements_val, true_labels_val
        )
        bm25_acc = (np.array(y_true_bm25_val) == np.array(y_pred_bm25_val)).mean()
        logger.info(f"[BM25 passages val accuracy] {bm25_acc:.4f}")

        # TF-IDF build (fit on passages + train statements)
        self._build_tfidf(self.passages, processed_statements_train)

        # Precompute passage tokens for Jaccard
        self.passage_token_lists = precompute_passage_tokens(
            self.passages,
            normalize=self.config.normalize,
            remove_stopwords=self.config.remove_stopwords,
            use_stemming=self.config.use_stemming,
        )

        # Build training data
        X_rows = []
        y_rows = []
        group = []

        for stmt_tokens, true_topic in zip(
            processed_statements_train, true_labels_train
        ):
            feature_vecs, labels = self._build_candidate_pairs_passages(
                self.passages, stmt_tokens, true_topic
            )
            if not feature_vecs:
                continue
            group.append(len(feature_vecs))
            X_rows.extend(feature_vecs)
            y_rows.extend(labels)

        X_train = np.array(X_rows, dtype=float)
        y_train = np.array(y_rows, dtype=int)
        if X_train.size == 0:
            raise RuntimeError("No training data constructed.")

        # Internal split for early stopping
        num_queries = len(group)
        stmt_indices = list(range(num_queries))
        train_grp_idxs, es_grp_idxs = train_test_split(
            stmt_indices,
            test_size=self.config.early_stop_frac,
            random_state=self.config.random_state,
        )

        def flatten(group_idxs, X_flat, y_flat, group_list):
            X_parts = []
            y_parts = []
            grp = []
            cursor = 0
            for qi, g in enumerate(group_list):
                segment = slice(cursor, cursor + g)
                if qi in group_idxs:
                    X_parts.append(X_flat[segment])
                    y_parts.append(y_flat[segment])
                    grp.append(g)
                cursor += g
            if not X_parts:
                raise RuntimeError("No data after split.")
            return np.vstack(X_parts), np.concatenate(y_parts), grp

        X_train_main, y_train_main, group_main = flatten(
            train_grp_idxs, X_train, y_train, group
        )
        X_es, y_es, group_es = flatten(es_grp_idxs, X_train, y_train, group)

        self.scaler = StandardScaler()
        X_train_main_scaled = self.scaler.fit_transform(X_train_main)
        X_es_scaled = self.scaler.transform(X_es)

        X_train_df = pd.DataFrame(X_train_main_scaled, columns=FEATURE_NAMES)
        X_es_df = pd.DataFrame(X_es_scaled, columns=FEATURE_NAMES)

        self.ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            importance_type="gain",
            learning_rate=0.01,
            n_estimators=1000,
            num_leaves=31,
            min_child_samples=20,
            random_state=self.config.random_state,
        )

        self.ranker.fit(
            X_train_df,
            y_train_main,
            group=group_main,
            eval_set=[(X_es_df, y_es)],
            eval_group=[group_es],
            eval_at=[1, 3, 5],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=0)],
        )

        logger.info("Training complete.")

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        with open(out_dir / "topic2id.json", "w") as f:
            json.dump(self.topic2id, f)

        # Save passages
        docs_path = out_dir / "passages.jsonl"
        with open(docs_path, "w") as f:
            for p in self.passages:
                record = {
                    "page_content": p.page_content,
                    "metadata": p.metadata,
                }
                f.write(json.dumps(record) + "\n")

        # Save token cache
        joblib.dump(self.passage_token_lists, out_dir / "passage_token_lists.joblib")

        # Save BM25 state
        joblib.dump(
            {
                "tokenized_corpus": self.passage_token_lists,
                "k1": self.config.k1,
                "b": self.config.b,
            },
            out_dir / "bm25_state.joblib",
        )

        # Save tfidf, passage_tfidf, scaler, ranker
        joblib.dump(self.tfidf, out_dir / "tfidf.joblib")
        joblib.dump(self.passage_tfidf, out_dir / "passage_tfidf.joblib")
        joblib.dump(self.scaler, out_dir / "scaler.joblib")
        joblib.dump(self.ranker, out_dir / "ranker.joblib")

        logger.info(f"Saved passage LTR retriever to {out_dir}")

    @classmethod
    def load(cls, model_dir: Path) -> "PassageLTRRetriever":
        with open(model_dir / "config.json") as f:
            cfg_dict = json.load(f)
        config = PassageLTRConfig(**cfg_dict)
        self = cls(config)

        with open(model_dir / "topic2id.json") as f:
            self.topic2id = json.load(f)
        self.id2topic = {v: k for k, v in self.topic2id.items()}

        # Load passages
        self.passages = []
        with open(model_dir / "passages.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                self.passages.append(
                    Document(page_content=rec["page_content"], metadata=rec["metadata"])
                )

        # Load caches / models
        self.passage_token_lists = joblib.load(model_dir / "passage_token_lists.joblib")
        bm25_state = joblib.load(model_dir / "bm25_state.joblib")
        self.bm25 = rank_bm25.BM25Okapi(
            bm25_state["tokenized_corpus"], k1=bm25_state["k1"], b=bm25_state["b"]
        )
        self.tfidf = joblib.load(model_dir / "tfidf.joblib")
        self.passage_tfidf = joblib.load(model_dir / "passage_tfidf.joblib")
        self.scaler = joblib.load(model_dir / "scaler.joblib")
        self.ranker = joblib.load(model_dir / "ranker.joblib")

        logger.info(f"Loaded passage retriever from {model_dir}")
        return self

    def retrieve(
        self, raw_statement: str, top_k: int = 1, use_ltr: bool = True
    ) -> List[Document]:
        stmt_tokens = self._prep_text(raw_statement)
        if not stmt_tokens:
            return []

        bm25_scores = self.bm25.get_scores(stmt_tokens)
        top_candidate_idxs = np.argsort(bm25_scores)[::-1][: self.config.candidate_pool]

        feature_matrix = []
        for idx in top_candidate_idxs:
            # Build feature vector for each candidate passage
            stmt_text = " ".join(stmt_tokens)
            stmt_vec = self.tfidf.transform([stmt_text])[0]
            passage_vec = self.passage_tfidf[idx]
            cos_sim = cosine_similarity(stmt_vec, passage_vec)[0, 0]

            stmt_set = set(stmt_tokens)
            passage_tokens = self.passage_token_lists[idx]
            passage_set = set(passage_tokens)
            jaccard = (
                (len(stmt_set & passage_set) / len(stmt_set | passage_set))
                if (stmt_set | passage_set)
                else 0.0
            )

            feat = [
                bm25_scores[idx],
                cos_sim,
                jaccard,
                len(stmt_tokens),
                len(passage_tokens),
            ]
            feature_matrix.append(feat)

        feature_matrix = np.array(feature_matrix, dtype=float)

        if use_ltr and self.ranker is not None:
            scaled = self.scaler.transform(feature_matrix)
            df = pd.DataFrame(scaled, columns=FEATURE_NAMES)
            scores = self.ranker.predict(df)
            order = np.argsort(scores)[::-1]
            sorted_idx = top_candidate_idxs[order]
            results = []
            for rank_pos, passage_idx in enumerate(sorted_idx[:top_k]):
                results.append(self.passages[passage_idx])
            return results
        else:
            sorted_idx = top_candidate_idxs[:top_k]
            return [self.passages[i] for i in sorted_idx]


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and save Passage-LTR retriever."
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory to serialize retriever.",
    )
    parser.add_argument(
        "--train_source", type=str, default="synthetic", help="data/<source>/statements"
    )
    parser.add_argument(
        "--val_source", type=str, default="train", help="data/<source>/statements"
    )
    parser.add_argument("--passage_size", type=int, default=100)
    parser.add_argument("--passage_overlap", type=int, default=25)
    parser.add_argument("--use_section_splitting", action="store_true", default=True)
    args = parser.parse_args()

    cfg = PassageLTRConfig(
        train_source=args.train_source,
        val_source=args.val_source,
        k1=2.5,
        b=1.0,
        normalize=False,
        remove_stopwords=True,
        use_stemming=False,
        negative_per_query=5,
        random_state=42,
        early_stop_frac=0.1,
        passage_size=args.passage_size,
        passage_overlap=args.passage_overlap,
        use_section_splitting=args.use_section_splitting,
        candidate_pool=100,
    )

    retriever = PassageLTRRetriever(cfg)
    retriever.fit()
    retriever.save(args.out)
