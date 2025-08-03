# ltr_retriever.py

import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics.pairwise import cosine_similarity

import rank_bm25
from langchain.schema import Document

from bm25_retriever import (
    preprocess_func,
    normalize_medical_text,
    preprocess_statements,
    load_cleaned_documents,
    load_topics,
    get_retrieval_predictions,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FEATURE_NAMES = ["bm25", "tfidf_cosine", "jaccard", "query_len", "doc_len"]


@dataclass
class LTRConfig:
    train_source: str = "synthetic"
    val_source: str = "train"
    k1: float = 1.5
    b: float = 0.75
    normalize: bool = False
    remove_stopwords: bool = False
    use_stemming: bool = False
    negative_per_query: int = 5
    random_state: int = 42
    early_stop_frac: float = 0.1  # for internal early stopping
    candidate_pool: int = 100  # number of top-BM25 docs to rerank at inference


class LTRRetriever:
    def __init__(self, config: LTRConfig):
        self.config = config
        self.bm25: Optional[rank_bm25.BM25Okapi] = None
        self.tfidf: Optional[TfidfVectorizer] = None
        self.scaler: Optional[StandardScaler] = None
        self.ranker: Optional[lgb.LGBMRanker] = None
        self.docs: List[Document] = []
        self.doc_tfidf = None  # matrix
        self.doc_texts: List[str] = []
        self.doc_token_lists: List[List[str]] = []  # for jaccard
        self.topic2id: Dict[str, int] = {}
        self.id2topic: Dict[int, str] = {}

    # ---------- internal helpers ----------
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

    def _build_tfidf(self, docs: List[Document], statements: List[List[str]]):
        self.doc_texts = [" ".join(self._prep_text(doc.page_content)) for doc in docs]
        statement_texts = [" ".join(tokens) for tokens in statements]
        self.tfidf = TfidfVectorizer()
        self.tfidf.fit(self.doc_texts + statement_texts)
        self.doc_tfidf = self.tfidf.transform(self.doc_texts)

    @staticmethod
    def _ndcg_at_k(relevances: List[int], k: int) -> float:
        def dcg(rs):
            return sum((2**r - 1) / np.log2(idx + 2) for idx, r in enumerate(rs[:k]))

        ideal = sorted(relevances, reverse=True)
        ideal_dcg = dcg(ideal)
        if ideal_dcg == 0:
            return 0.0
        return dcg(relevances) / ideal_dcg

    def _compute_feature_vector(
        self,
        stmt_tokens: List[str],
        doc_idx: int,
        bm25_scores: Optional[np.ndarray] = None,
    ) -> List[float]:
        # BM25 score
        bm25_score = 0.0
        if bm25_scores is not None:
            bm25_score = bm25_scores[doc_idx]
        else:
            bm25_score = self.bm25.get_scores(stmt_tokens)[doc_idx]

        # TF-IDF cosine
        stmt_text = " ".join(stmt_tokens)
        stmt_vec = self.tfidf.transform([stmt_text])[0]
        doc_vec = self.doc_tfidf[doc_idx]
        cos_sim = cosine_similarity(stmt_vec, doc_vec)[0, 0]

        # Jaccard
        stmt_set = set(stmt_tokens)
        doc_tokens = self.doc_token_lists[doc_idx]
        doc_set = set(doc_tokens)
        jaccard = (
            (len(stmt_set & doc_set) / len(stmt_set | doc_set))
            if (stmt_set | doc_set)
            else 0.0
        )

        return [
            bm25_score,
            cos_sim,
            jaccard,
            len(stmt_tokens),
            len(doc_tokens),
        ]

    # ---------- core methods ----------
    def fit(self) -> None:
        """
        Train the full retrieval pipeline using config.train_source / val_source.
        """
        base = Path("data")
        topics_json = base / "topics.json"
        cleaned_root = base / "cleaned_topics"

        # Load topics & docs
        self.topic2id, _ = load_topics(topics_json)
        self.id2topic = {v: k for k, v in self.topic2id.items()}
        self.docs = load_cleaned_documents(cleaned_root, self.topic2id, normalize=False)
        logger.info(f"Loaded {len(self.docs)} documents for training.")

        # Preprocess documents for BM25 and for jaccard
        self.doc_token_lists = [
            self._prep_text(doc.page_content) for doc in self.docs
        ]  # reused
        texts_processed = self.doc_token_lists  # same tokenization for BM25

        # Load & preprocess statements
        train_statements_dir = Path("data") / self.config.train_source / "statements"
        train_answers_dir = Path("data") / self.config.train_source / "answers"
        val_statements_dir = Path("data") / self.config.val_source / "statements"
        val_answers_dir = Path("data") / self.config.val_source / "answers"

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

        # BM25 baseline (optional logging)
        self.bm25 = rank_bm25.BM25Okapi(
            texts_processed, k1=self.config.k1, b=self.config.b
        )
        y_true_bm25_val, y_pred_bm25_val = get_retrieval_predictions(
            self.bm25, self.docs, processed_statements_val, true_labels_val
        )
        bm25_acc = (np.array(y_true_bm25_val) == np.array(y_pred_bm25_val)).mean()
        logger.info(f"[BM25 val accuracy] {bm25_acc:.4f}")

        # TF-IDF build (fit on train statements + docs)
        self._build_tfidf(self.docs, processed_statements_train)

        # Build training data for LTR
        rng = np.random.default_rng(self.config.random_state)
        X_rows = []
        y_rows = []
        group = []

        for stmt_tokens, true_topic in zip(
            processed_statements_train, true_labels_train
        ):
            # positive/negative pairs as in original code
            positives = [
                (di, doc)
                for di, doc in enumerate(self.docs)
                if doc.metadata.get("topic_id", -1) == true_topic
            ]
            negatives = [
                (di, doc)
                for di, doc in enumerate(self.docs)
                if doc.metadata.get("topic_id", -1) != true_topic
            ]
            if not positives:
                continue
            num_neg = min(self.config.negative_per_query, len(negatives))
            neg_indices = rng.choice(len(negatives), num_neg, replace=False)
            pairs = []
            for di, doc in positives:
                pairs.append((di, doc, 1))
            for idx in neg_indices:
                di, doc = negatives[idx]
                pairs.append((di, doc, 0))

            # OPTIMIZATION: Compute BM25 scores once for all documents
            all_bm25_scores = self.bm25.get_scores(stmt_tokens)

            # OPTIMIZATION: Transform statement to TF-IDF once
            stmt_text = " ".join(stmt_tokens)
            stmt_vec = self.tfidf.transform([stmt_text])[0]
            stmt_set = set(stmt_tokens)

            # Extract relevant document vectors for batch cosine similarity
            doc_indices = [di for di, _, _ in pairs]
            relevant_doc_vecs = self.doc_tfidf[doc_indices]

            # OPTIMIZATION: Batch compute cosine similarities
            cos_sims = cosine_similarity(stmt_vec, relevant_doc_vecs)[0]

            feature_vecs = []
            labels = []
            for idx, (di, doc, label) in enumerate(pairs):
                # BM25 score (from precomputed array)
                bm25_score = all_bm25_scores[di]

                # TF-IDF cosine (from batch computation)
                cos_sim = cos_sims[idx]

                # Jaccard (using precomputed tokens)
                doc_tokens = self.doc_token_lists[di]
                doc_set = set(doc_tokens)
                jaccard = (
                    (len(stmt_set & doc_set) / len(stmt_set | doc_set))
                    if (stmt_set | doc_set)
                    else 0.0
                )

                feats = [
                    bm25_score,
                    cos_sim,
                    jaccard,
                    len(stmt_tokens),
                    len(doc_tokens),
                ]
                feature_vecs.append(feats)
                labels.append(label)

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
        from sklearn.model_selection import train_test_split

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

        # build doc_token_lists already done; ensure they reflect config if normalized (redundant here)
        logger.info("Training complete.")

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save config
        with open(out_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        # Save topic mapping
        with open(out_dir / "topic2id.json", "w") as f:
            json.dump(self.topic2id, f)

        # Save documents (as simple JSONL)
        docs_path = out_dir / "docs.jsonl"
        with open(docs_path, "w") as f:
            for doc in self.docs:
                record = {"page_content": doc.page_content, "metadata": doc.metadata}
                f.write(json.dumps(record) + "\n")

        # Save precomputed token lists
        joblib.dump(self.doc_token_lists, out_dir / "doc_token_lists.joblib")

        # Save BM25: store tokenized corpus and parameters
        joblib.dump(
            {
                "tokenized_corpus": self.doc_token_lists,
                "k1": self.config.k1,
                "b": self.config.b,
            },
            out_dir / "bm25_state.joblib",
        )

        # Save TF-IDF, scaler, ranker
        joblib.dump(self.tfidf, out_dir / "tfidf.joblib")
        joblib.dump(self.doc_tfidf, out_dir / "doc_tfidf.joblib")
        joblib.dump(self.scaler, out_dir / "scaler.joblib")
        joblib.dump(self.ranker, out_dir / "ranker.joblib")

        logger.info(f"Saved retriever to {out_dir}")

    @classmethod
    def load(cls, model_dir: Path) -> "LTRRetriever":
        with open(model_dir / "config.json") as f:
            cfg_dict = json.load(f)
        config = LTRConfig(**cfg_dict)
        self = cls(config)

        # Load topic mapping
        with open(model_dir / "topic2id.json") as f:
            self.topic2id = json.load(f)
        self.id2topic = {v: k for k, v in self.topic2id.items()}

        # Load docs
        self.docs = []
        with open(model_dir / "docs.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                self.docs.append(
                    Document(page_content=rec["page_content"], metadata=rec["metadata"])
                )

        # Load precomputed things
        self.doc_token_lists = joblib.load(model_dir / "doc_token_lists.joblib")
        bm25_state = joblib.load(model_dir / "bm25_state.joblib")
        self.bm25 = rank_bm25.BM25Okapi(
            bm25_state["tokenized_corpus"],
            k1=bm25_state["k1"],
            b=bm25_state["b"],
        )
        self.tfidf = joblib.load(model_dir / "tfidf.joblib")
        self.doc_tfidf = joblib.load(model_dir / "doc_tfidf.joblib")
        self.scaler = joblib.load(model_dir / "scaler.joblib")
        self.ranker = joblib.load(model_dir / "ranker.joblib")
        # reconstruct doc_texts if needed
        self.doc_texts = [" ".join(toks) for toks in self.doc_token_lists]
        logger.info(f"Loaded retriever from {model_dir}")
        return self

    # ---------- inference ----------
    def retrieve(
        self, raw_statement: str, top_k: int = 1, use_ltr: bool = True
    ) -> List[Tuple[Document, float, Optional[float]]]:
        """
        Returns list of (Document, bm25_score, ltr_score) sorted by final score.
        If use_ltr=False, returns top-k by BM25 only (ltr_score=None).
        """
        stmt_tokens = self._prep_text(raw_statement)
        if not stmt_tokens:
            return []

        # BM25 scores over all docs
        bm25_scores = self.bm25.get_scores(stmt_tokens)
        # get candidate pool
        top_candidate_idxs = np.argsort(bm25_scores)[::-1][: self.config.candidate_pool]
        candidates: List[Any] = []
        feature_matrix: List[List[float]] = []
        for idx in top_candidate_idxs:
            feats = self._compute_feature_vector(
                stmt_tokens, idx, bm25_scores=bm25_scores
            )
            feature_matrix.append(feats)
        feature_matrix = np.array(feature_matrix, dtype=float)
        if use_ltr and self.ranker is not None:
            scaled = self.scaler.transform(feature_matrix)
            df = pd.DataFrame(scaled, columns=FEATURE_NAMES)
            scores = self.ranker.predict(df)
            order = np.argsort(scores)[::-1]
            sorted_idx = top_candidate_idxs[order]
            results = []
            for rank_pos, doc_idx in enumerate(sorted_idx[:top_k]):
                results.append(
                    (
                        self.docs[doc_idx],
                        float(bm25_scores[doc_idx]),
                        float(scores[order[rank_pos]]),
                    )
                )
            return results
        else:
            # BM25 only
            sorted_idx = top_candidate_idxs[:top_k]
            return [(self.docs[i], float(bm25_scores[i]), None) for i in sorted_idx]


# Optional CLI for training & saving
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and save LTR retriever.")
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
    args = parser.parse_args()

    cfg = LTRConfig(
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
        candidate_pool=100,
    )
    retriever = LTRRetriever(cfg)
    retriever.fit()
    retriever.save(args.out)
