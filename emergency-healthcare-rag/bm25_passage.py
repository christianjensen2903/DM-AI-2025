import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.metrics import accuracy_score
import rank_bm25

# You need to have langchain_core installed and your utils / normalizer available.
from langchain_core.documents import Document

from utils import load_topics, load_cleaned_documents
from text_normalizer import normalize_medical_text

# Ensure required NLTK data is available (idempotent)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


# --- reusable preprocessing helpers with caching to avoid re-instantiating ---
_STOP_WORDS = set(stopwords.words("english"))
_STEMMER = PorterStemmer()


def preprocess_text(
    text: str, remove_stopwords: bool = False, use_stemming: bool = False
) -> List[str]:
    tokens = word_tokenize(text.lower())

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOP_WORDS]

    if use_stemming:
        tokens = [_STEMMER.stem(t) for t in tokens]

    return tokens


def split_text_to_passages(
    text: str, passage_size: int = 200, overlap: int = 50
) -> List[str]:
    if passage_size <= 0:
        raise ValueError("passage_size must be positive")
    if not (0 <= overlap < passage_size):
        raise ValueError("overlap must be >= 0 and < passage_size")

    tokens = word_tokenize(text.lower())
    if not tokens:
        return []

    step = passage_size - overlap
    passages = []
    for start in range(0, len(tokens), step):
        end = start + passage_size
        chunk = tokens[start:end]
        if not chunk:
            break
        passages.append(" ".join(chunk))
        if end >= len(tokens):
            break
    return passages


class BM25PassageRetriever:
    def __init__(
        self,
        cleaned_root: Path,
        topics_json: Path,
        *,
        normalize: bool = False,
        remove_stopwords: bool = False,
        use_stemming: bool = False,
        passage_size: int = 200,
        overlap: int = 50,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Build a passage-level BM25 retriever. Only returns top-1 (k=1).

        Raises on invalid input / empty corpus.
        """
        self.normalize = normalize
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.passage_size = passage_size
        self.overlap = overlap
        self.k1 = k1
        self.b = b

        self._validate_params()

        # Load topic mapping and full documents
        self.topic2id, _ = load_topics(topics_json)
        self.full_docs = load_cleaned_documents(
            cleaned_root, self.topic2id, normalize=normalize
        )
        if not self.full_docs:
            raise RuntimeError(f"No full documents loaded from {cleaned_root}")

        # Build passage-level corpus
        self.corpus_tokenized: List[List[str]]
        self.passage_docs: List[Document]
        self._build_corpus()

        if not self.passage_docs:
            raise RuntimeError("Passage corpus is empty after splitting documents.")

        # Build BM25 retriever (BM25Plus for better scoring)
        self.retriever = rank_bm25.BM25Plus(
            corpus=self.corpus_tokenized, k1=self.k1, b=self.b
        )

    def _validate_params(self):
        if self.passage_size <= 0:
            raise ValueError("passage_size must be positive")
        if not (0 <= self.overlap < self.passage_size):
            raise ValueError("overlap must be >=0 and < passage_size")
        if self.k1 <= 0:
            raise ValueError("k1 must be positive")
        if not (0 <= self.b <= 1):
            raise ValueError("b must be between 0 and 1")

    def _build_corpus(self):
        self.corpus_tokenized = []
        self.passage_docs = []

        for doc_idx, doc in enumerate(self.full_docs):
            content = doc.page_content
            if self.normalize:
                content = normalize_medical_text(content)

            passages = split_text_to_passages(
                content, passage_size=self.passage_size, overlap=self.overlap
            )
            original_doc_id = doc.metadata.get("doc_id", f"doc_{doc_idx}")
            topic_id = doc.metadata.get("topic_id", -1)

            for p_idx, passage_text in enumerate(passages):
                tokens = preprocess_text(
                    passage_text,
                    remove_stopwords=self.remove_stopwords,
                    use_stemming=self.use_stemming,
                )
                if not tokens:
                    continue  # skip empty after preprocessing

                self.corpus_tokenized.append(tokens)

                passage_metadata: Dict[str, Any] = {
                    "topic_id": topic_id,
                    "original_doc_id": original_doc_id,
                    "passage_id": f"{original_doc_id}_passage_{p_idx}",
                }
                passage_doc = Document(
                    page_content=passage_text, metadata=passage_metadata
                )
                self.passage_docs.append(passage_doc)

    def retrieve_best_passage(
        self, statement: str
    ) -> Tuple[Optional[Document], int, float]:
        """
        Given a raw statement string, return the best matching passage (top-1),
        the predicted topic_id, and the BM25 score.

        Returns (None, -1, 0.0) if nothing could be retrieved.
        """
        if self.normalize:
            statement_proc = normalize_medical_text(statement, is_query=True)
        else:
            statement_proc = statement

        statement_tokens = preprocess_text(
            statement_proc,
            remove_stopwords=self.remove_stopwords,
            use_stemming=self.use_stemming,
        )
        if not statement_tokens:
            return None, -1, 0.0

        # BM25Plus doesn't expose a direct get_top_n for one; mimic using scores
        scores = self.retriever.get_scores(statement_tokens)  # length = num passages
        if not scores.any():
            return None, -1, 0.0

        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        best_doc = self.passage_docs[best_idx]
        predicted_topic = best_doc.metadata.get("topic_id", -1)
        return best_doc, predicted_topic, best_score

    def evaluate(
        self,
        statements_dir: Path,
        answers_dir: Path,
    ) -> Dict[str, Any]:
        """
        Evaluate on all statement_*.txt with matching answer JSONs. Returns dict with accuracy and raw lists.
        """
        processed_statement_texts = []
        true_labels = []
        stmt_paths = []

        if not statements_dir.exists() or not answers_dir.exists():
            raise FileNotFoundError(
                f"{statements_dir=} or {answers_dir=} does not exist."
            )

        for stmt_path in sorted(statements_dir.glob("statement_*.txt")):
            base = stmt_path.stem  # e.g., "statement_0001"
            answer_path = answers_dir / f"{base}.json"
            if not answer_path.exists():
                continue  # skip if no answer

            statement_text = stmt_path.read_text(encoding="utf-8")
            with answer_path.open("r", encoding="utf-8") as f:
                answer = json.load(f)
            true_topic_id = answer.get("statement_topic", -1)

            processed_statement_texts.append(statement_text)
            true_labels.append(true_topic_id)
            stmt_paths.append(stmt_path)

        if not processed_statement_texts:
            raise RuntimeError("No statements/answers found to evaluate.")

        y_true = []
        y_pred = []

        for statement_text, true_topic in zip(processed_statement_texts, true_labels):
            _, pred_topic, _ = self.retrieve_best_passage(statement_text)
            y_true.append(true_topic)
            y_pred.append(pred_topic)

        acc = accuracy_score(y_true, y_pred)
        return {
            "accuracy": acc,
            "y_true": y_true,
            "y_pred": y_pred,
            "total": len(y_true),
        }


def main():
    base = Path("data")
    topics_json = base / "topics.json"
    cleaned_root = base / "cleaned_topics"
    statements_dir = base / "train" / "statements"
    answers_dir = base / "train" / "answers"

    retriever = BM25PassageRetriever(
        cleaned_root=cleaned_root,
        topics_json=topics_json,
        normalize=True,
        remove_stopwords=False,
        use_stemming=False,
        passage_size=100,
        overlap=25,
        k1=2.5,
        b=0.25,
    )

    eval_result = retriever.evaluate(
        statements_dir=statements_dir,
        answers_dir=answers_dir,
    )
    print(
        f"Evaluation accuracy (k=1): {eval_result['accuracy']:.4f} over {eval_result['total']} statements"
    )


if __name__ == "__main__":
    main()
