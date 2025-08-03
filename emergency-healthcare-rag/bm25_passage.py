import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import Counter

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

# Ensure required NLTK data is available
nltk.download("punkt")
nltk.download("stopwords")


def preprocess_func(
    text: str, remove_stopwords: bool = False, use_stemming: bool = False
) -> List[str]:
    """
    Preprocess text by tokenizing and optionally removing stopwords and stemming.

    Args:
        text: Input text to preprocess
        remove_stopwords: Whether to remove English stopwords
        use_stemming: Whether to apply Porter stemming

    Returns:
        List of processed tokens
    """
    tokens = word_tokenize(text.lower())

    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words]

    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

    return tokens


def split_text_to_passages(
    text: str, passage_size: int = 200, overlap: int = 50
) -> List[str]:
    """
    Split a text into overlapping passages based on word tokens.

    Args:
        text: Full document text.
        passage_size: Number of tokens per passage.
        overlap: Number of tokens that overlap between consecutive passages.

    Returns:
        List of passage strings.
    """
    tokens = word_tokenize(text.lower())
    if passage_size <= 0:
        raise ValueError("passage_size must be positive")
    if overlap < 0 or overlap >= passage_size:
        raise ValueError("overlap must be >=0 and < passage_size")

    step = passage_size - overlap
    passages = []
    for start in range(0, max(1, len(tokens)), step):
        end = start + passage_size
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            break
        passage = " ".join(chunk_tokens)
        passages.append(passage)
        if end >= len(tokens):
            break
    return passages


def build_passage_corpus(
    docs: List[Document],
    passage_size: int = 200,
    overlap: int = 50,
    normalize: bool = False,
    remove_stopwords: bool = False,
    use_stemming: bool = False,
) -> Tuple[List[List[str]], List[Document]]:
    """
    From full documents, create passage-level Document objects and their tokenized representations.

    Returns:
        corpus_tokenized: list of token lists (one per passage)
        passage_docs: list of Document objects with updated metadata for passages
    """
    corpus_tokenized: List[List[str]] = []
    passage_docs: List[Document] = []

    for doc_idx, doc in enumerate(docs):
        content = doc.page_content
        if normalize:
            content = normalize_medical_text(content)

        passages = split_text_to_passages(
            content, passage_size=passage_size, overlap=overlap
        )
        original_doc_id = doc.metadata.get("doc_id", f"doc_{doc_idx}")
        topic_id = doc.metadata.get("topic_id", -1)

        for p_idx, passage_text in enumerate(passages):
            tokens = preprocess_func(
                passage_text,
                remove_stopwords=remove_stopwords,
                use_stemming=use_stemming,
            )
            corpus_tokenized.append(tokens)

            # Build passage-level metadata
            passage_metadata: Dict[str, Any] = {
                "topic_id": topic_id,
                "original_doc_id": original_doc_id,
                "passage_id": f"{original_doc_id}_passage_{p_idx}",
            }
            # Optionally carry over other metadata fields if needed
            passage_doc = Document(page_content=passage_text, metadata=passage_metadata)
            passage_docs.append(passage_doc)

    return corpus_tokenized, passage_docs


def preprocess_statements(
    statements_dir: Path,
    answers_dir: Path,
    normalize: bool = False,
    remove_stopwords: bool = True,
    use_stemming: bool = True,
) -> Tuple[List[List[str]], List[int], List[Path]]:
    """
    Pre-process all statements to avoid redundant processing during parameter tuning.

    Returns:
        Tuple of (processed_statements, true_labels, statement_paths)
    """
    processed_statements = []
    true_labels: List[int] = []
    statement_paths = []

    for stmt_path in sorted(statements_dir.glob("statement_*.txt")):
        base = stmt_path.stem  # e.g., "statement_0001"
        answer_path = answers_dir / f"{base}.json"
        if not answer_path.exists():
            continue

        statement_text = stmt_path.read_text(encoding="utf-8")
        with answer_path.open("r", encoding="utf-8") as f:
            answer = json.load(f)
        true_topic_id = answer.get("statement_topic", -1)

        if normalize:
            statement_text = normalize_medical_text(statement_text, is_query=True)

        statement_text_processed = preprocess_func(
            statement_text,
            remove_stopwords=remove_stopwords,
            use_stemming=use_stemming,
        )

        processed_statements.append(statement_text_processed)
        true_labels.append(true_topic_id)
        statement_paths.append(stmt_path)

    return processed_statements, true_labels, statement_paths


def get_retrieval_predictions(
    retriever: rank_bm25.BM25Okapi,
    passage_docs: List[Document],
    processed_statements: List[List[str]],
    true_labels: List[int],
    n: int = 1,
) -> Tuple[List[int], List[int]]:
    """
    Get retrieval predictions using the specified BM25 retriever over passages.

    Implements majority vote over top-n retrieved passages.

    Returns:
        Tuple of (y_true, y_pred)
    """
    y_true = []
    y_pred = []

    for statement_tokens, true_topic_id in zip(processed_statements, true_labels):
        retrieved = retriever.get_top_n(statement_tokens, passage_docs, n=n)
        if not retrieved:
            y_true.append(true_topic_id)
            y_pred.append(-1)
            continue

        # Collect the topic_ids from top-n passages
        top_topic_ids = [
            doc.metadata.get("topic_id", -1)
            for doc in retrieved
            if doc.metadata.get("topic_id", -1) != -1
        ]
        if not top_topic_ids:
            pred_topic_id = -1
        elif len(top_topic_ids) == 1 or n == 1:
            pred_topic_id = top_topic_ids[0]
        else:
            counts = Counter(top_topic_ids)
            most_common = counts.most_common()
            # If tie, pick the topic_id of the top-1 retrieved
            if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
                pred_topic_id = retrieved[0].metadata.get("topic_id", -1)
            else:
                pred_topic_id = most_common[0][0]

        y_true.append(true_topic_id)
        y_pred.append(pred_topic_id)

    return y_true, y_pred


def evaluate_bm25_config_on_passages(
    k1: float,
    b: float,
    docs: List[Document],
    texts_processed_passages: List[List[str]],
    passage_docs: List[Document],
    processed_statements: List[List[str]],
    true_labels: List[int],
    top_n: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate BM25 on passage-level corpus.

    Returns evaluation dictionary including accuracy and predictions.
    """
    retriever = rank_bm25.BM25Okapi(corpus=texts_processed_passages, k1=k1, b=b)

    y_true, y_pred = get_retrieval_predictions(
        retriever, passage_docs, processed_statements, true_labels, n=top_n
    )

    accuracy = accuracy_score(y_true, y_pred)

    return {
        "k1": k1,
        "b": b,
        "top_n": top_n,
        "accuracy": accuracy,
        "total_evaluated": len(y_true),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def test_bm25_params_on_passages(
    k1: float = 1.5,
    b: float = 0.75,
    normalize: bool = False,
    remove_stopwords: bool = False,
    use_stemming: bool = False,
    passage_size: int = 200,
    overlap: int = 50,
    top_n: int = 1,
):
    """Test BM25 over passages with specific parameters and preprocessing."""
    base = Path("data")
    topics_json = base / "topics.json"
    cleaned_root = base / "cleaned_topics"
    statements_dir = base / "synthetic" / "statements"
    answers_dir = base / "synthetic" / "answers"

    # Load topic mapping and full documents
    print("Loading data...")
    topic2id, _ = load_topics(topics_json)
    full_docs = load_cleaned_documents(cleaned_root, topic2id, normalize=False)
    print(f"Loaded {len(full_docs)} full documents")

    # Build passage-level corpus
    texts_processed_passages, passage_docs = build_passage_corpus(
        full_docs,
        passage_size=passage_size,
        overlap=overlap,
        normalize=normalize,
        remove_stopwords=remove_stopwords,
        use_stemming=use_stemming,
    )
    print(f"Built {len(passage_docs)} passages from documents")

    processed_statements, true_labels, _ = preprocess_statements(
        statements_dir, answers_dir, normalize, remove_stopwords, use_stemming
    )

    result = evaluate_bm25_config_on_passages(
        k1,
        b,
        passage_docs,
        texts_processed_passages,
        passage_docs,
        processed_statements,
        true_labels,
        top_n=top_n,
    )
    print(f"Accuracy over passages (top_n={top_n}): {result['accuracy']:.4f}")
    return result


def main():
    # Example run: passage size 250 tokens with 50-token overlap, using stopword removal
    test_bm25_params_on_passages(
        k1=2.5,
        b=1.0,
        normalize=True,
        remove_stopwords=False,
        use_stemming=False,
        passage_size=250,
        overlap=50,
        top_n=3,  # majority vote over top 3 passages
    )


if __name__ == "__main__":
    main()
