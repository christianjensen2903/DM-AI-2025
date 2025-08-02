from typing import Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever

from text_normalizer import normalize_medical_text


class NormalizedBM25Retriever(BaseRetriever):
    """BM25 retriever with medical text normalization applied to queries."""

    class Config:
        """Pydantic configuration to allow extra fields."""

        extra = "allow"
        arbitrary_types_allowed = True

    def __init__(self, documents: list[Document], k: int = 4, **kwargs):
        """
        Initialize the normalized BM25 retriever.

        Args:
            documents: List of documents (should already be normalized)
            k: Number of documents to retrieve
            **kwargs: Additional arguments for the base retriever
        """
        super().__init__(**kwargs)
        # Create the underlying BM25 retriever with normalized documents
        object.__setattr__(self, "k", k)
        object.__setattr__(
            self, "bm25_retriever", BM25Retriever.from_documents(documents, k=k)
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> list[Document]:
        """
        Retrieve relevant documents with query normalization.

        Args:
            query: Query string to search for
            run_manager: Optional run manager

        Returns:
            List of relevant documents
        """
        # Normalize the query using medical text normalization
        normalized_query = normalize_medical_text(query, is_query=True)

        # Use the normalized query for retrieval
        return self.bm25_retriever.invoke(normalized_query)

    def invoke(self, input: str, config=None, **kwargs) -> list[Document]:
        """Invoke the retriever with a query."""
        return self._get_relevant_documents(input)


class NormalizedVectorRetriever(BaseRetriever):
    """Vector retriever with medical text normalization applied to queries."""

    class Config:
        """Pydantic configuration to allow extra fields."""

        extra = "allow"
        arbitrary_types_allowed = True

    def __init__(self, vectorstore: Chroma, k: int = 4, **kwargs):
        """
        Initialize the normalized vector retriever.

        Args:
            vectorstore: Chroma vectorstore (should contain normalized documents)
            k: Number of documents to retrieve
            **kwargs: Additional arguments for the base retriever
        """
        super().__init__(**kwargs)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "vectorstore", vectorstore)
        object.__setattr__(
            self, "vector_retriever", vectorstore.as_retriever(search_kwargs={"k": k})
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> list[Document]:
        """
        Retrieve relevant documents with query normalization.

        Args:
            query: Query string to search for
            run_manager: Optional run manager

        Returns:
            List of relevant documents
        """
        # Normalize the query using medical text normalization
        normalized_query = normalize_medical_text(query, is_query=True)

        # Use the normalized query for retrieval
        return self.vector_retriever.invoke(normalized_query)

    def invoke(self, input: str, config=None, **kwargs) -> list[Document]:
        """Invoke the retriever with a query."""
        return self._get_relevant_documents(input)


class NormalizedEnsembleRetriever(BaseRetriever):
    """Ensemble retriever with medical text normalization applied to queries."""

    class Config:
        """Pydantic configuration to allow extra fields."""

        extra = "allow"
        arbitrary_types_allowed = True

    def __init__(
        self,
        documents: list[Document],
        vectorstore: Chroma,
        k: int = 4,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        **kwargs
    ):
        """
        Initialize the normalized ensemble retriever.

        Args:
            documents: List of documents (should already be normalized)
            vectorstore: Chroma vectorstore (should contain normalized documents)
            k: Number of documents to retrieve from each retriever
            bm25_weight: Weight for BM25 retriever in ensemble
            vector_weight: Weight for vector retriever in ensemble
            **kwargs: Additional arguments for the base retriever
        """
        super().__init__(**kwargs)

        # Create normalized retrievers
        bm25_retriever = NormalizedBM25Retriever(documents, k=k)
        vector_retriever = NormalizedVectorRetriever(vectorstore, k=k)

        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[bm25_weight, vector_weight],
        )

        object.__setattr__(self, "k", k)
        object.__setattr__(self, "bm25_retriever", bm25_retriever)
        object.__setattr__(self, "vector_retriever", vector_retriever)
        object.__setattr__(self, "ensemble_retriever", ensemble_retriever)

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> list[Document]:
        """
        Retrieve relevant documents using ensemble approach with query normalization.

        Args:
            query: Query string to search for
            run_manager: Optional run manager

        Returns:
            List of relevant documents
        """
        # Note: Individual retrievers will handle normalization
        return self.ensemble_retriever.invoke(query)

    def invoke(self, input: str, config=None, **kwargs) -> list[Document]:
        """Invoke the retriever with a query."""
        return self._get_relevant_documents(input)


def create_normalized_bm25_retriever(
    documents: list[Document], k: int = 4
) -> NormalizedBM25Retriever:
    """
    Create a normalized BM25 retriever from documents.

    Args:
        documents: List of documents (should already be normalized)
        k: Number of documents to retrieve

    Returns:
        Normalized BM25 retriever
    """
    return NormalizedBM25Retriever(documents, k=k)


def create_normalized_ensemble_retriever(
    documents: list[Document],
    vectorstore: Chroma,
    k: int = 4,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
) -> NormalizedEnsembleRetriever:
    """
    Create a normalized ensemble retriever.

    Args:
        documents: List of documents (should already be normalized)
        vectorstore: Chroma vectorstore (should contain normalized documents)
        k: Number of documents to retrieve from each retriever
        bm25_weight: Weight for BM25 retriever in ensemble
        vector_weight: Weight for vector retriever in ensemble

    Returns:
        Normalized ensemble retriever
    """
    return NormalizedEnsembleRetriever(
        documents=documents,
        vectorstore=vectorstore,
        k=k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
    )
