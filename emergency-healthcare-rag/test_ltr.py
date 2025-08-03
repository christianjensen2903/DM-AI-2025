from ltr_retriever import LTRRetriever
from pathlib import Path

retriever = LTRRetriever.load(Path("models"))
results = retriever.retrieve(
    "Coronary heart disease affects approximately 15.5 million people in the United States, with the American Heart Association estimating that a person experiences a heart attack every 41 seconds.",
    top_k=3,
)
for doc, bm25_score, ltr_score in results:
    print("BM25:", bm25_score, "LTR:", ltr_score)
    print(doc.metadata, doc.page_content[:200], "...\n")
