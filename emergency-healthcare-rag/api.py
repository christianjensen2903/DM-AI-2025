import uvicorn
from fastapi import FastAPI
import datetime
import time
from utils import load_topics
from pydantic import BaseModel
from pathlib import Path
from utils import load_cleaned_documents
from retrieval import build_retrievers
from text_normalizer import normalize_medical_text
import dotenv
from typing import List, Dict
from ollama import chat

dotenv.load_dotenv()

base = Path("data")
topics_json = base / "topics.json"
cleaned_root = base / "cleaned_topics"

topic2id, id2topic = load_topics(topics_json)

normalize = True

documents = load_cleaned_documents(cleaned_root, topic2id, normalize=normalize)

retriever = build_retrievers(documents)

HOST = "0.0.0.0"
PORT = 8000


class MedicalStatementRequestDto(BaseModel):
    statement: str


class MedicalStatementResponseDto(BaseModel):
    statement_is_true: int
    statement_topic: int


class LLMPredictionRequestDto(BaseModel):
    statement: str


app = FastAPI()
start_time = time.time()


@app.get("/api")
def hello():
    return {
        "service": "emergency-healthcare-rag",
        "uptime": "{}".format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get("/")
def index():
    return "Your endpoint is running!"


def format_prompt(statement: str, snippets: List[Dict]) -> str:

    prompt = """
You are a helpful medical assistant. Your task is to determine whether the following medical statement is supported by the evidence provided or not.

Statement:
{statement}

Retrieved Snippets:
{snippets}

Based only on the above snippets, please answer either True or False and nothing else.
Only say true if it directly says it in one of the snippets.
"""
    snippets_text = "\n\n".join(
        f"Snippet {i+1}:\n{s['content']}" for i, s in enumerate(snippets)
    )
    return prompt.format(
        statement=statement.strip(),
        snippets=snippets_text.strip(),
    )


def query_llm(prompt: str) -> str:
    response = chat(
        "qwen3:32b",
        messages=[{"role": "user", "content": prompt}],
        think=False,
        options={"temperature": 0},
    )
    content = response.message.content
    if not content:
        print(f"No response from LLM: {response}")
        return "False"

    return content.strip()


def parse_llm_response(llm_response: str) -> bool:
    """Parse LLM response to extract truth value and predicted topic"""
    # Clean the response - remove markdown code blocks if present
    cleaned_response = llm_response.strip()
    return cleaned_response.lower() == "true"


@app.post("/predict", response_model=MedicalStatementResponseDto)
def predict_llm_endpoint(request: LLMPredictionRequestDto):
    start = time.time()

    if normalize:
        request.statement = normalize_medical_text(request.statement, is_query=True)

    # Retrieval
    retrieved = retriever.invoke(request.statement)

    # Get top 5 snippets with topic information
    top_snippets = []
    for doc in retrieved[:10]:
        snippet_info = {
            "content": doc.metadata.get("original_content", doc.page_content),
            "topic_name": doc.metadata.get("topic_name", "Unknown"),
            "topic_id": doc.metadata.get("topic_id", -1),
        }
        top_snippets.append(snippet_info)

    # Format prompt and query LLM
    prompt = format_prompt(request.statement, top_snippets)
    llm_response = query_llm(prompt)

    # Parse LLM response
    statement_is_true = parse_llm_response(llm_response)

    # Get predicted topic (topic of the first snippet)
    predicted_topic_id = top_snippets[0]["topic_id"]

    response = MedicalStatementResponseDto(
        statement_is_true=statement_is_true,
        statement_topic=predicted_topic_id,
    )

    end = time.time()
    print(f"LLM Prediction Time: {end - start:2f}")
    return response


if __name__ == "__main__":

    uvicorn.run("api:app", host=HOST, port=PORT)
