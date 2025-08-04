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
import openai
import dotenv
from typing import List

dotenv.load_dotenv()

base = Path("data")
topics_json = base / "topics.json"
cleaned_root = base / "cleaned_topics"

topic2id, _ = load_topics(topics_json)

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


class LLMPredictionResponseDto(BaseModel):
    statement_is_true: bool
    statement_topic: int
    retrieved_snippets: List[str]
    llm_response: str


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


@app.post("/predict", response_model=MedicalStatementResponseDto)
def predict_endpoint(request: MedicalStatementRequestDto):

    start = time.time()

    if normalize:
        request.statement = normalize_medical_text(request.statement, is_query=True)

    # Retrieval
    retrieved = retriever.invoke(request.statement)
    statement_topic = retrieved[0].metadata.get("topic_id", -1)

    processed_query = retriever.preprocess_func(request.statement)
    score = max(retriever.vectorizer.get_scores(processed_query))

    statement_is_true = score >= 79.33035

    response = MedicalStatementResponseDto(
        statement_is_true=statement_is_true, statement_topic=statement_topic
    )
    end = time.time()
    print(f"Time: {end - start:2f}")
    return response


def format_prompt(statement: str, snippets: List[str]) -> str:
    prompt = """
You are a helpful medical assistant. Your task is to determine whether the following medical statement is supported by the evidence provided.

Statement:
{statement}

Retrieved Snippets:
{snippets}

Based only on the above snippets, is the statement true or false? Reply with a single word: True or False.
"""
    snippets_text = "\n\n".join(f"Snippet {i+1}:\n{s}" for i, s in enumerate(snippets))
    return prompt.format(statement=statement.strip(), snippets=snippets_text.strip())


client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="dummy")


def query_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="local", messages=[{"role": "user", "content": prompt}], temperature=0.0
    )
    content = response.choices[0].message.content
    return content.strip() if content else "False"


@app.post("/predict_llm", response_model=LLMPredictionResponseDto)
def predict_llm_endpoint(request: LLMPredictionRequestDto):
    start = time.time()

    if normalize:
        request.statement = normalize_medical_text(request.statement, is_query=True)

    # Retrieval
    retrieved = retriever.invoke(request.statement)
    statement_topic = retrieved[0].metadata.get("topic_id", -1)

    # Get top 5 snippets
    top_snippets = [doc.page_content for doc in retrieved[:5]]

    # Format prompt and query LLM
    prompt = format_prompt(request.statement, top_snippets)
    llm_response = query_llm(prompt)

    # Parse LLM response
    statement_is_true = llm_response.lower().startswith("true")

    response = LLMPredictionResponseDto(
        statement_is_true=statement_is_true,
        statement_topic=statement_topic,
        retrieved_snippets=top_snippets,
        llm_response=llm_response,
    )

    end = time.time()
    print(f"LLM Prediction Time: {end - start:2f}")
    return response


if __name__ == "__main__":

    uvicorn.run("api:app", host=HOST, port=PORT)
