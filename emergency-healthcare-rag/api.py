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
import json

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
You are a helpful medical assistant. Your task is to determine whether the following medical statement is supported by the evidence provided and predict the most relevant medical topic.

Statement:
{statement}

Retrieved Snippets:
{snippets}

Based only on the above snippets, please provide your response in the following format:
{{"statement_is_true": true/false, "statement_topic": <topic_id>}}

Determine if the statement is true or false based on the evidence, and identify the most relevant medical topic.
"""
    snippets_text = "\n\n".join(
        f"Snippet {i+1} (Topic: {s['topic_name']}, Topic ID: {s['topic_id']}):\n{s['content']}"
        for i, s in enumerate(snippets)
    )
    return prompt.format(
        statement=statement.strip(),
        snippets=snippets_text.strip(),
    )


def query_llm(prompt: str) -> str:
    response = chat(
        "deepseek-r1:32b", messages=[{"role": "user", "content": prompt}], think=False
    )
    content = response.message.content
    if not content:
        print(f"No response from LLM: {response}")
        return "{{'statement_is_true': False, 'statement_topic': -1}}"

    return content.strip()


def parse_llm_response(llm_response: str) -> tuple[bool, int]:
    """Parse LLM response to extract truth value and predicted topic"""
    # Clean the response - remove markdown code blocks if present
    cleaned_response = llm_response.strip()

    # Remove markdown code block markers if present
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]  # Remove "```json"
    if cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]  # Remove "```"
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]  # Remove trailing "```"

    cleaned_response = cleaned_response.strip()

    try:
        response = json.loads(cleaned_response)
        return response["statement_is_true"], response["statement_topic"]
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Original LLM response: {llm_response}")
        print(f"Cleaned response: {cleaned_response}")
        return False, -1


@app.post("/predict", response_model=MedicalStatementResponseDto)
def predict_llm_endpoint(request: LLMPredictionRequestDto):
    start = time.time()

    if normalize:
        request.statement = normalize_medical_text(request.statement, is_query=True)

    # Retrieval
    retrieved = retriever.invoke(request.statement)

    # Get top 5 snippets with topic information
    top_snippets = []
    for doc in retrieved[:5]:
        snippet_info = {
            "content": doc.page_content,
            "topic_name": doc.metadata.get("topic_name", "Unknown"),
            "topic_id": doc.metadata.get("topic_id", -1),
        }
        top_snippets.append(snippet_info)

    # Format prompt and query LLM
    prompt = format_prompt(request.statement, top_snippets)
    llm_response = query_llm(prompt)

    # Parse LLM response
    statement_is_true, predicted_topic_id = parse_llm_response(llm_response)

    response = MedicalStatementResponseDto(
        statement_is_true=statement_is_true,
        statement_topic=predicted_topic_id,
    )

    end = time.time()
    print(f"LLM Prediction Time: {end - start:2f}")
    return response


if __name__ == "__main__":

    uvicorn.run("api:app", host=HOST, port=PORT)
