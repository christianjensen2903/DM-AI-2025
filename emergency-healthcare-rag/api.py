import uvicorn
from fastapi import FastAPI
import datetime
import time
from utils import validate_prediction, load_topics
from model import predict
from loguru import logger
from pydantic import BaseModel
from pathlib import Path
from utils import load_cleaned_documents
from retrieval import build_retrievers
from text_normalizer import normalize_medical_text
from pyngrok import ngrok
from retrieval_analysis import check_top2_different

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


port = 8000

# Setup ngrok tunnel
ngrok_tunnel = ngrok.connect(port)
print()
print()
print(ngrok_tunnel.public_url)
print()
print()

# Run the server
uvicorn.run(app, port=port)
