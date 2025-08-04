import time
import uvicorn
import datetime
from fastapi import Body, FastAPI
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
from pyngrok import ngrok
from heuristic import HeuristicAgent

app = FastAPI()
start_time = time.time()

agent = HeuristicAgent()


@app.post("/predict", response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):

    actions = agent.decide(request)

    return RaceCarPredictResponseDto(actions=actions)


@app.get("/api")
def hello():
    return {
        "service": "race-car-usecase",
        "uptime": "{}".format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get("/")
def index():
    return "Your endpoint is running!"


HOST = "0.0.0.0"
PORT = 9052

# Setup ngrok tunnel
# ngrok_tunnel = ngrok.connect(PORT)
# print()
# print()
# print(ngrok_tunnel.public_url)
# print()
# print()

# if __name__ == "__main__":

#     uvicorn.run("api:app", host=HOST, port=PORT)
