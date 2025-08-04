from pydantic import BaseModel


class RaceCarPredictRequestDto(BaseModel):
    did_crash: bool
    elapsed_ticks: int
    distance: float
    velocity: dict[str, float]
    sensors: dict[str, float | None]


class RaceCarPredictResponseDto(BaseModel):
    actions: list[str]
    # 'ACCELERATE'
    # 'DECELERATE'
    # 'STEER_LEFT'
    # 'STEER_RIGHT'
    # 'NOTHING''
