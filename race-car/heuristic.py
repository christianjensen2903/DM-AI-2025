from dtos import RaceCarPredictRequestDto
from enum import Enum
import math

ACTIONS = ["NOTHING", "ACCELERATE", "DECELERATE", "STEER_RIGHT", "STEER_LEFT"]


def switch_up():
    return ["STEER_RIGHT"] * 47 + ["STEER_LEFT"] * 47


def switch_down():
    return ["STEER_LEFT"] * 47 + ["STEER_RIGHT"] * 47


def brake():
    return ["DECELERATE"] * 47


def find_safest_side(
    sensors: dict[str, float | None], min_gap: float = 1.0, hysteresis: float = 0.1
) -> str | None:
    """
    Returns "left" or "right" if that side has clearly more clearance,
    otherwise None.
    min_gap: minimum required clearance on a side to consider it.
    hysteresis: relative difference required to prefer one side over the other.
    """
    left_keys = [
        "back_left_back",
        "left_back",
        "left_side_back",
        "left_side",
        "left_side_front",
        "left_front",
    ]
    right_keys = [
        "back_right_back",
        "right_back",
        "right_side_back",
        "right_side",
        "right_side_front",
        "right_front",
    ]

    def min_clearance(keys):
        vals = []
        for k in keys:
            v = sensors.get(k, 1000) or 1000
            vals.append(v)

        return min(vals)

    left = min_clearance(left_keys)
    right = min_clearance(right_keys)
    print(f"Left: {left}, Right: {right}")

    # Neither side has enough clearance
    if left < min_gap and right < min_gap:
        return None

    # Prefer side with meaningfully larger clearance
    if left > right * (1 + hysteresis) and left >= min_gap:
        return "left"
    if right > left * (1 + hysteresis) and right >= min_gap:
        return "right"

    return None


class DrivingState(Enum):
    DRIVING = "driving"
    BRAKING = "braking"
    MEASURING = "measuring"


class HeuristicAgent:
    def __init__(self):
        self.has_braked = False
        self.last_measurement: dict[str, float | None] = {}
        self.current_lane = 0
        self.max_speed = 20
        self.driving_state = DrivingState.DRIVING

    def decide(self, state: RaceCarPredictRequestDto) -> list[str]:

        front = state.sensors.get("front")
        prev_front = self.last_measurement.get("front")
        back = state.sensors.get("back")
        prev_back = self.last_measurement.get("back")

        self.last_measurement = {}

        # print(f"Driving state: {self.driving_state}")

        if self.driving_state == DrivingState.DRIVING:
            if front or back:
                self.driving_state = DrivingState.MEASURING
                self.last_measurement = state.sensors.copy()
                if front:
                    return ["DECELERATE"]
                else:
                    return ["NOTHING"]
            else:
                return self._drive(state)
        elif self.driving_state == DrivingState.MEASURING:

            if front and prev_front:
                dv = prev_front - front

                brake_amount = int(dv // 0.1)
                print(f"Braking by: {brake_amount}")
                if brake_amount > 0:
                    self.driving_state = DrivingState.BRAKING
                    return ["DECELERATE"] * brake_amount
                else:
                    self.driving_state = DrivingState.DRIVING
                    return self._drive(state)
            elif front:
                self.driving_state = DrivingState.MEASURING
                self.last_measurement = state.sensors.copy()
                return ["DECELERATE"]
            elif back and prev_back:
                self.driving_state = DrivingState.DRIVING
                dv = prev_back - back
                if dv > 0:  # Car is getting closer
                    return self._switch_lane(state)
                else:
                    return self._drive(state)
            elif back:
                self.driving_state = DrivingState.MEASURING
                self.last_measurement = state.sensors.copy()
                return ["NOTHING"]
            else:
                self.driving_state = DrivingState.DRIVING
                return self._drive(state)
        elif self.driving_state == DrivingState.BRAKING:
            self.driving_state = DrivingState.DRIVING
            return self._switch_lane(state)

        return ["NOTHING"]

    def _drive(self, state: RaceCarPredictRequestDto) -> list[str]:
        ego_speed = state.velocity["x"]
        max_actions = 20
        min_actions = 10
        threshold_speed = self.max_speed / 2  # 10 m/s if max_speed is 20

        dv = self.max_speed - ego_speed
        needed_steps = int(dv // 0.1)

        # 0) If you’re so close you need <1 tick, just stop accelerating.
        if dv <= 0.1:
            return ["NOTHING"] * max_actions

        # 1) Full throttle up to threshold
        if ego_speed < threshold_speed:
            print(f"Accelerating by: {max_actions}")
            return ["ACCELERATE"] * max_actions

        # 2) Beyond threshold: compute linear taper fraction
        dv_range = self.max_speed - threshold_speed
        proportion = dv / dv_range  # goes 1→0 as speed goes 10→20

        # 3) Ceil so any fraction gives ≥1, then cap by needed_steps
        raw = max_actions * proportion
        taper_steps = min(needed_steps, math.ceil(raw))

        # 4) Floor to at least min_actions
        accelerate_amount = max(taper_steps, min_actions)

        print(f"Accelerating by: {accelerate_amount}")
        return ["ACCELERATE"] * accelerate_amount

    def _switch_lane(self, state: RaceCarPredictRequestDto) -> list[str]:
        safest_side = find_safest_side(state.sensors, min_gap=10)
        print(f"Safest side: {safest_side}")
        if safest_side == "left":
            self.current_lane -= 1
            return switch_up()
        elif safest_side == "right":
            self.current_lane += 1
            return switch_down()
        else:
            return ["NOTHING"]
