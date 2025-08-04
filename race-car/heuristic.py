from dtos import RaceCarPredictRequestDto

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
            v = sensors.get(k, 1000)
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


class HeuristicAgent:
    def __init__(self):
        self.is_braking = False
        self.current_lane = 0
        self.max_speed = 10

    def decide(self, state: RaceCarPredictRequestDto) -> list[str]:
        ego_speed = state.velocity["x"]
        print(f"Speed: {ego_speed}")

        if state.sensors.get("front") and not self.is_braking:
            print("Braking. Distance to front: ", state.sensors.get("front"))
            self.is_braking = True
            return brake()

        if self.is_braking:
            print("Finding safest side")
            # Check which lane is safe i.e. has maximum distance to side sensors
            # If there is no safe lane, do nothing
            # If there is a safe lane, switch to it
            # If there is a safe lane, switch to it
            safest_side = find_safest_side(state.sensors, min_gap=10)
            print(f"Safest side: {safest_side}")
            if safest_side == "left":
                self.current_lane -= 1
                self.is_braking = False
                return switch_up()
            elif safest_side == "right":
                self.current_lane += 1
                self.is_braking = False
                return switch_down()
            else:
                return ["NOTHING"]

        if ego_speed < self.max_speed:
            print("Accelerating")
            return ["ACCELERATE"] * 20

        return ["NOTHING"]
