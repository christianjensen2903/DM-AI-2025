from src.game.core import GameState
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto

ACTIONS = ["NOTHING", "ACCELERATE", "DECELERATE", "STEER_RIGHT", "STEER_LEFT"]


class HeuristicAgent:
    def __init__(self):
        self.last_action = "NOTHING"
        self.last_steer = None

    def decide(self, state: RaceCarPredictRequestDto) -> str:
        ego_speed = state.velocity["x"]

        # Helper getters with defaults
        def get(*names, default=1000.0):
            return min([state.sensors.get(n, default) for n in names])

        front = state.sensors.get("front", 1000.0)
        left_front = get("left_front", "left_side_front", "front_left_front")
        right_front = get("right_front", "right_side_front", "front_right_front")
        left_side = state.sensors.get("left_side", 1000.0)
        right_side = state.sensors.get("right_side", 1000.0)

        # Dynamic safe distance scales with speed (so faster -> more lookahead)
        safe_front = max(
            250, ego_speed * 15
        )  # tune multiplier if velocity units change

        print(front < safe_front, safe_front, front)

        # 1. Immediate danger ahead: try to dodge
        if front < safe_front:
            # choose side with more clearance
            if right_front > left_front + 50 and right_front > 150:
                self.last_steer = "STEER_RIGHT"
                return "STEER_RIGHT"
            elif left_front > right_front + 50 and left_front > 150:
                self.last_steer = "STEER_LEFT"
                return "STEER_LEFT"
            else:
                # no good dodge path; slow down if moving fast
                if ego_speed > 5:
                    return "DECELERATE"
                else:
                    return "NOTHING"

        # 2. Lane centering: if drifted too close to side walls
        side_diff = left_side - right_side
        # if too close to left wall, steer right; vice versa
        if side_diff < -120:
            self.last_steer = "STEER_RIGHT"
            return "STEER_RIGHT"
        if side_diff > 120:
            self.last_steer = "STEER_LEFT"
            return "STEER_LEFT"

        # 3. If path ahead is clear, accelerate (but cap to avoid overshooting)
        if front > 600 and ego_speed < 25:
            return "ACCELERATE"

        # 4. Mild adjustments: prefer to keep last steering until obstacle changes
        if self.last_steer in ("STEER_LEFT", "STEER_RIGHT"):
            # if the direction is still reasonably safe, keep it for smoothing
            if self.last_steer == "STEER_LEFT" and left_front > 300:
                return "STEER_LEFT"
            if self.last_steer == "STEER_RIGHT" and right_front > 300:
                return "STEER_RIGHT"
            # else reset
            self.last_steer = None

        # 5. Default: maintain speed / nothing
        return "NOTHING"

    def next_actions(
        self, state: RaceCarPredictRequestDto, batch_size: int = 10
    ) -> list[str]:
        action = self.decide(state)
        self.last_action = action
        return [action] * batch_size
