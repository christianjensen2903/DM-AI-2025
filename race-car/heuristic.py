from dtos import RaceCarPredictRequestDto
from enum import Enum
import math
import logging

# Configure logging - can be disabled by setting level to logging.CRITICAL or higher
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# To disable logging, uncomment the line below:
# logger.setLevel(logging.CRITICAL)

SENSOR_ANGLES = {
    0.0: "left_side",
    22.5: "left_side_front",
    45.0: "left_front",
    67.5: "front_left_front",
    90.0: "front",
    112.5: "front_right_front",
    135.0: "right_front",
    157.5: "right_side_front",
    180.0: "right_side",
    202.5: "right_side_back",
    225.0: "right_back",
    247.5: "back_right_back",
    270.0: "back",
    292.5: "back_left_back",
    315.0: "left_back",
    337.5: "left_side_back",
}

ACTIONS = ["NOTHING", "ACCELERATE", "DECELERATE", "STEER_RIGHT", "STEER_LEFT"]


def switch_up():
    return ["STEER_RIGHT"] * 47 + ["STEER_LEFT"] * 47


def switch_down():
    return ["STEER_LEFT"] * 47 + ["STEER_RIGHT"] * 47


def brake():
    return ["DECELERATE"] * 47


def safe_lane_change_distances(
    lane_width: float, factor: float = 1.5
) -> dict[str, float]:
    """
    Minimum clearance each sensor must report for the adjacent lane to be empty.

    lane_width : width of one lane (same units as sensor range)
    factor     : lateral clearance needed (1 lane to centre + 0.5 lane to far edge = 1.5 by default)

    Returns {sensor_name: distance_or_inf}
    """
    out = {}
    for angle, name in SENSOR_ANGLES.items():
        cos_theta = math.cos(math.radians(angle))
        out[name] = (
            math.inf
            if abs(cos_theta) < 1e-12
            else round(factor * lane_width / abs(cos_theta), 2)
        )
    return out


lane_change_thresholds = safe_lane_change_distances(224, 1.45)


def find_safest_side(
    sensors: dict[str, float | None], required_quorum: int = 4
) -> str | None:
    """
    Returns "left" or "right" if that side has enough laterally-relevant sensors clear,
    otherwise None.

    required_quorum: minimum number of lateral sensors on a side that must meet their
                     individual thresholds to consider the side safe.
    """
    left_side_keys = [
        "left_side",
        "left_side_back",
        "left_side_front",
        "back_left_back",
        "left_back",
        "left_front",
        "front_left_front",
    ]
    right_side_keys = [
        "right_side",
        "right_side_back",
        "right_side_front",
        "back_right_back",
        "right_back",
        "right_front",
        "front_right_front",
    ]

    def evaluate_side(side_keys):
        safe_count = 0
        considered = []
        min_measured = float("inf")

        for k in side_keys:
            v = sensors.get(k, 1000) or 1000
            threshold = lane_change_thresholds.get(k, float("inf"))

            # Skip non-lateral sensors (e.g., front/back facing where threshold is inf)
            if not math.isfinite(threshold):
                continue

            is_safe = v >= threshold
            if is_safe:
                safe_count += 1
            considered.append((k, v, threshold, is_safe))
            if v < min_measured:
                min_measured = v

        # No lateral sensors considered? Then side can't be declared safe.
        if not considered:
            return False, float("-inf")

        side_safe = safe_count >= required_quorum
        return side_safe, min_measured

    left_safe, left_min = evaluate_side(left_side_keys)
    right_safe, right_min = evaluate_side(right_side_keys)

    if not left_safe and not right_safe:
        # Too chatty at INFO; reduce noise
        logger.debug("Neither side is safe for lane change")
        return None

    if left_safe and not right_safe:
        logger.info("Only left side is safe")
        return "left"
    if right_safe and not left_safe:
        logger.info("Only right side is safe")
        return "right"

    # Both sides safe → pick the one with more clearance (higher minimum reading)
    if left_min > right_min:
        logger.info(
            f"Both sides safe, preferring left (left: {left_min:.1f}, right: {right_min:.1f})"
        )
        return "left"
    else:
        logger.info(
            f"Both sides safe, preferring right (left: {left_min:.1f}, right: {right_min:.1f})"
        )
        return "right"


class DrivingState(Enum):
    DRIVING = "driving"
    BRAKING = "braking"
    MEASURING = "measuring"


class HeuristicAgent:
    def __init__(self):
        self.has_braked = False
        self.last_measurement: dict[str, float | None] = {}
        self.max_speed = 32.5
        self.driving_state = DrivingState.DRIVING
        self.speed_match_threshold = 30
        self.speed_matching_distance = 100
        # Lane change state tracking
        self.lane_change_target = None  # "left" or "right"
        self.LANE_CHANGE_TICKS = len(switch_up())
        self.DECEL_PER_STEP = 0.1  # your action granularity
        self.HEADWAY_SEC = 2.0  # ~2-second rule
        self.MIN_HEADWAY = 20.0  # meters

        self.ticks = 0
        self.base_target_speed = 10.0  # starting target speed (m/s)
        self.ramp_per_tick = 0.25

    def _current_target_speed(self) -> float:
        # Linear ramp: base + slope * ticks, but never above hard cap
        return self.base_target_speed + self.ramp_per_tick * self.ticks

    def decide(self, state: RaceCarPredictRequestDto) -> list[str]:
        self.ticks += 1

        front = state.sensors.get("front")
        prev_front = self.last_measurement.get("front")
        back = state.sensors.get("back")
        prev_back = self.last_measurement.get("back")

        self.last_measurement = {}

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

                steps, side = self._brake_steps_for_safe_lane_change(
                    front, prev_front, state.velocity["x"], state.sensors
                )

                if side is not None:
                    brake_seq = ["DECELERATE"] * steps if steps > 0 else []
                    lane_seq = self._lane_change_actions(side)
                    self.driving_state = DrivingState.DRIVING
                    logger.info(
                        f"One-shot lane change to {side}; pre-brake {steps} steps; total actions {len(brake_seq)+len(lane_seq)}"
                    )
                    return brake_seq + lane_seq

                # No safe side yet → be conservative this tick
                dv = prev_front - front
                brake_amount = max(1, int(dv // 0.1) + 3) if dv > 0 else 1
                self.driving_state = DrivingState.DRIVING
                return ["DECELERATE"] * brake_amount

            elif front:
                self.driving_state = DrivingState.MEASURING
                self.last_measurement = state.sensors.copy()
                return ["DECELERATE"]
            elif back and prev_back:
                back_car_speed = self._estimate_car_speed(
                    back, prev_back, state.velocity["x"], "back"
                )

                # NEW: one-shot lane change (pre-accelerate just enough, then commit)
                steps, side = self._accelerate_steps_for_safe_lane_change(
                    back, prev_back, state.velocity["x"], state.sensors
                )

                if side is not None:
                    accel_seq = ["ACCELERATE"] * steps if steps > 0 else []
                    lane_seq = self._lane_change_actions(side)  # switch_up/down
                    self.driving_state = DrivingState.DRIVING
                    logger.info(
                        f"One-shot lane change due to rear car; side={side}, pre-accel={steps}, total_actions={len(accel_seq)+len(lane_seq)}"
                    )
                    return accel_seq + lane_seq

                # No safe side yet → be conservative this tick:
                # If the rear car is closing (dv>0) and we can still speed up, nudge forward.
                self.driving_state = DrivingState.DRIVING
                dv = prev_back - back
                if dv > 0:
                    nudge = max(
                        1,
                        min(
                            20,
                            math.ceil(
                                (back_car_speed - state.velocity["x"])
                                / self.DECEL_PER_STEP
                            ),
                        ),
                    )
                    logger.info(f"No safe side; conservative accel by: {nudge}")
                    return ["ACCELERATE"] * nudge
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
            # Legacy fallback: if we ever land here, finish in one go.
            self.driving_state = DrivingState.DRIVING
            if getattr(self, "lane_change_target", None):
                return self._lane_change_actions(self.lane_change_target)
            return self._drive(state)

        return ["NOTHING"]

    def _lane_change_actions(self, side: str) -> list[str]:
        return switch_up() if side == "left" else switch_down()

    def _accelerate_steps_for_safe_lane_change(
        self,
        back: float,
        prev_back: float,
        ego_speed: float,
        sensors: dict[str, float | None],
    ) -> tuple[int, str | None]:
        """
        Minimal accel steps so a rear car won't catch us before we finish the lane change.
        Returns (steps_to_accelerate, target_side). If no side safe yet -> (0, None).
        """
        side = find_safest_side(sensors)
        if side is None:
            return 0, None

        back_speed = self._estimate_car_speed(back, prev_back, ego_speed, "back")
        rel_closing = max(
            0.0, back_speed - ego_speed
        )  # how fast the rear car is closing
        safety_buffer = max(self.MIN_HEADWAY, ego_speed * self.HEADWAY_SEC)

        # Max allowed closing rate so that after LANE_CHANGE_TICKS we still have the buffer
        max_allowed_rel = max(0.0, (back - safety_buffer) / self.LANE_CHANGE_TICKS)

        if rel_closing <= max_allowed_rel:
            # Already safe to start lane change
            return 0, side

        # Increase ego speed so relative closing <= allowed
        target_ego_speed = back_speed - max_allowed_rel
        delta_v = max(0.0, target_ego_speed - ego_speed)
        steps = math.ceil(delta_v / self.DECEL_PER_STEP)  # same 0.1 step granularity
        return steps, side

    def _estimate_car_speed(
        self,
        current_distance: float,
        prev_distance: float,
        ego_speed: float,
        sensor_type: str,
    ) -> float:
        """
        Estimate the speed of another car based on sensor distance changes.
        Returns the estimated speed of the other car.
        """
        if prev_distance is None:
            return 0.0

        # Calculate relative velocity (positive means approaching)
        distance_change = prev_distance - current_distance

        if sensor_type == "front":
            # For front sensor: if distance decreases, other car is slower than us
            # If distance increases, other car is faster than us
            other_car_speed = ego_speed - distance_change
        else:  # back sensor
            # For back sensor: if distance decreases, other car is faster than us
            # If distance increases, other car is slower than us
            other_car_speed = ego_speed + distance_change

        return max(0.0, other_car_speed)  # Speed can't be negative

    def _brake_steps_for_safe_lane_change(
        self,
        front: float,
        prev_front: float,
        ego_speed: float,
        sensors: dict[str, float | None],
    ) -> tuple[int, str | None]:
        side = find_safest_side(sensors)
        if side is None:
            return 0, None

        front_speed = self._estimate_car_speed(front, prev_front, ego_speed, "front")
        rel_speed = max(0.0, ego_speed - front_speed)
        safety_buffer = max(self.MIN_HEADWAY, ego_speed * self.HEADWAY_SEC)

        max_allowed_rel = max(0.0, (front - safety_buffer) / self.LANE_CHANGE_TICKS)

        if rel_speed <= max_allowed_rel:
            return 0, side

        target_ego_speed = front_speed + max_allowed_rel
        delta_v = max(0.0, ego_speed - target_ego_speed)
        steps = math.ceil(delta_v / self.DECEL_PER_STEP)
        return steps, side

    def _drive(self, state: RaceCarPredictRequestDto) -> list[str]:
        ego_speed = state.velocity["x"]
        print(f"Ego speed: {ego_speed}")
        max_actions = 20
        min_actions = 10

        target = self._current_target_speed()  # << use ramping target
        print(f"Target speed: {target}")

        # Normal acceleration towards the *ramping* target
        dv = target - ego_speed
        if dv <= 0.1:  # close enough for this tick
            return ["NOTHING"] * max_actions

        threshold_speed = target / 2  # keep your tapering behavior relative to target
        needed_steps = int(max(0, dv) // 0.1)

        if ego_speed < threshold_speed:
            return ["ACCELERATE"] * max_actions

        dv_range = max(0.1, target - threshold_speed)
        proportion = max(0.0, min(1.0, dv / dv_range))
        raw = max_actions * proportion
        taper_steps = min(needed_steps, math.ceil(raw))
        accelerate_amount = max(taper_steps, min_actions)
        return ["ACCELERATE"] * accelerate_amount
