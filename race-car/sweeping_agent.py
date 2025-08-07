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


def turn_right(amount: int):
    return ["STEER_LEFT"] * amount + ["STEER_RIGHT"] * amount


def turn_left(amount: int):
    return ["STEER_RIGHT"] * amount + ["STEER_LEFT"] * amount


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


def safe_lane_change_distances(
    distance: float,
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
            math.inf if abs(cos_theta) < 1e-12 else round(distance / abs(cos_theta), 2)
        )
    return out


# lane_change_thresholds = safe_lane_change_distances(224, 1.45)
same_lane_thresholds = safe_lane_change_distances(100)
short_lane_threshold = safe_lane_change_distances(230)
long_lane_threshold = safe_lane_change_distances(500)

left_side_keys = [
    "left_side",
    "left_side_back",
    "left_side_front",
    "back_left_back",
    "left_back",
    "left_front",
]
right_side_keys = [
    "right_side",
    "right_side_back",
    "right_side_front",
    "back_right_back",
    "right_back",
    "right_front",
]


def check_side_safety(
    sensors: dict[str, float | None], side_keys: list[str], thresholds: dict[str, float]
) -> bool:
    # Check all sensors for this side using individual thresholds
    sensor_status = {}

    for k in side_keys:
        v = sensors.get(k, 1000) or 1000
        threshold = thresholds.get(k, float("inf"))
        is_safe = v >= threshold
        sensor_status[k] = is_safe

    # Check if all sensors are safe
    return all(status for status in sensor_status.values())


def find_safest_side(sensors: dict[str, float | None]) -> str | None:
    left_side_min = min(sensors[k] or 1000 for k in left_side_keys)
    right_side_min = min(sensors[k] or 1000 for k in right_side_keys)

    logger.info(
        f"Side clearances - left: {left_side_min:.1f}, right: {right_side_min:.1f}"
    )
    if left_side_min < right_side_min:
        return "left"
    else:
        return "right"


class AgentState(Enum):
    ACCELERATING = "accelerating"
    SWEEPING = "sweeping"
    DONE = "done"
    MEASURING_SPEED = "measuring_speed"
    MATCHING_SPEED = "matching_speed"


class SweepingAgent:
    def __init__(self):
        self.state = AgentState.ACCELERATING
        self.position = 6
        self.off_center_amount = 29
        self.center_opposite_amount = 35
        self.last_measurement: dict[str, float | None] = {}

    def decide(self, state: RaceCarPredictRequestDto) -> list[str]:
        logger.info(f"Current state: {self.state.value}, position: {self.position}")
        if self.state == AgentState.ACCELERATING:
            return self._accelerate(state)
        elif self.state == AgentState.SWEEPING:
            return self._sweep(state)
        else:
            logger.info("In default state, doing nothing")
            return ["NOTHING"] * 50

    def _sweep(self, state: RaceCarPredictRequestDto) -> list[str]:
        logger.info(f"Sweeping at position {self.position}")

        if state.sensors["back"]:
            if self.position % 3 == 1:
                logger.info("Back sensor detected, turning right")
                return turn_right(self.off_center_amount) + self._accelerate(state)
            elif self.position % 3 == 2:
                logger.info("Back sensor detected, turning left")
                return turn_left(self.off_center_amount) + self._accelerate(state)
            else:
                logger.info("Back sensor detected, continuing acceleration")
                return self._accelerate(state)

        if state.sensors["front"]:
            safest_side = find_safest_side(state.sensors)
            if self.position == 10 and safest_side == "left":
                if check_side_safety(
                    state.sensors, left_side_keys, short_lane_threshold
                ):
                    self.state = AgentState.ACCELERATING
                    return turn_left(self.center_opposite_amount)
                else:
                    return self._match_speed(state)
            elif self.position == 2 and safest_side == "right":
                if check_side_safety(
                    state.sensors, right_side_keys, short_lane_threshold
                ):
                    return turn_right(self.off_center_amount) + self._accelerate(state)
                else:
                    return self._match_speed(state)
            elif self.position == 11 and safest_side == "left":
                return turn_left(self.off_center_amount) + self._accelerate(state)
            elif self.position == 12 and safest_side == "right":
                return turn_right(self.off_center_amount) + self._accelerate(state)
            else:
                keys = left_side_keys if safest_side == "left" else right_side_keys
                if check_side_safety(state.sensors, keys, long_lane_threshold):
                    return turn_left(
                        self.off_center_amount * 2 + self.center_opposite_amount * 2
                    ) + self._accelerate(state)
                elif check_side_safety(state.sensors, keys, short_lane_threshold):
                    return turn_left(
                        self.off_center_amount + self.center_opposite_amount
                    ) + self._accelerate(state)
                else:
                    return self._match_speed(state)

        if state.sensors["front"]:
            return self._match_speed(state)

        return turn_right(self.off_center_amount)

    def _match_speed(self, state: RaceCarPredictRequestDto) -> list[str]:
        if self.state == AgentState.MEASURING_SPEED:
            front_car_speed = self._estimate_car_speed(
                state.sensors["front"] or 1000,
                self.last_measurement["front"] or 1000,
                state.velocity["x"],
                "front",
            )
            logger.info(
                f"Estimated front car speed: {front_car_speed:.1f}, ego speed: {state.velocity['x']:.1f}"
            )
            self.state = AgentState.ACCELERATING
            return self._calculate_speed_matching_actions(
                state.velocity["x"], front_car_speed
            )
        else:
            logger.info("Starting speed measurement")
            self.state = AgentState.MEASURING_SPEED
            self.last_measurement = state.sensors.copy()
            return ["BRAKE"]

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

        return max(0.0, other_car_speed)  # Speed can't be negativ

    def _calculate_speed_matching_actions(
        self, ego_speed: float, target_speed: float, max_actions: int = 20
    ) -> list[str]:
        """
        Calculate actions needed to match the speed of another car.
        """
        speed_diff = target_speed - ego_speed

        if abs(speed_diff) < 0.1:  # Already close enough
            return ["NOTHING"] * max_actions

        if speed_diff > 0:  # Need to accelerate
            needed_steps = min(max_actions, int(speed_diff // 0.1))
            return ["ACCELERATE"] * needed_steps + ["NOTHING"] * (
                max_actions - needed_steps
            )
        else:  # Need to decelerate
            needed_steps = min(max_actions, int(abs(speed_diff) // 0.1))
            return ["DECELERATE"] * needed_steps + ["NOTHING"] * (
                max_actions - needed_steps
            )

    def _start_sweeping(self, state: RaceCarPredictRequestDto) -> list[str]:
        logger.info(f"Starting sweep pattern at position {self.position}")
        self.state = AgentState.SWEEPING
        # Seek to just before the line
        self.position -= 1
        return ["STEER_RIGHT"] * 29 + ["STEER_LEFT"] * 29

    def _accelerate(self, state: RaceCarPredictRequestDto) -> list[str]:
        self.state = AgentState.ACCELERATING
        logger.info(f"Accelerating at speed {state.velocity['x']:.1f}")

        # If no back anymore
        if not state.sensors["back"]:
            logger.info("No back sensor detected, starting sweep pattern")
            return self._start_sweeping(state)

        return ["ACCELERATE"] * 50
