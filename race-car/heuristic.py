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
logger.setLevel(logging.CRITICAL)

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
partial_lane_change_thresholds = safe_lane_change_distances(224, 1.15)


def find_safest_side(sensors: dict[str, float | None]) -> str | None:
    """
    Returns "left" or "right" if that side has clearly more clearance,
    otherwise None.
    min_gap: minimum required clearance on a side to consider it.
    hysteresis: relative difference required to prefer one side over the other.
    """
    # Side sensors (perpendicular to car direction) - safe at 600
    left_side_keys = [
        "left_side",
        "left_side_back",
        "left_side_front",
        "back_left_back",
        "front_left_front",
        "left_back",
        "left_front",
    ]
    right_side_keys = [
        "right_side",
        "right_side_back",
        "right_side_front",
        "back_right_back",
        "front_right_front",
        "right_back",
        "right_front",
    ]

    def check_side_safety(side_keys):
        # Check all sensors for this side using individual thresholds
        all_sensors = side_keys
        sensor_status = {}

        for k in all_sensors:
            v = sensors.get(k, 1000) or 1000
            threshold = lane_change_thresholds.get(k, float("inf"))
            is_safe = v >= threshold
            sensor_status[k] = (v, threshold, is_safe)

        # Check if all sensors are safe
        all_safe = all(status[2] for status in sensor_status.values())

        # Get minimum values for comparison
        min_side = (
            min(sensor_status[k][0] for k in side_keys) if side_keys else float("inf")
        )

        return all_safe, min_side

    left_safe, left_min = check_side_safety(left_side_keys)
    right_safe, right_min = check_side_safety(right_side_keys)

    # Neither side is safe
    if not left_safe and not right_safe:
        logger.info("Neither side is safe for lane change")
        return None

    # Only one side is safe
    if left_safe and not right_safe:
        logger.info("Only left side is safe")
        return "left"
    if right_safe and not left_safe:
        logger.info("Only right side is safe")
        return "right"

    # Both sides are safe - prefer the one with more clearance
    # Use the minimum of side and angled sensors for comparison
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
    LANE_CHANGING = "lane_changing"


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

    def decide(self, state: RaceCarPredictRequestDto) -> list[str]:

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
                # Check if we should speed match with front car
                front_car_speed = self._estimate_car_speed(
                    front, prev_front, state.velocity["x"], "front"
                )
                if self._should_match_speed(front_car_speed, front):
                    self.driving_state = DrivingState.DRIVING
                    logger.info(
                        f"Speed matching with front car during measuring: target speed {front_car_speed:.1f}"
                    )
                    return self._calculate_speed_matching_actions(
                        state.velocity["x"], front_car_speed
                    )

                dv = prev_front - front
                brake_amount = (
                    int(dv // 0.1) + 3
                )  # Car vary in speed. Be on the safe side
                if brake_amount > 0:
                    self.driving_state = DrivingState.BRAKING
                    logger.info(f"Braking by: {brake_amount}")
                    return ["DECELERATE"] * brake_amount
                else:
                    self.driving_state = DrivingState.LANE_CHANGING
                    return self._start_lane_change(state)
            elif front:
                self.driving_state = DrivingState.MEASURING
                self.last_measurement = state.sensors.copy()
                return ["DECELERATE"]
            elif back and prev_back:
                # Check if we should speed match with back car
                back_car_speed = self._estimate_car_speed(
                    back, prev_back, state.velocity["x"], "back"
                )
                if self._should_match_speed(back_car_speed, back):
                    self.driving_state = DrivingState.DRIVING
                    logger.info(
                        f"Speed matching with back car during measuring: target speed {back_car_speed:.1f}"
                    )
                    return self._calculate_speed_matching_actions(
                        state.velocity["x"], back_car_speed
                    )

                self.driving_state = DrivingState.DRIVING
                dv = prev_back - back
                if dv > 0:  # Car is getting closer
                    return self._start_lane_change(state)
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
            self.driving_state = DrivingState.LANE_CHANGING
            return self._start_lane_change(state)
        elif self.driving_state == DrivingState.LANE_CHANGING:
            return self._continue_lane_change(state)

        return ["NOTHING"]

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

    def _should_match_speed(
        self, estimated_speed: float, current_distance: float
    ) -> bool:
        """
        Determine if we should match speed with a nearby car.
        Returns True if the other car is close enough and at a significant speed.
        """
        if current_distance < self.speed_matching_distance:
            return False

        # Check if the other car's speed is significant (above threshold percentage of max speed)
        return estimated_speed >= self.max_speed * 0.9

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

    def _drive(self, state: RaceCarPredictRequestDto) -> list[str]:
        ego_speed = state.velocity["x"]
        max_actions = 20
        min_actions = 10
        threshold_speed = self.max_speed / 2  # 10 m/s if max_speed is 20

        # Check for speed matching opportunities with nearby cars
        front = state.sensors.get("front")
        back = state.sensors.get("back")
        prev_front = self.last_measurement.get("front")
        prev_back = self.last_measurement.get("back")

        # Check front car for speed matching
        if front and prev_front:
            front_car_speed = self._estimate_car_speed(
                front, prev_front, ego_speed, "front"
            )
            if self._should_match_speed(front_car_speed, front):
                logger.info(
                    f"Speed matching with front car: target speed {front_car_speed:.1f}, ego speed {ego_speed:.1f}"
                )
                return self._calculate_speed_matching_actions(
                    ego_speed, front_car_speed, max_actions
                )

        # Check back car for speed matching
        if back and prev_back:
            back_car_speed = self._estimate_car_speed(
                back, prev_back, ego_speed, "back"
            )
            if self._should_match_speed(back_car_speed, back):
                logger.info(
                    f"Speed matching with back car: target speed {back_car_speed:.1f}, ego speed {ego_speed:.1f}"
                )
                return self._calculate_speed_matching_actions(
                    ego_speed, back_car_speed, max_actions
                )

        # Normal acceleration logic if no speed matching is needed
        dv = self.max_speed - ego_speed
        needed_steps = int(dv // 0.1)

        # 0) If you're so close you need <1 tick, just stop accelerating.
        if dv <= 0.1:
            return ["NOTHING"] * max_actions

        # 1) Full throttle up to threshold
        if ego_speed < threshold_speed:
            logger.info(f"Accelerating by: {max_actions}")
            return ["ACCELERATE"] * max_actions

        # 2) Beyond threshold: compute linear taper fraction
        dv_range = self.max_speed - threshold_speed
        proportion = dv / dv_range  # goes 1→0 as speed goes 10→20

        # 3) Ceil so any fraction gives ≥1, then cap by needed_steps
        raw = max_actions * proportion
        taper_steps = min(needed_steps, math.ceil(raw))

        # 4) Floor to at least min_actions
        accelerate_amount = max(taper_steps, min_actions)

        logger.info(f"Accelerating by: {accelerate_amount}")
        return ["ACCELERATE"] * accelerate_amount

    def _switch_lane(self, state: RaceCarPredictRequestDto) -> list[str]:
        safest_side = find_safest_side(state.sensors)
        logger.info(f"Safest side: {safest_side}")
        if safest_side == "left":
            return switch_up()
        elif safest_side == "right":
            return switch_down()
        else:
            return ["NOTHING"] * 50

    def _start_lane_change(self, state: RaceCarPredictRequestDto) -> list[str]:
        """
        Start a two-step lane change process.
        First step: move partially into the target lane.
        """

        safest_side = find_safest_side(state.sensors)

        if safest_side is None:
            logger.info("No safe side found. Staying in the current lane.")
            self.driving_state = DrivingState.DRIVING
            return ["NOTHING"] * 50

        logger.info(f"Starting lane change to: {safest_side}")

        self.lane_change_target = safest_side

        # Move partially into the target lane (about 1/3 of the way)
        if safest_side == "left":
            actions = ["STEER_RIGHT"] * 16 + ["STEER_LEFT"] * 16  # Partial move up
        else:  # right
            actions = ["STEER_LEFT"] * 16 + ["STEER_RIGHT"] * 16  # Partial move down

        self.lane_change_actions_remaining = len(actions)
        self.driving_state = DrivingState.LANE_CHANGING

        return actions

    def _continue_lane_change(self, state: RaceCarPredictRequestDto) -> list[str]:
        """
        Continue the lane change process.
        Step 1: Check if the partial move is still safe
        Step 2: Either complete the lane change or abort
        """

        # Check if the target side is still safe
        if self._is_target_side_safe(state, self.lane_change_target):
            if self.lane_change_target == "left":
                actions = ["STEER_RIGHT"] * 45 + [
                    "STEER_LEFT"
                ] * 45  # Complete the move up
            else:  # right
                actions = ["STEER_LEFT"] * 45 + [
                    "STEER_RIGHT"
                ] * 45  # Complete the move down

            self.lane_change_actions_remaining = len(actions)
            logger.info(f"Completing lane change to {self.lane_change_target}")
            self.driving_state = DrivingState.DRIVING
            return actions
        else:
            # Abort the lane change
            logger.info(
                f"Lane change to {self.lane_change_target} no longer safe, aborting"
            )
            self.driving_state = DrivingState.DRIVING
            if self.lane_change_target == "left":
                return ["STEER_LEFT"] * 16 + ["STEER_RIGHT"] * 16
            else:
                return ["STEER_RIGHT"] * 16 + ["STEER_LEFT"] * 16

    def _is_target_side_safe(
        self, state: RaceCarPredictRequestDto, target_side: str
    ) -> bool:
        """
        Check if the target side is still safe for completing the lane change.
        """
        if target_side == "left":
            side_keys = [
                "left_side",
                "left_side_back",
                "left_side_front",
                "back_left_back",
                "left_back",
                "left_front",
            ]
        else:  # right
            side_keys = [
                "right_side",
                "right_side_back",
                "right_side_front",
                "back_right_back",
                "right_back",
                "right_front",
            ]

        # Check all sensors for this side using individual thresholds
        for key in side_keys:
            value = state.sensors.get(key, 1000) or 1000
            threshold = partial_lane_change_thresholds.get(key, float("inf"))
            if value < threshold:
                logger.info(
                    f"Target side {target_side} unsafe: {key} = {value:.1f} < {threshold:.1f}"
                )
                return False

        logger.info(f"Target side {target_side} is still safe")
        return True
