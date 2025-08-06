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
    logger.info(f"Left: {left}, Right: {right}")

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
        self.max_speed = 30
        self.driving_state = DrivingState.DRIVING
        self.speed_matching_threshold = (
            0.8  # Match speed if other car is 80% of max speed
        )
        self.speed_matching_distance = 100

    def decide(self, state: RaceCarPredictRequestDto) -> list[str]:

        front = state.sensors.get("front")
        prev_front = self.last_measurement.get("front")
        back = state.sensors.get("back")
        prev_back = self.last_measurement.get("back")

        self.last_measurement = {}

        logger.debug(f"Driving state: {self.driving_state}")

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
                brake_amount = int(dv // 0.1)
                logger.info(f"Braking by: {brake_amount}")
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
        speed_threshold = self.max_speed * self.speed_matching_threshold
        return estimated_speed >= speed_threshold

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
        safest_side = find_safest_side(state.sensors, min_gap=10)
        logger.info(f"Safest side: {safest_side}")
        if safest_side == "left":
            self.current_lane -= 1
            return switch_up()
        elif safest_side == "right":
            self.current_lane += 1
            return switch_down()
        else:
            return ["NOTHING"]
