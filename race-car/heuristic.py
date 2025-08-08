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


TICK_SECONDS = 0.1  # approximate sim tick
ACCEL_PER_ACTION = 0.1  # m/s gained per "ACCELERATE"
SAFETY_TTC = 3.0  # seconds: start worrying below this TTC
SAFETY_MARGIN = 0.5  # seconds margin for maneuver completion
MATCH_TOL = 0.5  # m/s: "good enough" speed match tolerance
MAX_ACCEL_ACTIONS_PER_DECISION = 50  # cap bursts


class HeuristicAgent:
    def __init__(self):
        self.last_measurement: dict[str, float | None] = {}
        # --- speed ramp config ---
        self.base_max_speed = 20.0  # starting target (m/s)
        self.speed_ramp_rate = 0.01
        self.elapsed_time = 0.0  # internal clock
        self.max_speed = self.base_max_speed
        # --------------------------
        self.driving_state = DrivingState.DRIVING
        self.lane_change_target = None  # "left" or "right"

    def _update_max_speed(self, state):
        """
        Gradually increase target max speed as time progresses.
        Tries to use state's dt if available, otherwise assumes ~0.1s per tick.
        """

        self.elapsed_time = state.elapsed_ticks
        target = self.base_max_speed + self.speed_ramp_rate * self.elapsed_time
        self.max_speed = target
        print(f"Max speed: {self.max_speed}")

    def decide(self, state: RaceCarPredictRequestDto) -> list[str]:
        self._update_max_speed(state)

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
                dv = prev_back - back
                if dv > 0:  # rear car approaching
                    # Try to resolve by speed matching first
                    match_actions = self._resolve_rear_threat(state, back, prev_back)
                    if match_actions is not None:
                        self.driving_state = DrivingState.DRIVING
                        return match_actions
                    # Not feasible to match -> lane change
                    self.driving_state = DrivingState.LANE_CHANGING
                    return self._start_lane_change(state)
                else:
                    self.driving_state = DrivingState.DRIVING
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

    def _resolve_rear_threat(
        self, state: RaceCarPredictRequestDto, back: float, prev_back: float
    ) -> list[str] | None:
        """
        If a rear car is approaching, try to match speed unless it's infeasible in time.
        Returns an action list if we should *stay in lane* and accelerate to match.
        Returns None if we should *not* try to match (caller should change lanes).
        """

        ego_speed = state.velocity["x"]
        other_speed = self._estimate_car_speed(
            current_distance=back,
            prev_distance=prev_back,
            ego_speed=ego_speed,
            sensor_type="back",
        )

        rel_speed = other_speed - ego_speed  # >0 means rear car is closing
        if rel_speed <= 0:
            return ["NOTHING"]  # it's not actually a threat

        # Time to collision
        if back <= 0:
            ttc = 0.0
        else:
            ttc = back / rel_speed

        # If TTC is healthy, no need to do anything fancy.
        if ttc > SAFETY_TTC:
            # Gentle accelerate toward max_speed but no panic
            dv = min(self.max_speed - ego_speed, rel_speed)
            if dv <= 0:
                return ["NOTHING"]
            steps = min(
                MAX_ACCEL_ACTIONS_PER_DECISION, max(10, int(dv / ACCEL_PER_ACTION))
            )
            logger.info(
                f"Rear closing but TTC OK ({ttc:.2f}s). Accelerating {steps} to reduce delta."
            )
            return ["ACCELERATE"] * steps

        # TTC is short: can we match speed before collision window closes?
        time_available = max(0.0, ttc - SAFETY_MARGIN)
        dv_needed = max(0.0, (other_speed - ego_speed) - MATCH_TOL)

        if dv_needed <= 0:
            # Already close enough in speed; hold or minor accel
            return ["NOTHING"] * 10

        # How many accelerate actions needed to close dv_needed?
        steps_needed = math.ceil(dv_needed / ACCEL_PER_ACTION)

        # How many steps can we apply within time_available?
        steps_possible = int(time_available / TICK_SECONDS)

        # Also cannot exceed our max_speed target
        final_speed_if_matched = ego_speed + steps_needed * ACCEL_PER_ACTION
        exceeds_max = final_speed_if_matched > self.max_speed + 1e-6

        if steps_needed <= steps_possible and not exceeds_max:
            steps = min(steps_needed, MAX_ACCEL_ACTIONS_PER_DECISION)
            logger.info(
                f"Rear threat TTC {ttc:.2f}s: matching speed in-lane with {steps} ACCELERATE."
            )
            return ["ACCELERATE"] * steps

        # Not feasible to match speed in time (or would exceed max_speed): caller should change lanes
        logger.info(
            f"Rear threat TTC {ttc:.2f}s: cannot match speed in time (need {steps_needed}, possible {steps_possible}, exceeds_max={exceeds_max})."
        )
        return None

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

    def _drive(self, state: RaceCarPredictRequestDto) -> list[str]:
        ego_speed = state.velocity["x"]
        max_actions = 20
        min_actions = 10
        threshold_speed = self.max_speed / 2  # 10 m/s if max_speed is 20

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
