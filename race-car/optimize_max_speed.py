#!/usr/bin/env python3
"""
Optimize the max_speed parameter of HeuristicAgent by testing different values
across multiple seeds and analyzing performance metrics.
"""

import sys
import os
import pygame
from statistics import mean, stdev
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import src.game.core as game_core
from src.game.core import initialize_game_state, game_loop
from heuristic import HeuristicAgent


@dataclass
class TestResult:
    """Results from a single test run."""

    max_speed: float
    seed: int
    distance: float
    crashed: bool
    elapsed_ticks: int
    elapsed_time_ms: float
    avg_speed: float


def run_single_test(max_speed: float, seed: int) -> TestResult:
    """Run a single test with given max_speed and seed."""
    # Clean up any existing pygame instance
    try:
        pygame.quit()
    except:
        pass

    # Initialize pygame (required for headless mode)
    os.environ["SDL_VIDEODRIVER"] = "dummy"  # Use dummy video driver for headless
    pygame.init()

    try:
        # Initialize game state
        initialize_game_state(
            api_url="http://localhost:8000",  # Not used since we use HeuristicAgent directly
            seed_value=seed,
            sensor_removal=0,
        )

        # Check that STATE and agent are properly initialized
        if game_core.STATE is None or game_core.STATE.agent is None:
            raise Exception("STATE or agent not properly initialized")

        # Modify the agent's max_speed
        game_core.STATE.agent.max_speed = max_speed

        # Run the game loop (headless)
        # Temporarily redirect stdout to suppress game over messages
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            game_loop(verbose=False, log_actions=False, log_path="")

        # Calculate average speed
        avg_speed = (
            game_core.STATE.distance / max(game_core.STATE.ticks, 1)
            if game_core.STATE.ticks > 0
            else 0
        )

        result = TestResult(
            max_speed=max_speed,
            seed=seed,
            distance=game_core.STATE.distance,
            crashed=game_core.STATE.crashed,
            elapsed_ticks=game_core.STATE.ticks,
            elapsed_time_ms=game_core.STATE.elapsed_game_time,
            avg_speed=avg_speed,
        )

        return result

    except Exception as e:
        print(f"Error in test (max_speed={max_speed}, seed={seed}): {e}")
        return TestResult(
            max_speed=max_speed,
            seed=seed,
            distance=0,
            crashed=True,
            elapsed_ticks=0,
            elapsed_time_ms=0,
            avg_speed=0,
        )
    finally:
        pygame.quit()


def run_tests_for_speed(max_speed: float, seeds: List[int]) -> List[TestResult]:
    """Run tests for a specific max_speed across multiple seeds."""
    results = []

    print(f"Testing max_speed={max_speed}")

    for i, seed in enumerate(seeds):
        if i % 10 == 0:
            print(f"  Progress: {i+1}/{len(seeds)} seeds completed")

        result = run_single_test(max_speed, seed)
        results.append(result)

    return results


def analyze_results(results: List[TestResult]) -> Dict:
    """Analyze test results and calculate statistics."""
    if not results:
        return {}

    distances = [r.distance for r in results]
    crash_rate = sum(1 for r in results if r.crashed) / len(results)
    avg_speeds = [r.avg_speed for r in results]
    completion_rates = [1 if not r.crashed else 0 for r in results]

    # Calculate percentiles manually
    def percentile(data, p):
        """Calculate percentile without numpy"""
        n = len(data)
        if n == 0:
            return 0
        sorted_data = sorted(data)
        index = p / 100.0 * (n - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    distance_percentiles = {
        "p25": percentile(distances, 25),
        "p50": percentile(distances, 50),  # median
        "p75": percentile(distances, 75),
        "p90": percentile(distances, 90),
        "p95": percentile(distances, 95),
    }

    speed_percentiles = {
        "p25": percentile(avg_speeds, 25),
        "p50": percentile(avg_speeds, 50),
        "p75": percentile(avg_speeds, 75),
        "p90": percentile(avg_speeds, 90),
        "p95": percentile(avg_speeds, 95),
    }

    analysis = {
        "max_speed": results[0].max_speed,
        "num_runs": len(results),
        "distance": {
            "mean": mean(distances),
            "std": stdev(distances) if len(distances) > 1 else 0,
            "min": min(distances),
            "max": max(distances),
            **{f"distance_{k}": v for k, v in distance_percentiles.items()},
        },
        "avg_speed": {
            "mean": mean(avg_speeds),
            "std": stdev(avg_speeds) if len(avg_speeds) > 1 else 0,
            "min": min(avg_speeds),
            "max": max(avg_speeds),
            **{f"speed_{k}": v for k, v in speed_percentiles.items()},
        },
        "crash_rate": crash_rate,
        "completion_rate": 1 - crash_rate,
        "success_score": mean(distances) * (1 - crash_rate),  # Combined metric
    }

    return analysis


def optimize_max_speed(
    speed_range: Tuple[float, float] = (20, 50),
    speed_step: float = 2.5,
    num_seeds: int = 100,
    seed_start: int = 1,
) -> List[Dict]:
    """
    Optimize max_speed by testing different values.

    Args:
        speed_range: (min_speed, max_speed) to test
        speed_step: Step size between tested speeds
        num_seeds: Number of random seeds to test per speed
        seed_start: Starting seed value

    Returns:
        List of dictionaries with analysis results for each tested speed
    """
    # Generate test parameters
    speeds_to_test = []
    current_speed = speed_range[0]
    while current_speed <= speed_range[1]:
        speeds_to_test.append(current_speed)
        current_speed += speed_step

    seeds = list(range(seed_start, seed_start + num_seeds))

    print(f"Optimizing max_speed over range {speed_range} with step {speed_step}")
    print(f"Testing {len(speeds_to_test)} different speeds with {num_seeds} seeds each")
    print(f"Total tests: {len(speeds_to_test) * num_seeds}")
    print(f"Speeds to test: {list(speeds_to_test)}")
    print()

    all_results = []
    analyses = []

    # Test each speed
    for speed in speeds_to_test:
        speed_results = run_tests_for_speed(speed, seeds)
        all_results.extend(speed_results)

        # Analyze results for this speed
        analysis = analyze_results(speed_results)
        analyses.append(analysis)

        print(f"Completed max_speed={speed}")
        print(
            f"  Distance: {analysis['distance']['mean']:.1f} ± {analysis['distance']['std']:.1f}"
        )
        print(f"  Crash rate: {analysis['crash_rate']:.1%}")
        print(f"  Success score: {analysis['success_score']:.1f}")
        print()

    # Save detailed results as JSON
    results_data = [
        {
            "max_speed": r.max_speed,
            "seed": r.seed,
            "distance": r.distance,
            "crashed": r.crashed,
            "elapsed_ticks": r.elapsed_ticks,
            "elapsed_time_ms": r.elapsed_time_ms,
            "avg_speed": r.avg_speed,
        }
        for r in all_results
    ]

    with open("max_speed_optimization_detailed.json", "w") as f:
        json.dump(results_data, f, indent=2)

    return analyses


def main():
    """Main optimization function."""
    print("🚗 Starting max_speed optimization for HeuristicAgent")
    print("=" * 60)

    # Run optimization
    results_list = optimize_max_speed(
        speed_range=(
            10,
            40,
        ),  # Test speeds from 30 to 42.5 (focused range around current 35)
        speed_step=2.5,  # Test every 2.5 units
        num_seeds=15,  # 15 seeds per speed for good statistics
        seed_start=1,
    )

    # Save summary results
    with open("max_speed_optimization_summary.json", "w") as f:
        json.dump(results_list, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 60)

    # Sort by success score (distance * completion_rate)
    results_sorted = sorted(
        results_list, key=lambda x: x["success_score"], reverse=True
    )

    print("\nTop 5 performing speeds (by success score):")
    top_5 = results_sorted[:5]
    for row in top_5:
        print(
            f"  {row['max_speed']:5.1f}: "
            f"distance={row['distance']['mean']:6.1f}±{row['distance']['std']:5.1f}, "
            f"crash_rate={row['crash_rate']:5.1%}, "
            f"score={row['success_score']:6.1f}"
        )

    best_speed = results_sorted[0]["max_speed"]
    best_row = results_sorted[0]

    print(f"\n🏆 RECOMMENDED MAX_SPEED: {best_speed}")
    print(
        f"   Average distance: {best_row['distance']['mean']:.1f} ± {best_row['distance']['std']:.1f}"
    )
    print(f"   Crash rate: {best_row['crash_rate']:.1%}")
    print(f"   Success score: {best_row['success_score']:.1f}")
    print(
        f"   Distance percentiles: P50={best_row['distance']['distance_p50']:.1f}, "
        f"P75={best_row['distance']['distance_p75']:.1f}, "
        f"P90={best_row['distance']['distance_p90']:.1f}"
    )

    print(f"\nDetailed results saved to:")
    print(f"  - max_speed_optimization_summary.json")
    print(f"  - max_speed_optimization_detailed.json")


if __name__ == "__main__":
    main()
