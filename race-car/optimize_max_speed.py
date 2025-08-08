#!/usr/bin/env python3
"""
Hyperparameter tuner for HeuristicAgent (new API).

It will:
- Randomly sample candidate configs over sensible ranges
- Evaluate each over a set of seeds
- Keep the best K and do a small local refinement
- Output JSON: tuner_detailed.json, tuner_summary.json
- Print the top results

Integration:
1) If your old loop exists (src.game.core.initialize_game_state/game_loop),
   this script will use it automatically.

2) Otherwise, implement the two adapter functions below:
   - adapter_reset(seed: int, agent: HeuristicAgent) -> None
   - adapter_run() -> dict   # returns metrics: distance, crashed, ticks, elapsed_time_ms (required)
                              # optionally: lane_change_aborts, ttc_violations, etc.

Refactor note:
- If you added __init__ parameters to HeuristicAgent matching the keys in CFG,
  the script will pass them. If not, it will set attributes after construction.
"""

import os, sys, json, math, random, contextlib, io, time
from statistics import mean
from dataclasses import dataclass, asdict

# --- Optional old engine support -------------------------------------------------
HAVE_OLD_ENV = False
try:
    import pygame
    import src.game.core as game_core
    from src.game.core import initialize_game_state, game_loop

    HAVE_OLD_ENV = True
except Exception:
    HAVE_OLD_ENV = False

# --- Your agent -----------------------------------------------------------------
from heuristic import HeuristicAgent  # make sure this is your *new* agent


# --- Adapter (NEW environment): fill these if you don't have old engine ----------
def adapter_reset(seed: int, agent: HeuristicAgent) -> None:
    """
    TODO: Reset your current simulator/environment with the given seed and agent.
    Example:
        env.reset(seed=seed)
        env.set_agent(agent)
    """
    raise NotImplementedError(
        "Implement adapter_reset() if old engine isn't available."
    )


def adapter_run() -> dict:
    """
    TODO: Run the episode to completion and return a metrics dict.
    REQUIRED keys:
        - distance (float)
        - crashed (bool)
        - ticks (int)
        - elapsed_time_ms (float)
    OPTIONAL keys (if you can provide them):
        - lane_change_aborts (int)
        - ttc_violations (int)
    """
    raise NotImplementedError("Implement adapter_run() if old engine isn't available.")


# --- Ranges to sample ------------------------------------------------------------
def sample_config() -> dict:
    """Random (mostly uniform) sampling of the key knobs."""
    return dict(
        base_max_speed=random.uniform(15.0, 35.0),
        speed_ramp_rate=10 ** random.uniform(-2.7, -1.5),  # ~0.002–0.032 per tick
        safety_ttc=random.uniform(1.8, 4.0),
        safety_margin=random.uniform(0.2, 0.9),
        match_tol=random.uniform(0.2, 1.2),
        max_accel_actions=random.randint(20, 90),
    )


def jitter(base: dict) -> dict:
    """Small local neighborhood around a promising config."""

    def clip(v, lo, hi):
        return max(lo, min(hi, v))

    cfg = dict(base)
    cfg["base_max_speed"] = clip(
        base["base_max_speed"] + random.uniform(-2.5, 2.5), 10, 45
    )
    cfg["speed_ramp_rate"] = clip(
        base["speed_ramp_rate"] * (1 + random.uniform(-0.4, 0.4)), 0.001, 0.06
    )
    cfg["safety_ttc"] = clip(base["safety_ttc"] + random.uniform(-0.4, 0.4), 1.2, 5.0)
    cfg["safety_margin"] = clip(
        base["safety_margin"] + random.uniform(-0.2, 0.2), 0.1, 1.5
    )
    cfg["match_tol"] = clip(base["match_tol"] + random.uniform(-0.3, 0.3), 0.1, 1.8)
    cfg["max_accel_actions"] = clip(
        base["max_accel_actions"] + random.randint(-10, 10), 10, 120
    )
    return cfg


# --- Evaluation ------------------------------------------------------------------
@dataclass
class EpisodeResult:
    seed: int
    distance: float
    crashed: bool
    ticks: int
    elapsed_time_ms: float
    extra: dict


def _make_agent(cfg: dict) -> HeuristicAgent:
    """Construct agent. If your class has __init__ params, pass them; else setattr."""
    try:
        agent = HeuristicAgent(**cfg)  # works if you refactored __init__
        return agent
    except TypeError:
        # Fallback: instantiate and set attributes
        agent = HeuristicAgent()
        for k, v in cfg.items():
            if hasattr(agent, k):
                setattr(agent, k, v)
        return agent


def run_episode(seed: int, cfg: dict) -> EpisodeResult:
    # Silence noisy sim prints
    with contextlib.redirect_stdout(io.StringIO()):
        if HAVE_OLD_ENV:
            # headless pygame setup
            try:
                import pygame

                os.environ["SDL_VIDEODRIVER"] = "dummy"
                pygame.quit()
                pygame.init()
            except Exception:
                pass

            # initialize & plug agent
            initialize_game_state(
                api_url="http://localhost:8000", seed_value=seed, sensor_removal=0
            )

            if game_core.STATE is None:
                raise RuntimeError("Game STATE not initialized")
            game_core.STATE.agent = _make_agent(cfg)

            # run loop
            game_loop(verbose=False, log_actions=False, log_path="")

            st = game_core.STATE
            extra = {}
            # Optionally pull more fields if your STATE has them
            return EpisodeResult(
                seed=seed,
                distance=float(getattr(st, "distance", 0.0)),
                crashed=bool(getattr(st, "crashed", False)),
                ticks=int(getattr(st, "ticks", 0)),
                elapsed_time_ms=float(getattr(st, "elapsed_game_time", 0.0)),
                extra=extra,
            )
        else:
            # Use your adapter
            agent = _make_agent(cfg)
            adapter_reset(seed, agent)
            metrics = adapter_run()
            required = ["distance", "crashed", "ticks", "elapsed_time_ms"]
            for k in required:
                if k not in metrics:
                    raise KeyError(f"adapter_run must return key: '{k}'")
            extra = {k: v for k, v in metrics.items() if k not in required}
            return EpisodeResult(
                seed=seed,
                distance=float(metrics["distance"]),
                crashed=bool(metrics["crashed"]),
                ticks=int(metrics["ticks"]),
                elapsed_time_ms=float(metrics["elapsed_time_ms"]),
                extra=extra,
            )


def evaluate_config(cfg: dict, seeds: list[int]) -> tuple[float, dict]:
    runs = [run_episode(s, cfg) for s in seeds]
    distances = [r.distance for r in runs]
    crashes = [r.crashed for r in runs]

    crash_rate = sum(crashes) / max(1, len(crashes))
    score = mean(distances) * (1.0 - crash_rate)

    # Optional penalties if extra metrics exist
    # e.g., ttc_violations per km, lane_change_aborts rate
    total_km = max(1e-6, sum(distances) / 1000.0)
    ttc_viol = sum(r.extra.get("ttc_violations", 0) for r in runs) / total_km
    abort_rate = sum(1 for r in runs if r.extra.get("lane_change_aborts", 0) > 0) / len(
        runs
    )
    # Tune weights if you wire these up:
    score -= 200.0 * ttc_viol
    score -= 200.0 * abort_rate

    info = dict(
        cfg=cfg,
        num_runs=len(runs),
        mean_distance=mean(distances),
        crash_rate=crash_rate,
        ttc_per_km=ttc_viol,
        abort_rate=abort_rate,
        min_distance=min(distances),
        max_distance=max(distances),
    )
    return score, info


# --- Tuner -----------------------------------------------------------------------
def tune(
    num_candidates: int = 40,
    seeds_small: int = 10,
    seeds_big: int = 30,
    top_k: int = 8,
    seed_start_small: int = 100,
    seed_start_big: int = 1000,
):
    random.seed(42)

    print(f"Coarse sweep: {num_candidates} candidates × {seeds_small} seeds")
    coarse = []
    for i in range(num_candidates):
        cfg = sample_config()
        score, info = evaluate_config(
            cfg, list(range(seed_start_small, seed_start_small + seeds_small))
        )
        info["score_small"] = score
        coarse.append(info)
        print(
            f"[{i+1:3d}/{num_candidates}] score={score:9.1f} dist={info['mean_distance']:7.1f} crash={info['crash_rate']:.1%} cfg={cfg}"
        )

    coarse.sort(key=lambda x: x["score_small"], reverse=True)
    finalists = coarse[:top_k]

    print("\nRefinement around top candidates...")
    refined = []
    for j, f in enumerate(finalists, 1):
        cfg2 = jitter(f["cfg"])
        score_big, info_big = evaluate_config(
            cfg2, list(range(seed_start_big, seed_start_big + seeds_big))
        )
        info_big["score_big"] = score_big
        refined.append(info_big)
        print(
            f"[{j:2d}/{top_k}] score_big={score_big:9.1f} dist={info_big['mean_distance']:7.1f} crash={info_big['crash_rate']:.1%}"
        )

    refined.sort(key=lambda x: x["score_big"], reverse=True)
    return coarse, refined


# --- Main ------------------------------------------------------------------------
def main():
    t0 = time.time()
    coarse, refined = tune()
    dt = time.time() - t0

    # Save results
    with open("tuner_detailed.json", "w") as f:
        json.dump(dict(coarse=coarse, refined=refined), f, indent=2)
    best = refined[0]
    summary = {
        "best_cfg": best["cfg"],
        "score_big": best["score_big"],
        "mean_distance": best["mean_distance"],
        "crash_rate": best["crash_rate"],
        "ttc_per_km": best["ttc_per_km"],
        "abort_rate": best["abort_rate"],
        "elapsed_seconds": dt,
        "note": "Score = mean(distance)*(1-crash) minus penalties if provided.",
    }
    with open("tuner_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("TOP RESULTS")
    print("=" * 60)
    print(f"Best score: {best['score_big']:.1f}")
    print(f"Mean distance: {best['mean_distance']:.1f}")
    print(f"Crash rate: {best['crash_rate']:.1%}")
    print(f"TTC/km: {best['ttc_per_km']:.2f}  Abort rate: {best['abort_rate']:.2f}")
    print("Recommended config:")
    for k, v in best["cfg"].items():
        print(f"  {k}: {v}")
    print("\nSaved: tuner_detailed.json, tuner_summary.json")


if __name__ == "__main__":
    main()
