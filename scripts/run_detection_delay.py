from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--beta", type=float, default=0.99)
    parser.add_argument("--beta-grid", default=",".join(f"{value / 100:.2f}" for value in range(55, 100, 2)))
    parser.add_argument("--theta", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--q-min", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=23)
    parser.add_argument("--pre-steps", type=int, default=100)
    parser.add_argument("--post-steps", type=int, default=2000)
    parser.add_argument("--replications", type=int, default=2000)
    parser.add_argument("--shifts", default="0.10,0.20,0.30")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lambda-grid-size", type=int, default=65)
    return parser.parse_args()


def percentile(values: np.ndarray, value: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, value))


def summarize(
    delays: np.ndarray,
    false_alarms: np.ndarray,
    threshold_name: str,
    threshold: float,
    replications: int,
    exposure: np.ndarray,
) -> dict[str, float | int | str | None]:
    valid = ~false_alarms
    detected = valid & (delays > 0)
    detected_delays = delays[detected]
    commits = exposure[detected]
    valid_count = int(valid.sum())
    return {
        "threshold_name": threshold_name,
        "threshold": threshold,
        "replications": replications,
        "valid_replications": valid_count,
        "prechange_false_alarm_rate": float(false_alarms.mean()),
        "detection_rate_by_horizon": float(detected.sum() / valid_count) if valid_count else None,
        "detected_delay_mean": float(detected_delays.mean()) if detected_delays.size else None,
        "detected_delay_median": percentile(detected_delays, 50),
        "detected_delay_p90": percentile(detected_delays, 90),
        "detected_delay_p95": percentile(detected_delays, 95),
        "simulated_prealarm_excess_risk_commits_mean": float(commits.mean()) if commits.size else None,
        "simulated_prealarm_excess_risk_commits_median": percentile(commits, 50),
        "simulated_prealarm_excess_risk_commits_p90": percentile(commits, 90),
        "simulated_prealarm_excess_risk_commits_p95": percentile(commits, 95),
        "censored_replications": int((valid & (delays == 0)).sum()),
    }


def run_replay(
    risks: np.ndarray,
    theta: float,
    alpha: float,
    grid_size: int,
    pre_steps: int,
    lambda_grid_size: int,
    q_min: float,
    postchange_mean: np.ndarray,
    fixed_index: int,
) -> tuple[dict[str, dict[str, np.ndarray]], float]:
    if q_min != 1.0:
        raise ValueError("This controlled-risk diagnostic requires full evaluation (q_min=1).")
    replications, total_steps, boundaries = risks.shape
    if boundaries != grid_size:
        raise ValueError("The replay must contain risks for every boundary in the grid.")
    lambda_cap = 1.0 / (2.0 * ((1.0 / q_min) - 1.0 + theta))
    lambda_grid = np.linspace(0.0, lambda_cap, lambda_grid_size)
    candidate_scores = np.zeros((replications, boundaries, lambda_grid_size), dtype=np.float64)
    log_monitor = np.zeros((replications, boundaries), dtype=np.float64)
    thresholds = {
        "fixed_boundary": np.log(1.0 / alpha),
        "adaptive_grid": np.log(grid_size / alpha),
    }
    state = {
        name: {
            "delays": np.zeros(replications, dtype=np.int32),
            "false_alarms": np.zeros(replications, dtype=bool),
            "exposure": np.zeros(replications, dtype=np.int32),
        }
        for name in thresholds
    }

    for step in range(total_steps):
        selected = np.argmax(candidate_scores, axis=2)
        lambda_t = lambda_grid[selected]
        centered = risks[:, step] - theta
        log_monitor += np.log1p(lambda_t * centered)
        candidate_scores += np.log1p(centered[:, :, None] * lambda_grid[None, None, :])

        for name, log_threshold in thresholds.items():
            if name == "fixed_boundary":
                crossed = log_monitor[:, fixed_index] >= log_threshold
                excess = np.full(replications, postchange_mean[fixed_index] > theta)
            else:
                feasible = log_monitor < log_threshold
                crossed = ~feasible.any(axis=1)
                chosen = np.argmax(feasible, axis=1)
                excess = postchange_mean[chosen] > theta
            if step < pre_steps:
                state[name]["false_alarms"] |= crossed
            else:
                pending = state[name]["delays"] == 0
                state[name]["exposure"] += (pending & ~crossed & excess & ~state[name]["false_alarms"]).astype(np.int32)
                new_detection = crossed & pending & ~state[name]["false_alarms"]
                state[name]["delays"][new_detection] = step - pre_steps + 1

    return state, lambda_cap


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    output_path = Path(args.output)
    payload = json.loads(source_path.read_text())
    beta_grid = sorted(float(value) for value in args.beta_grid.split(","))
    if len(beta_grid) != args.grid_size or args.beta not in beta_grid:
        raise ValueError("The fixed boundary and grid size must agree with --beta-grid.")
    beta_keys = [str(value) for value in beta_grid]
    observed = np.array(
        [
            [float(snapshot["oracle_risks"][key]) for key in beta_keys]
            for snapshot in payload["snapshots"]
            if all(key in snapshot["oracle_risks"] for key in beta_keys)
        ],
        dtype=np.float64,
    )
    if observed.size == 0:
        raise ValueError("No complete boundary-risk vectors found in the source run.")
    if not np.isfinite(observed).all() or np.any((observed < 0.0) | (observed > 1.0)):
        raise ValueError("Source risks must be finite and lie in [0, 1].")
    if observed[:, beta_grid.index(args.beta)].mean() > args.theta:
        raise ValueError("The fixed-boundary source risk must be controlled before the change.")

    shifts = [float(value) for value in args.shifts.split(",") if value.strip()]
    total_steps = args.pre_steps + args.post_steps
    rng = np.random.default_rng(args.seed)
    sampled = observed[
        rng.integers(0, len(observed), size=(args.replications, total_steps))
    ]
    rows: list[dict[str, object]] = []

    for shift in shifts:
        risks = sampled.copy()
        risks[:, args.pre_steps :] = np.minimum(
            1.0,
            risks[:, args.pre_steps :] + shift,
        )
        state, lambda_cap = run_replay(
            risks=risks,
            theta=args.theta,
            alpha=args.alpha,
            grid_size=args.grid_size,
            pre_steps=args.pre_steps,
            lambda_grid_size=args.lambda_grid_size,
            q_min=args.q_min,
            postchange_mean=np.minimum(1.0, observed + shift).mean(axis=0),
            fixed_index=beta_grid.index(args.beta),
        )
        post_mean = float(np.minimum(1.0, observed[:, beta_grid.index(args.beta)] + shift).mean())
        for name, threshold in (
            ("fixed_boundary", 1.0 / args.alpha),
            ("adaptive_grid", args.grid_size / args.alpha),
        ):
            row = summarize(
                delays=state[name]["delays"],
                false_alarms=state[name]["false_alarms"],
                threshold_name=name,
                threshold=threshold,
                replications=args.replications,
                exposure=state[name]["exposure"],
            )
            row.update(
                {
                    "source": str(source_path),
                    "model": payload.get("editor_runtime", {})
                    .get("resolved_overrides", {})
                    .get("model_name"),
                    "editor": payload.get("editor_runtime", {}).get("method", "alphaedit"),
                    "dataset": payload.get("dataset_path"),
                    "beta": args.beta,
                    "theta": args.theta,
                    "alpha": args.alpha,
                    "grid_size": args.grid_size,
                    "pre_steps": args.pre_steps,
                    "post_steps": args.post_steps,
                    "observed_risk_count": len(observed),
                    "prechange_risk_mean": float(observed[:, beta_grid.index(args.beta)].mean()),
                    "beta_grid": beta_grid,
                    "experiment_type": "joint_boundary_risk_replay",
                    "risk_shift": shift,
                    "postchange_risk_mean": post_mean,
                    "postchange_excess_risk": post_mean - args.theta,
                    "q_min": args.q_min,
                    "lambda_cap": lambda_cap,
                    "seed": args.seed,
                }
            )
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": str(source_path),
                "settings": vars(args),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with output_path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
