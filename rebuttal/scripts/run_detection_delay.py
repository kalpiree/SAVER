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
) -> dict[str, float | int | str | None]:
    valid = ~false_alarms
    detected = valid & (delays > 0)
    detected_delays = delays[detected]
    commits = detected_delays - 1
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
        "worst_case_prealarm_commits_mean": float(commits.mean()) if commits.size else None,
        "worst_case_prealarm_commits_median": percentile(commits, 50),
        "worst_case_prealarm_commits_p90": percentile(commits, 90),
        "worst_case_prealarm_commits_p95": percentile(commits, 95),
    }


def run_replay(
    risks: np.ndarray,
    theta: float,
    alpha: float,
    grid_size: int,
    pre_steps: int,
    lambda_grid_size: int,
    q_min: float,
) -> tuple[dict[str, dict[str, np.ndarray]], float]:
    replications, total_steps = risks.shape
    lambda_cap = 1.0 / (2.0 * ((1.0 / q_min) - 1.0 + theta))
    lambda_grid = np.linspace(0.0, lambda_cap, lambda_grid_size)
    candidate_scores = np.zeros((replications, lambda_grid_size), dtype=np.float64)
    log_monitor = np.zeros(replications, dtype=np.float64)
    thresholds = {
        "fixed_boundary": np.log(1.0 / alpha),
        "grid_adjusted_beta_max": np.log(grid_size / alpha),
    }
    state = {
        name: {
            "delays": np.zeros(replications, dtype=np.int32),
            "false_alarms": np.zeros(replications, dtype=bool),
        }
        for name in thresholds
    }

    for step in range(total_steps):
        selected = np.argmax(candidate_scores, axis=1)
        lambda_t = lambda_grid[selected]
        centered = risks[:, step] - theta
        log_monitor += np.log1p(lambda_t * centered)
        candidate_scores += np.log1p(centered[:, None] * lambda_grid[None, :])

        for name, log_threshold in thresholds.items():
            crossed = log_monitor >= log_threshold
            if step < pre_steps:
                state[name]["false_alarms"] |= crossed
            else:
                pending = state[name]["delays"] == 0
                new_detection = crossed & pending & ~state[name]["false_alarms"]
                state[name]["delays"][new_detection] = step - pre_steps + 1

    return state, lambda_cap


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    output_path = Path(args.output)
    payload = json.loads(source_path.read_text())
    beta_key = str(args.beta)
    observed = np.array(
        [
            float(snapshot["oracle_risks"][beta_key])
            for snapshot in payload["snapshots"]
            if beta_key in snapshot["oracle_risks"]
        ],
        dtype=np.float64,
    )
    if observed.size == 0:
        raise ValueError(f"No oracle risks found for beta={args.beta}")

    shifts = [float(value) for value in args.shifts.split(",") if value.strip()]
    total_steps = args.pre_steps + args.post_steps
    rng = np.random.default_rng(args.seed)
    sampled = observed[
        rng.integers(0, observed.size, size=(args.replications, total_steps))
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
        )
        post_mean = float(np.minimum(1.0, observed + shift).mean())
        for name, threshold in (
            ("fixed_boundary", 1.0 / args.alpha),
            ("grid_adjusted_beta_max", args.grid_size / args.alpha),
        ):
            row = summarize(
                delays=state[name]["delays"],
                false_alarms=state[name]["false_alarms"],
                threshold_name=name,
                threshold=threshold,
                replications=args.replications,
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
                    "observed_risk_count": int(observed.size),
                    "prechange_risk_mean": float(observed.mean()),
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
