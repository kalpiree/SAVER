import argparse
import csv
import json
import math
from pathlib import Path
import statistics


RUN_FIELDS = (
    "protocol_version", "mode", "seed", "limit", "completed_steps", "config",
    "protocol", "events_path", "stream_sha256", "audit_sha256",
)
LOCALITY_FIELDS = (
    "locality_first_token_risk", "locality_complete_error",
    "locality_before_accuracy", "locality_candidate_accuracy",
    "locality_committed_accuracy",
)
EVENT_FIELDS = (
    "step", "source_id", "accepted", "pre_monitor_ratio", "post_monitor_ratio",
    "benign", "candidate_success", "target_token_count",
    "first_token_generality_risk", "full_answer_error",
    "multi_token_locality_count",
) + LOCALITY_FIELDS
IDENTITY_FIELDS = ("scope", "run_path", "mode", "seed", "run_count")
STAGE_FIELDS = (
    "stage", "stage_start", "stage_end", "pre_monitor_bin", "request_count",
    "benign_count", "accepted_benign", "proportion",
)
EPISODE_FIELDS = (
    "episode_id", "start_step", "readmission_step", "end_step", "delay",
    "observed_duration", "rejected_steps", "right_censored",
)
RECOVERY_FIELDS = (
    "submitted_steps", "rejected_steps", "episode_count", "recovered_count",
    "right_censored_count", "recovered_delay_sum", "median_recovered_delay",
)
ANSWER_FIELDS = (
    "submitted_steps", "accepted_steps", "rejected_steps",
    "multi_token_request_count", "accepted_multi_token_count",
    "rejected_multi_token_count", "candidate_success_count",
    "complete_success_count", "complete_success_proportion",
    "generality_correlation_count", "generality_spearman",
    "multi_token_locality_count", "locality_step_count",
    "locality_audit_observation_count", "locality_committed_accuracy_sum",
    "mean_committed_locality_accuracy", "locality_correlation_count",
    "locality_spearman",
)
TABLE_FIELDS = {
    "stage_acceptance": IDENTITY_FIELDS + STAGE_FIELDS,
    "recovery_episodes": IDENTITY_FIELDS + EPISODE_FIELDS,
    "recovery_summary": IDENTITY_FIELDS + RECOVERY_FIELDS,
    "complete_answers": IDENTITY_FIELDS + ANSWER_FIELDS,
}
REFERENCE_FIELDS = (
    "audit_reference_sha256", "audit_reference_count", "excluded_audit_ids",
    "initial_audit", "biographies_sha256",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fields(value, fields, where):
    _require(isinstance(value, dict), f"{where}: expected a JSON object")
    missing = sorted(set(fields) - value.keys())
    _require(not missing, f"{where}: missing fields: {', '.join(missing)}")


def _integer(value, where, minimum=None):
    _require(type(value) is int, f"{where}: expected an integer")
    if minimum is not None:
        _require(value >= minimum, f"{where}: must be >= {minimum}")


def _number(value, where, nullable=False, maximum=None):
    if value is None and nullable:
        return
    _require(type(value) in (int, float), f"{where}: expected a finite number")
    _require(value >= 0, f"{where}: must be nonnegative")
    if type(value) is float:
        _require(math.isfinite(value), f"{where}: expected a finite number")
    if maximum is not None:
        _require(value <= maximum, f"{where}: must be <= {maximum}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def _parse_json(text, where):
    try:
        value = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant,
        )
        json.dumps(value, allow_nan=False)
        return value
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from exc


def _same_number(left, right):
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_protocol(run, where):
    supported = {
        "hard_gate_sampled_risk": False,
        "monotone_beta_search": False,
        "boundary_policy": "adaptive",
        "rejection_policy": "continue",
        "stop_on_boundary_saturation": False,
    }
    for section in ("config", "protocol"):
        settings = run[section]
        for key, expected in supported.items():
            if key in settings:
                _require(
                    type(settings[key]) is type(expected) and settings[key] == expected,
                    f"{where}: unsupported {section}.{key}; expected {expected!r}",
                )
        for key in ("dataset", "dataset_name", "dataset_type"):
            if key in settings:
                name = settings[key]
                _require(
                    isinstance(name, str)
                    and name.lower().replace("_", "").replace("-", "") == "counterfact",
                    f"{where}: R3 requires CounterFact, not {section}.{key}={name!r}",
                )


def _validate_event(event, previous, run, index, where):
    _fields(event, EVENT_FIELDS, where)
    _integer(event["step"], f"{where}.step", 1)
    _require(event["step"] == index, f"{where}: expected contiguous step {index}")
    _require(
        isinstance(event["source_id"], str) and bool(event["source_id"].strip()),
        f"{where}.source_id: expected a nonempty string",
    )
    for key in ("accepted", "benign", "candidate_success"):
        _require(type(event[key]) is bool, f"{where}.{key}: expected a boolean")
    _integer(event["target_token_count"], f"{where}.target_token_count", 1)
    _integer(event["multi_token_locality_count"], f"{where}.multi_token_locality_count", 0)
    for key in ("first_token_generality_risk", "full_answer_error"):
        _number(event[key], f"{where}.{key}", maximum=1)
    _require(
        event["full_answer_error"] == float(not event["candidate_success"]),
        f"{where}: full_answer_error must equal not candidate_success",
    )
    audit_fields = ("audit_before_accuracy", "audit_candidate_accuracy")
    if any(key in event for key in audit_fields):
        _fields(event, audit_fields, where)
        for key in audit_fields:
            _number(event[key], f"{where}.{key}", maximum=1)
        _require(
            event["benign"] == (
                event["candidate_success"]
                and event["audit_candidate_accuracy"] >= event["audit_before_accuracy"]
            ),
            f"{where}: benign disagrees with candidate success and all-bank audit accuracy",
        )
    count = event["multi_token_locality_count"]
    for key in LOCALITY_FIELDS:
        if count:
            _number(event[key], f"{where}.{key}", maximum=1)
        else:
            _require(event[key] is None, f"{where}.{key}: must be null when locality count is zero")
    if count:
        _require(
            _same_number(event["locality_complete_error"], 1 - event["locality_candidate_accuracy"]),
            f"{where}: locality_complete_error must equal 1 - locality_candidate_accuracy",
        )
    for key in ("pre_monitor_ratio", "post_monitor_ratio"):
        _number(event[key], f"{where}.{key}", nullable=run["mode"] != "saver")
    if run["mode"] == "saver":
        _require(
            event["accepted"] == (event["post_monitor_ratio"] < 1),
            f"{where}: rejection must be equivalent to post_monitor_ratio >= 1",
        )
        if previous is not None:
            _require(
                _same_number(event["pre_monitor_ratio"], previous["post_monitor_ratio"]),
                f"{where}: pre_monitor_ratio differs from previous post_monitor_ratio",
            )
    else:
        _require(event["accepted"], f"{where}: unconstrained runs must accept every request")
    if previous is not None:
        _require(
            count == previous["multi_token_locality_count"],
            f"{where}: multi-token locality audit subset count changed",
        )
        if count:
            _require(
                _same_number(
                    event["locality_before_accuracy"], previous["locality_committed_accuracy"],
                ),
                f"{where}: locality_before_accuracy differs from the prior committed model",
            )
    if count:
        expected = event[
            "locality_candidate_accuracy" if event["accepted"] else "locality_before_accuracy"
        ]
        _require(
            _same_number(event["locality_committed_accuracy"], expected),
            f"{where}: committed locality accuracy must reflect acceptance or rollback",
        )


def _load_run(path):
    path = Path(path).expanduser().resolve()
    try:
        run = _parse_json(path.read_text(encoding="utf-8"), str(path))
        _fields(run, RUN_FIELDS, str(path))
        _integer(run["protocol_version"], f"{path}.protocol_version")
        _require(run["protocol_version"] == 1, f"{path}: unsupported protocol_version")
        _require(run["mode"] in ("saver", "unconstrained"), f"{path}: unsupported mode")
        _integer(run["seed"], f"{path}.seed")
        _integer(run["limit"], f"{path}.limit", 1)
        _integer(run["completed_steps"], f"{path}.completed_steps", 0)
        _require(run["completed_steps"] == run["limit"], f"{path}: incomplete run")
        for key in ("config", "protocol"):
            _require(isinstance(run[key], dict), f"{path}.{key}: expected an object")
        if "seed" in run["config"]:
            _integer(run["config"]["seed"], f"{path}.config.seed")
            _require(run["config"]["seed"] == run["seed"], f"{path}: config.seed mismatch")
        _validate_protocol(run, str(path))
        for key in ("stream_sha256", "audit_sha256", "audit_reference_sha256", "biographies_sha256"):
            if key not in run or (key == "biographies_sha256" and run[key] is None):
                continue
            value = run[key]
            _require(
                isinstance(value, str) and len(value) == 64
                and all(char in "0123456789abcdefABCDEF" for char in value),
                f"{path}.{key}: expected a 64-character SHA-256 hex digest",
            )
            run[key] = value.lower()
        _require(
            isinstance(run["events_path"], str) and bool(run["events_path"].strip()),
            f"{path}.events_path: expected a nonempty path",
        )
        events_path = Path(run["events_path"])
        if not events_path.is_absolute():
            events_path = path.parent / events_path
        events_path = events_path.resolve()
        events = []
        source_ids = set()
        with events_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                where = f"{events_path}:{line_number}"
                event = _parse_json(line, where)
                _validate_event(event, events[-1] if events else None, run, line_number, where)
                _require(
                    event["source_id"] not in source_ids,
                    f"{where}: duplicate source_id: {event['source_id']!r}",
                )
                source_ids.add(event["source_id"])
                events.append(event)
                _require(len(events) <= run["completed_steps"], f"{where}: extra event")
        _require(len(events) == run["completed_steps"], f"{path}: incomplete events log")
        if "audit_reference_count" in run:
            _integer(run["audit_reference_count"], f"{path}.audit_reference_count", 0)
            _require(
                run["audit_reference_count"] >= events[0]["multi_token_locality_count"],
                f"{path}: eligible multi-token audit count exceeds audit_reference_count",
            )
        if "initial_audit" in run:
            initial = run["initial_audit"]
            _require(isinstance(initial, dict), f"{path}.initial_audit: expected an object")
            count = events[0]["multi_token_locality_count"]
            if "multi_count" in initial:
                _integer(initial["multi_count"], f"{path}.initial_audit.multi_count", 0)
                _require(initial["multi_count"] == count, f"{path}: initial audit count mismatch")
            for key in ("multi_accuracy", "multi_risk"):
                if key in initial:
                    if count:
                        _number(initial[key], f"{path}.initial_audit.{key}", maximum=1)
                    else:
                        _require(
                            initial[key] is None,
                            f"{path}.initial_audit.{key}: must be null when locality count is zero",
                        )
            if count and "multi_accuracy" in initial:
                _require(
                    _same_number(initial["multi_accuracy"], events[0]["locality_before_accuracy"]),
                    f"{path}: first event locality baseline differs from initial audit",
                )
        return {**run, "run_path": str(path), "events_path": str(events_path), "events": events}
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{path}: cannot read run: {exc}") from exc


def _validate_comparison(runs):
    seen = set()
    reference = runs[0]
    reference_config = {key: value for key, value in reference["config"].items() if key != "seed"}
    for run in runs:
        where = run["run_path"]
        identity = (run["seed"], run["mode"])
        _require(identity not in seen, f"{where}: duplicate seed+mode: {identity}")
        seen.add(identity)
        config = {key: value for key, value in run["config"].items() if key != "seed"}
        _require(_canonical(config) == _canonical(reference_config), f"{where}: mismatched config")
        for key in ("protocol", "stream_sha256", "audit_sha256", "limit"):
            _require(
                _canonical(run[key]) == _canonical(reference[key]),
                f"{where}: mismatched {key}",
            )
        for key in REFERENCE_FIELDS:
            _require(
                (key in run) == (key in reference)
                and _canonical(run.get(key)) == _canonical(reference.get(key)),
                f"{where}: mismatched {key}",
            )
        for event, expected in zip(run["events"], reference["events"]):
            for key in ("source_id", "target_token_count", "multi_token_locality_count"):
                _require(
                    event[key] == expected[key],
                    f"{where}: step {event['step']} mismatched {key} despite matching hashes",
                )


def _average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman(xs, ys):
    xs, ys = list(xs), list(ys)
    _require(len(xs) == len(ys), "Spearman inputs must have equal length")
    if len(xs) < 2:
        return None
    left, right = _average_ranks(xs), _average_ranks(ys)
    mean = (len(xs) + 1) / 2
    left = [value - mean for value in left]
    right = [value - mean for value in right]
    left_ss = math.fsum(value * value for value in left)
    right_ss = math.fsum(value * value for value in right)
    if not left_ss or not right_ss:
        return None
    correlation = math.fsum(x * y for x, y in zip(left, right)) / math.sqrt(left_ss * right_ss)
    return max(-1.0, min(1.0, correlation))


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _stage_acceptance(events, limit):
    early_end, middle_end = (limit + 2) // 3, (2 * limit + 2) // 3
    stages = (("early", 1, early_end), ("middle", early_end + 1, middle_end),
              ("late", middle_end + 1, limit))
    bins = ("<0.8", "[0.8,1)", ">=1")
    rows = {}
    for stage, start, end in stages:
        for label in bins:
            rows[(stage, label)] = {
                "stage": stage, "stage_start": start, "stage_end": end,
                "pre_monitor_bin": label, "request_count": 0,
                "benign_count": 0, "accepted_benign": 0,
            }
    for event in events:
        step, ratio = event["step"], event["pre_monitor_ratio"]
        stage = "early" if step <= early_end else "middle" if step <= middle_end else "late"
        label = bins[0] if ratio < 0.8 else bins[1] if ratio < 1 else bins[2]
        row = rows[(stage, label)]
        row["request_count"] += 1
        row["benign_count"] += int(event["benign"])
        row["accepted_benign"] += int(event["benign"] and event["accepted"])
    for row in rows.values():
        row["proportion"] = _ratio(row["accepted_benign"], row["benign_count"])
    return list(rows.values())


def _episodes(events):
    episodes = []
    start = None
    for event in events:
        if not event["accepted"] and start is None:
            start = event["step"]
        elif event["accepted"] and start is not None:
            end = event["step"]
            episodes.append({
                "episode_id": len(episodes) + 1, "start_step": start,
                "readmission_step": end, "end_step": end, "delay": end - start,
                "observed_duration": end - start, "rejected_steps": end - start,
                "right_censored": False,
            })
            start = None
    if start is not None:
        end = events[-1]["step"]
        episodes.append({
            "episode_id": len(episodes) + 1, "start_step": start,
            "readmission_step": None, "end_step": end, "delay": None,
            "observed_duration": end - start, "rejected_steps": end - start + 1,
            "right_censored": True,
        })
    return episodes


def _recovery_summary(events, episodes):
    delays = [episode["delay"] for episode in episodes if not episode["right_censored"]]
    return {
        "submitted_steps": len(events),
        "rejected_steps": sum(not event["accepted"] for event in events),
        "episode_count": len(episodes), "recovered_count": len(delays),
        "right_censored_count": len(episodes) - len(delays),
        "recovered_delay_sum": sum(delays),
        "median_recovered_delay": statistics.median(delays) if delays else None,
    }


def _complete_answers(events):
    targets = [event for event in events if event["target_token_count"] > 1]
    locality = [event for event in events if event["multi_token_locality_count"] > 0]
    accepted = sum(event["accepted"] for event in events)
    accepted_targets = sum(event["accepted"] for event in targets)
    successes = sum(event["accepted"] and event["candidate_success"] for event in targets)
    accuracy_sum = math.fsum(event["locality_committed_accuracy"] for event in locality)
    return {
        "submitted_steps": len(events), "accepted_steps": accepted,
        "rejected_steps": len(events) - accepted,
        "multi_token_request_count": len(targets),
        "accepted_multi_token_count": accepted_targets,
        "rejected_multi_token_count": len(targets) - accepted_targets,
        "candidate_success_count": sum(event["candidate_success"] for event in targets),
        "complete_success_count": successes,
        "complete_success_proportion": _ratio(successes, len(targets)),
        "generality_correlation_count": len(targets),
        "generality_spearman": spearman(
            (event["first_token_generality_risk"] for event in targets),
            (event["full_answer_error"] for event in targets),
        ),
        "multi_token_locality_count": events[0]["multi_token_locality_count"],
        "locality_step_count": len(locality),
        "locality_audit_observation_count": sum(
            event["multi_token_locality_count"] for event in locality
        ),
        "locality_committed_accuracy_sum": accuracy_sum,
        "mean_committed_locality_accuracy": _ratio(accuracy_sum, len(locality)),
        "locality_correlation_count": len(locality),
        "locality_spearman": spearman(
            (event["locality_first_token_risk"] for event in locality),
            (event["locality_complete_error"] for event in locality),
        ),
    }


def _summarize(runs, scope):
    events = [event for run in runs for event in run["events"]]
    identity = {
        "scope": scope, "run_path": runs[0]["run_path"] if scope == "run" else None,
        "mode": runs[0]["mode"], "seed": runs[0]["seed"] if scope == "run" else None,
        "run_count": len(runs),
    }
    episodes = []
    for run in runs:
        for episode in _episodes(run["events"]):
            episodes.append({
                **identity, "scope": "run", "run_path": run["run_path"],
                "seed": run["seed"], "run_count": 1, **episode,
            })
    stages = _stage_acceptance(events, runs[0]["limit"]) if identity["mode"] == "saver" else []
    return {
        **identity, "seeds": [run["seed"] for run in runs], "limit": runs[0]["limit"],
        "stage_acceptance": [{**identity, **row} for row in stages],
        "recovery_episodes": episodes,
        "recovery_summary": {**identity, **_recovery_summary(events, episodes)},
        "complete_answers": {**identity, **_complete_answers(events)},
    }


def analyze_runs(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    runs = [_load_run(path) for path in paths]
    _require(bool(runs), "at least one run summary is required")
    _validate_comparison(runs)
    runs.sort(key=lambda run: (run["mode"], run["seed"]))
    per_run = []
    for run in runs:
        metadata = {key: value for key, value in run.items() if key != "events"}
        per_run.append({**metadata, **_summarize([run], "run")})
    pooled = {
        mode: _summarize([run for run in runs if run["mode"] == mode], "pooled")
        for mode in ("saver", "unconstrained") if any(run["mode"] == mode for run in runs)
    }
    summaries = per_run + list(pooled.values())
    return {
        "protocol_version": 1, "run_count": len(runs),
        "aggregation": {
            "stage_end_steps": [
                (runs[0]["limit"] + 2) // 3, (2 * runs[0]["limit"] + 2) // 3, runs[0]["limit"],
            ],
            "pre_monitor_bin_edges": [0.8, 1.0],
            "min_target_token_count": 2,
            "correlation": "spearman",
            "rank_method": "average",
        },
        "runs": per_run, "pooled": pooled,
        "stage_acceptance": [row for summary in summaries for row in summary["stage_acceptance"]],
        "recovery_episodes": [row for summary in per_run for row in summary["recovery_episodes"]],
        "recovery_summary": [summary["recovery_summary"] for summary in summaries],
        "complete_answers": [summary["complete_answers"] for summary in summaries],
    }


def _write_outputs(summary, output_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    targets = [output_dir / f"{table}.csv" for table in TABLE_FIELDS]
    targets.append(output_dir / "summary.json")
    for target in targets:
        _require(not target.exists() and not target.is_symlink(), f"refusing to overwrite {target}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for table, fields in TABLE_FIELDS.items():
        with (output_dir / f"{table}.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary[table])
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate protocol-v1 SAVER R1/R2/R3 run logs.")
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = analyze_runs(args.runs)
        _write_outputs(summary, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
