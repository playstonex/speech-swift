#!/usr/bin/env python3
"""Summarize the last (warm) pass and transcript parity of release profiles."""
import json
import math
import pathlib
import random
import statistics
import sys

root = pathlib.Path(sys.argv[1])


def load(backend, label):
    return json.loads((root / f"{backend}-{label}.json").read_text())


def warm_rows(run):
    last = max(row["pass"] for row in run["rows"])
    return [row for row in run["rows"] if row["pass"] == last]


def metrics(runs):
    rows = [row for run in runs for row in warm_rows(run)]
    seconds = sorted(row["seconds"] for row in rows)
    return {
        "warm_samples": len(rows),
        "median_ms": statistics.median(seconds) * 1000,
        "p95_ms": seconds[math.ceil(len(seconds) * 0.95) - 1] * 1000,
        "weighted_rtf": sum(seconds) / sum(row["audio_seconds"] for row in rows),
        "wer_percent": 100 * sum(row["word_errors"] for row in rows) / sum(row["reference_words"] for row in rows),
        "max_peak_physical_mib": max(snapshot["memory"]["peak_physical_bytes"] for run in runs for snapshot in run["snapshots"]) / 1048576,
        "median_load_seconds": statistics.median(run["load_seconds"] for run in runs),
        "memory_growth_last_pass_mib": [
            (run["snapshots"][-2]["memory"]["physical_bytes"] - run["snapshots"][-3]["memory"]["physical_bytes"]) / 1048576
            for run in runs],
    }


summary = {}
for backend in ("mlx", "coreml"):
    baseline = [load(backend, "baseline-a"), load(backend, "baseline-b")]
    managed = [load(backend, "managed-a"), load(backend, "managed-b")]
    control = [load(backend, "current-native")]
    baseline_metrics, managed_metrics = metrics(baseline), metrics(managed)
    baseline_clips = {}
    managed_clips = {}
    for runs, destination in [(baseline, baseline_clips), (managed, managed_clips)]:
        for run in runs:
            for row in warm_rows(run):
                destination.setdefault(row["path"], []).append(row["seconds"])
    paths = sorted(baseline_clips)
    log_ratios = [math.log(statistics.mean(managed_clips[path]) / statistics.mean(baseline_clips[path])) for path in paths]
    generator = random.Random(17)
    bootstrap = sorted(math.expm1(statistics.mean(generator.choices(log_ratios, k=len(paths)))) * 100 for _ in range(4000))
    differences = []
    comparisons = 0
    for before, after in zip(baseline, managed):
        by_key = {(row["pass"], row["index"]): row for row in before["rows"]}
        for row in after["rows"]:
            original = by_key[(row["pass"], row["index"])]
            comparisons += 1
            if original["text"] != row["text"]:
                differences.append({"clip": pathlib.Path(row["path"]).name, "pass": row["pass"],
                                    "before": original["text"], "after": row["text"]})
    summary[backend] = {
        "baseline": baseline_metrics, "managed": managed_metrics, "current_native": metrics(control),
        "weighted_rtf_change_percent": 100 * (managed_metrics["weighted_rtf"] / baseline_metrics["weighted_rtf"] - 1),
        "paired_geometric_latency_change_percent": math.expm1(statistics.mean(log_ratios)) * 100,
        "paired_bootstrap_95_percent": [bootstrap[100], bootstrap[3899]],
        "transcript_comparisons": comparisons, "transcript_differences": differences,
    }

(root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
