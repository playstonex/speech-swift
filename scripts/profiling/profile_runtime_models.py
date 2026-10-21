#!/usr/bin/env python3
"""Run baseline/runtime profiles sequentially in ABBA order, then a native control.

Build both worktrees with build_runtime_profile.py and compile their MLX
metallibs first. Each child owns one model and exits before the next starts.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import time
from runtime_profile_common import require_idle

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline", type=pathlib.Path, required=True)
parser.add_argument("--current", type=pathlib.Path, required=True)
parser.add_argument("--manifest", type=pathlib.Path, required=True)
parser.add_argument("--output-dir", type=pathlib.Path, required=True)
parser.add_argument("--backends", nargs="+", default=["mlx", "coreml"], choices=["mlx", "coreml"])
parser.add_argument("--repeats", type=int, default=2)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
plan = [("baseline-a", args.baseline, "native"), ("managed-a", args.current, "managed"),
        ("managed-b", args.current, "managed"), ("baseline-b", args.baseline, "native"),
        ("current-native", args.current, "native")]
metadata = {"start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "baseline_binary_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
            "current_binary_sha256": hashlib.sha256(args.current.read_bytes()).hexdigest(),
            "repeats": args.repeats, "runs": []}
try:
    for backend in args.backends:
        for label, binary, mode in plan:
            name = f"{backend}-{label}"
            # A second SwiftPM process or model benchmark invalidates this run
            # and can exhaust unified memory. Check again before every model.
            require_idle()
            print(f"START {name}", flush=True)
            start = time.monotonic()
            with (args.output_dir / f"{name}.log").open("w") as log:
                subprocess.run([str(binary.resolve()), "--backend", backend, "--mode", mode,
                                "--manifest", str(args.manifest.resolve()), "--repeats", str(args.repeats),
                                "--output", str((args.output_dir / f"{name}.json").resolve())],
                               stdout=log, stderr=log, check=True)
            elapsed = time.monotonic() - start
            metadata["runs"].append({"name": name, "wall_seconds": elapsed})
            print(f"DONE {name} {elapsed:.2f}s", flush=True)
finally:
    metadata["end_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (args.output_dir / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
