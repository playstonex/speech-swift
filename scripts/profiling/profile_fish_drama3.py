#!/usr/bin/env python3
"""Profile exported Drama 3 in release mode, then verify every WAV in a separate ASR process."""
import argparse
import datetime
import hashlib
import json
import math
import platform
import shutil
import statistics
import subprocess
import time
from pathlib import Path

from runtime_profile_common import require_idle

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."


def active_jobs(ignore_pid=None):
    names = {"swift-build", "swift-test", "swift-frontend", "swift-driver", "xctest", "speech", "audio",
             "speech-server", "audio-server", "fish-drama3-profile", "metal", "metallib"}
    processes = []
    for row in subprocess.check_output(["ps", "-axo", "pid=,ppid=,comm="], text=True).splitlines():
        fields = row.split(None, 2)
        if len(fields) == 3:
            processes.append((int(fields[0]), int(fields[1]), Path(fields[2]).name))
    ignored = {ignore_pid} if ignore_pid is not None else set()
    while True:
        descendants = {pid for pid, parent, _ in processes if parent in ignored}
        if descendants <= ignored:
            break
        ignored |= descendants
    return [(pid, name) for pid, _, name in processes if pid not in ignored
            and (name in names or name.endswith(("-bench", "-profile")))]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def validate_bundle(bundle):
    manifest = json.loads((bundle / "soniqo_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Bundle manifest filenames must be flat")
        path = bundle / name
        if path.is_symlink() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError("Bundle checksum mismatch: " + name)
    return manifest


def summarize(profile, verification):
    rows = profile["rows"]
    warm = [row for row in rows if row["phase"] == "warm"]
    first = [row for row in rows if row["phase"] == "first"]
    if len(first) != 1 or not warm or len(warm) != profile["warm_repeats"]:
        raise ValueError("Incomplete inference measurements")
    verified = {row["id"]: row for row in verification}
    for row in rows:
        identity = f"{profile['mode']}-{row['index']}"
        if identity not in verified or not verified[identity]["passed"]:
            raise ValueError("ASR round-trip failed or missing: " + identity)
        if not all(math.isfinite(row[key]) and row[key] > 0 for key in ("generation_seconds", "audio_seconds")):
            raise ValueError("Invalid timing or audio duration")
    result = {"mode": profile["mode"], "load_seconds": profile["load_seconds"],
            "first_synthesis_seconds": first[0]["generation_seconds"],
            "warm_synthesis_median_seconds": statistics.median(row["generation_seconds"] for row in warm),
            "warm_synthesis_min_seconds": min(row["generation_seconds"] for row in warm),
            "warm_synthesis_max_seconds": max(row["generation_seconds"] for row in warm),
            "warm_audio_median_seconds": statistics.median(row["audio_seconds"] for row in warm),
            "warm_rtf_median": statistics.median(row["generation_seconds"] / row["audio_seconds"] for row in warm),
            "peak_rss_bytes": profile["final_memory"]["peak_rss_bytes"],
            "peak_physical_bytes": profile["final_memory"]["peak_physical_bytes"],
            "loaded_physical_bytes": profile["loaded_memory"]["physical_bytes"],
            "baseline_physical_bytes": profile["baseline_memory"]["physical_bytes"],
            "warm_repeats": len(warm), "asr_round_trips_passed": len(rows),
            "reference_encoding_in_generation": profile["reference_encoding_in_generation"]}
    if profile.get("delivery") == "streaming":
        for row in rows:
            latency = row.get("first_audio_seconds", 0)
            if not math.isfinite(latency) or not 0 < latency <= row["generation_seconds"] or row.get("chunk_count", 0) < 1:
                raise ValueError("Invalid first-audio measurement")
        result.update(first_audio_seconds=first[0]["first_audio_seconds"],
                      warm_first_audio_median_seconds=statistics.median(row["first_audio_seconds"] for row in warm),
                      warm_first_audio_min_seconds=min(row["first_audio_seconds"] for row in warm),
                      warm_first_audio_max_seconds=max(row["first_audio_seconds"] for row in warm))
    return result


def build(jobs):
    require_idle()
    destination = ROOT / ".build/fish-drama3-profile-sources"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("FishDrama3Benchmark.swift"), destination)
    for name in ("Memory.swift", "Normalizer.swift", "WER.swift"):
        shutil.copy2(ROOT / "Sources/AsrBenchmark" / name, destination)
    package = ROOT / "Package.swift"
    original = package.read_bytes()
    addition = b'''
package.products.append(.executable(name: "fish-drama3-profile", targets: ["FishDrama3Benchmark"]))
package.targets.append(.executableTarget(name: "FishDrama3Benchmark",
    dependencies: ["FishAudioTTS", "AudioCommon", "Qwen3ASR", .product(name: "MLX", package: "mlx-swift")],
    path: ".build/fish-drama3-profile-sources"))
'''
    try:
        package.write_bytes(original + addition)
        subprocess.run(["swift", "build", "-c", "release", "--disable-sandbox", "--product",
                        "fish-drama3-profile", "-j", str(jobs)], cwd=ROOT, check=True)
    finally:
        package.write_bytes(original)
    subprocess.run([str(ROOT / "scripts/build_mlx_metallib.sh"), "release"], cwd=ROOT, check=True)


def run(command, log_path):
    require_idle()
    if conflicts := active_jobs():
        raise RuntimeError(f"Other build/model processes are active: {conflicts}")
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if conflicts := active_jobs(ignore_pid=process.pid):
                    raise RuntimeError(f"Another job started; discarding this measurement: {conflicts}")
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)


def runtime_fingerprint():
    sources = sorted((ROOT / "Sources/FishAudioTTS").glob("*.swift"))
    sources += [ROOT / "Package.swift", ROOT / "Sources/AudioCLILib/SpeakCommand.swift"]
    result = hashlib.sha256()
    for path in sources:
        result.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes() + b"\0")
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--warm-repeats", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--delivery", choices=("buffered", "streaming", "both"), default="buffered")
    parser.add_argument("--chunk-frames", type=int, choices=range(1, 129), default=3)
    parser.add_argument("--memory-mode", choices=("balanced", "unrestricted"), default="balanced")
    parser.add_argument("--binary", type=Path, help="Existing release profiler, for baseline comparison")
    parser.add_argument("--runtime-sources-sha256", help="Required source fingerprint for an existing baseline binary")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**64 or args.jobs < 1:
        parser.error("Seed must fit UInt64 and jobs must be positive")
    if args.binary and (args.build or not args.runtime_sources_sha256 or len(args.runtime_sources_sha256) != 64
                        or any(c not in "0123456789abcdef" for c in args.runtime_sources_sha256)):
        parser.error("An existing --binary requires its --runtime-sources-sha256 and cannot use --build")
    args.bundle = args.bundle.resolve()
    args.reference = args.reference.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Validating exported file checksums", flush=True)
    manifest = validate_bundle(args.bundle)
    if args.build:
        build(args.jobs)
    binary = args.binary.resolve() if args.binary else ROOT / ".build/release/fish-drama3-profile"
    if not binary.is_file():
        parser.error("Release profiler is missing; pass --build")
    profiles = []
    inputs = []
    deliveries = ("buffered", "streaming") if args.delivery == "both" else (args.delivery,)
    for mode, delivery in ((mode, delivery) for delivery in deliveries for mode in ("text-only", "voice-clone")):
        label = mode + ("-stream" if delivery == "streaming" else "")
        print("Measuring " + label, flush=True)
        folder = args.output_dir / label
        folder.mkdir(exist_ok=True)
        output = folder / "profile.json"
        run([str(binary), "--bundle", str(args.bundle), "--mode", mode, "--text", args.text,
             "--reference", str(args.reference), "--reference-text", args.reference_text,
             "--seed", str(args.seed), "--warm-repeats", str(args.warm_repeats), "--chunk-frames", str(args.chunk_frames),
             "--memory-mode", args.memory_mode,
             "--output", str(output)] + (["--streaming"] if delivery == "streaming" else []), folder / "profile.log")
        profile = json.loads(output.read_text())
        for row in profile["rows"]:
            row["wav_sha256"] = digest(folder / row["audio_file"])
        profiles.append(profile)
        for row in profile["rows"]:
            inputs.append({"id": f"{label}-{row['index']}", "text": row["text"], "path": str(folder / row["audio_file"])})
    asr_manifest = args.output_dir / "asr-inputs.json"
    asr_manifest.write_text(json.dumps(inputs, indent=2) + "\n")
    print("Verifying all WAVs with independent ASR", flush=True)
    asr_path = args.output_dir / "asr-results.json"
    run([str(binary), "--asr-manifest", str(asr_manifest), "--output", str(asr_path)], args.output_dir / "asr.log")
    verification = json.loads(asr_path.read_text())
    report = {"schema_version": 1, "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
              "unified_memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
              "macos": platform.mac_ver()[0], "build_configuration": "release",
              "source_repo": manifest["source_repo"], "source_revision": manifest["source_revision"],
              "runtime_sources_sha256": args.runtime_sources_sha256 if args.binary else runtime_fingerprint(),
              "model_config_sha256": manifest["files"]["config.json"]["sha256"],
              "binary_sha256": digest(binary), "warm_repeats": args.warm_repeats, "seed": args.seed,
              "delivery": args.delivery,
              "memory_mode": args.memory_mode,
              "reference": {"wav_sha256": digest(args.reference), "text": args.reference_text},
              "measurement": "Complete waveform and, for streaming, first nonempty PCM callback; excluding model load, reference file I/O and WAV writes; cloning includes reference encoding on every call. Local cached weights; no disk-cold claim.",
              "independent_asr": verification["model_id"],
              "summary": [summarize(profile, verification["rows"]) for profile in profiles],
              "profiles": profiles, "asr": verification}
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in report["summary"]:
        if "warm_first_audio_median_seconds" in row:
            print(f"{row['mode']}: warm first PCM {row['warm_first_audio_median_seconds']*1000:.1f} ms", flush=True)
        print(f"{row['mode']}: load {row['load_seconds']:.3f}s; warm synthesis {row['warm_synthesis_median_seconds']:.3f}s; "
              f"RTF {row['warm_rtf_median']:.3f}; peak RSS {row['peak_rss_bytes']/2**30:.3f} GiB; "
              f"physical {row['peak_physical_bytes']/2**30:.3f} GiB; ASR {row['asr_round_trips_passed']}/{len(profiles[0]['rows'])}", flush=True)


if __name__ == "__main__":
    main()
