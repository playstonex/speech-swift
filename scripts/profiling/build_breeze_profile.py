#!/usr/bin/env python3
"""Build a release Breeze TTS profiler with isolated synthesis/ASR process modes.

python3 scripts/profiling/build_breeze_profile.py
.build/release/breeze-tts-bench --bundle out/Breeze-TTS-2-MLX-8bit --output /tmp/breeze.json
.build/release/breeze-tts-bench --verify /tmp/breeze.json --output /tmp/breeze-asr.json
"""
import pathlib
import shutil
import subprocess
processes = subprocess.check_output(["ps", "-axo", "comm="], text=True).splitlines()
build_names = {"swift-build", "swift-test", "swift-frontend", "xcodebuild", "xctest", "speech", "audio", "speech-server", "breeze-tts-bench", "asr-bench", "vad-bench", "diarization-bench"}
conflicts = [name for name in processes if pathlib.Path(name).name in build_names or ".build/" in name]
if conflicts:
    raise RuntimeError(f"Finish other build/test/benchmark processes first: {conflicts}")
repo = pathlib.Path(__file__).resolve().parents[2]
destination = repo / ".build/breeze-profile-sources"
destination.mkdir(parents=True, exist_ok=True)
shutil.copy2(pathlib.Path(__file__).with_name("BreezeTTSBenchmark.swift"), destination)
for filename in ("Normalizer.swift", "WER.swift", "Memory.swift"):
    shutil.copy2(repo / "Sources/AsrBenchmark" / filename, destination)
manifest = repo / "Package.swift"
original = manifest.read_bytes()
addition = '''
package.products.append(.executable(name: "breeze-tts-bench", targets: ["BreezeTTSProfile"]))
package.targets.append(.executableTarget(name: "BreezeTTSProfile", dependencies: ["BreezeTTS", "AudioCommon", "Qwen3ASR", "BenchmarkSupport"],
    path: ".build/breeze-profile-sources"))
'''
try:
    manifest.write_bytes(original + addition.encode())
    subprocess.run(["swift", "build", "-c", "release", "--disable-sandbox", "--product", "breeze-tts-bench", "-j", "4"],
                   cwd=repo, check=True)
    subprocess.run(["./scripts/build_mlx_metallib.sh", "release"], cwd=repo, check=True)
finally:
    manifest.write_bytes(original)
