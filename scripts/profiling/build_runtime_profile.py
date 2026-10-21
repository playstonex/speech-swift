#!/usr/bin/env python3
"""Build a release-only profiling harness without keeping a package target.

Usage: python3 scripts/profiling/build_runtime_profile.py /path/to/worktree
Run only when no other Swift build, test or model process is active.
"""
import pathlib
import shutil
import subprocess
import sys
from runtime_profile_common import require_idle

require_idle()
repo = pathlib.Path(sys.argv[1]).resolve()
source = pathlib.Path(__file__).with_name("RuntimeOwnershipProfile.swift")
destination = repo / ".build/runtime-profile-sources"
destination.mkdir(parents=True, exist_ok=True)
shutil.copy2(source, destination / "RuntimeOwnershipProfile.swift")
for filename in ("Normalizer.swift", "WER.swift"):
    shutil.copy2(repo / "Sources/AsrBenchmark" / filename, destination / filename)
adapter = (repo / "Sources/Qwen3ASRRuntime").is_dir()
if adapter:
    playback = (repo / "Sources/AudioCommon/PlaybackSampleBuffer.swift").read_text()
    (destination / "PlaybackSampleBuffer.swift").write_text("import AudioCommon\n" + playback)
manifest = repo / "Package.swift"
original = manifest.read_bytes()
dependencies = '["Qwen3ASR", "AudioCommon", "AudioServer"' + (', "Qwen3ASRRuntime", "SpeechRuntime"' if adapter else '') + ']'
settings = '[.define("RUNTIME_ADAPTER")]' if adapter else '[]'
addition = f'''
package.products.append(.executable(name: "runtime-ownership-profile", targets: ["RuntimeOwnershipProfile"]))
package.targets.append(.executableTarget(name: "RuntimeOwnershipProfile", dependencies: {dependencies},
    path: ".build/runtime-profile-sources", swiftSettings: {settings}))
'''
try:
    manifest.write_bytes(original + addition.encode())
    subprocess.run(["swift", "build", "-c", "release", "--disable-sandbox", "--scratch-path",
                    ".build/index-build", "--product", "runtime-ownership-profile", "--jobs", "6"],
                   cwd=repo, check=True)
finally:
    manifest.write_bytes(original)
