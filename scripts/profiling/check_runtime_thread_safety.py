#!/usr/bin/env python3
"""Run Foundation/runtime/playback stress with Swift 6 and Thread Sanitizer.

Uses production source directly; no model downloads or GPU inference. Run
sequentially with the other builds, tests and performance measurements.
"""
import pathlib
import subprocess
from runtime_profile_common import require_idle

require_idle()
root = pathlib.Path(__file__).resolve().parents[2]
output = root / ".build/runtime-thread-safety"
output.mkdir(parents=True, exist_ok=True)
player = (root / "Sources/AudioCommon/StreamingAudioPlayer.swift").read_text()
ring_source = player.split("/// Streams TTS audio via AVAudioEngine", maxsplit=1)
assert len(ring_source) == 2 and "public final class AudioSampleRingBuffer" in ring_source[0]
ring = output / "AudioSampleRingBuffer.swift"
ring.write_text(ring_source[0] + "\n#endif\n")
binary = output / "runtime-thread-safety"
sources = sorted((root / "Sources/SpeechRuntime").glob("*.swift"))
sources += [ring, root / "Sources/AudioCommon/PlaybackSampleBuffer.swift",
            pathlib.Path(__file__).with_name("RuntimeOwnershipStress.swift")]
subprocess.run(["xcrun", "swiftc", "-swift-version", "6", "-strict-concurrency=complete", "-O", "-g",
                "-sanitize=thread", "-parse-as-library", *map(str, sources), "-o", str(binary)], check=True)
subprocess.run([str(binary)], check=True)
