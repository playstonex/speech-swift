"""Admission check shared by local runtime verification tools."""
import pathlib
import subprocess


def require_idle():
    processes = subprocess.check_output(["ps", "-axo", "comm="], text=True).splitlines()
    names = {"swift-build", "swift-test", "xctest", "asr-bench", "vad-bench", "diarization-bench",
             "speech", "audio", "speech-server", "audio-server", "runtime-ownership-profile"}
    conflicts = [process for process in processes if pathlib.Path(process).name in names]
    if conflicts:
        raise RuntimeError(f"Other build/model processes are active: {conflicts}")
