#!/usr/bin/env python3
"""Sequential HTTP/WebSocket latency profile against a local release harness.

Requires soundfile, numpy and websockets. The fixture must be mono 24 kHz.
Use --concurrent only with a server that serializes mutable model access.
"""
import argparse
import base64
import concurrent.futures
import hashlib
import http.client
import io
import json
import pathlib
import socket
import subprocess
import time

import numpy as np
import soundfile as sf
from websockets.sync.client import connect
from runtime_profile_common import require_idle

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=pathlib.Path, required=True)
parser.add_argument("--audio", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--requests", type=int, default=12)
parser.add_argument("--concurrent", type=int, default=0)
parser.add_argument("--tts", action="store_true", help="Also verify cached Kokoro HTTP and WebSocket audio")
parser.add_argument("--coreml", action="store_true", help="Also verify the managed CoreML HTTP/WebSocket routes")
args = parser.parse_args()
require_idle()
with socket.socket() as reservation:
    reservation.bind(("127.0.0.1", 0))
    port = reservation.getsockname()[1]

audio, rate = sf.read(args.audio, dtype="float32")
assert rate == 24000 and audio.ndim == 1, "Use the mono 24 kHz ASR fixture"
pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
boundary = "RuntimeOwnershipProfileBoundary"
body = (f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\nqwen3\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        'Content-Type: audio/wav\r\n\r\n').encode() + args.audio.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()


def request(path, payload=None, content_type=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    try:
        start = time.perf_counter()
        headers = {"Content-Type": content_type or f"multipart/form-data; boundary={boundary}"} if payload else {}
        connection.request("POST" if payload else "GET", path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read()
        seconds = time.perf_counter() - start
        assert response.status == 200, (response.status, data)
        if "json" in response.getheader("Content-Type", ""):
            decoded = json.loads(data)
        else:
            waveform, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
            decoded = {"audio_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                       "samples": len(waveform), "sample_rate": sample_rate,
                       "rms": float(np.sqrt(np.mean(waveform ** 2)))}
        return {"seconds": seconds, "response": decoded}
    finally:
        connection.close()


result = {"audio": str(args.audio.resolve()), "audio_seconds": len(audio) / rate,
          "http": [], "websocket": [], "health_during_inference": [], "concurrent_http": []}
log_path = args.output.with_suffix(".server.log")


def receive_until(ws, event):
    for _ in range(1000):
        message = json.loads(ws.recv(timeout=120))
        assert message["type"] != "error", message
        if message["type"] == event:
            return message
    raise RuntimeError(f"Event not received: {event}")


with log_path.open("w") as log:
    server = subprocess.Popen([str(args.binary.resolve()), "--serve", "--port", str(port)], stdout=log, stderr=log)
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                request("/health")
                break
            except (OSError, AssertionError):
                assert server.poll() is None, f"Server exited; see {log_path}"
                if time.monotonic() > deadline:
                    raise TimeoutError("Server failed to start")
                time.sleep(0.05)
        result["cold_http"] = request("/v1/audio/transcriptions", body)
        request("/v1/audio/transcriptions", body)
        for index in range(args.requests):
            row = request("/v1/audio/transcriptions", body)
            assert "tomorrow" in row["response"]["text"].lower(), row
            result["http"].append(row)
            print(f"HTTP {index}: {row['seconds']:.6f}s", flush=True)

        with connect(f"ws://127.0.0.1:{port}/v1/realtime", max_size=16 * 1024 * 1024) as ws:
            receive_until(ws, "session.created")
            ws.send(json.dumps({"type": "session.update", "session": {
                "input_audio_transcription": {"model": "qwen3"}, "input_audio_format": "pcm16"}}))
            receive_until(ws, "session.updated")
            encoded = base64.b64encode(pcm).decode()
            append = json.dumps({"type": "input_audio_buffer.append", "audio": encoded})
            for index in range(args.requests + 2):
                start = time.perf_counter()
                ws.send(append)
                ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                message = receive_until(ws, "conversation.item.input_audio_transcription.completed")
                seconds = time.perf_counter() - start
                assert "tomorrow" in message["transcript"].lower(), message
                if index >= 2:
                    result["websocket"].append({"seconds": seconds, "transcript": message["transcript"]})
                    print(f"WS {index - 2}: {seconds:.6f}s", flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(2, args.concurrent)) as executor:
            inference = executor.submit(request, "/v1/audio/transcriptions", body)
            while not inference.done():
                result["health_during_inference"].append(request("/health")["seconds"])
                time.sleep(0.01)
            assert "tomorrow" in inference.result()["response"]["text"].lower()
            if args.concurrent:
                start = time.perf_counter()
                futures = [executor.submit(request, "/v1/audio/transcriptions", body) for _ in range(args.concurrent)]
                result["concurrent_http"] = [future.result() for future in futures]
                result["concurrent_wall_seconds"] = time.perf_counter() - start
                assert all("tomorrow" in row["response"]["text"].lower() for row in result["concurrent_http"])
        if args.coreml:
            rows = request("/v1/models")["response"]["data"]
            result["coreml_descriptor"] = next(row for row in rows if row["id"] == "qwen3-asr-coreml")
            assert result["coreml_descriptor"]["backend"] == "coreml"
            assert result["coreml_descriptor"]["input_sample_rate"] == 16000
            assert result["coreml_descriptor"]["streaming"] == "buffered"
            coreml_body = body.replace(b'name="model"\r\n\r\nqwen3\r\n', b'name="model"\r\n\r\nqwen3-coreml\r\n', 1)
            assert coreml_body != body
            result["coreml_http"] = [request("/v1/audio/transcriptions", coreml_body) for _ in range(2)]
            assert all("tomorrow" in row["response"]["text"].lower() for row in result["coreml_http"])
            result["coreml_websocket"] = []
            with connect(f"ws://127.0.0.1:{port}/v1/realtime", max_size=16 * 1024 * 1024) as ws:
                receive_until(ws, "session.created")
                ws.send(json.dumps({"type": "session.update", "session": {
                    "input_audio_transcription": {"model": "qwen3-coreml"}, "input_audio_format": "pcm16"}}))
                receive_until(ws, "session.updated")
                for _ in range(2):
                    start = time.perf_counter()
                    ws.send(append)
                    ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    message = receive_until(ws, "conversation.item.input_audio_transcription.completed")
                    assert "tomorrow" in message["transcript"].lower(), message
                    result["coreml_websocket"].append({"seconds": time.perf_counter() - start,
                                                       "transcript": message["transcript"]})
        if args.tts:
            text = "The quick brown fox jumps over the lazy dog."
            speech = json.dumps({"model": "kokoro", "voice": "af_heart", "input": text,
                                 "response_format": "wav"}).encode()
            request("/v1/audio/speech", speech, "application/json")
            result["tts_http"] = [request("/v1/audio/speech", speech, "application/json") for _ in range(4)]
            assert all(row["response"]["samples"] > 2400 and row["response"]["rms"] > 0.001 for row in result["tts_http"])
            result["tts_websocket"] = []
            with connect(f"ws://127.0.0.1:{port}/v1/realtime", max_size=16 * 1024 * 1024) as ws:
                receive_until(ws, "session.created")
                ws.send(json.dumps({"type": "session.update", "session": {"model": "kokoro", "language": "english"}}))
                receive_until(ws, "session.updated")
                for index in range(5):
                    start = time.perf_counter()
                    first_audio = None
                    chunks = []
                    ws.send(json.dumps({"type": "response.create", "response": {"instructions": text}}))
                    while True:
                        message = json.loads(ws.recv(timeout=120))
                        assert message["type"] != "error", message
                        if message["type"] == "response.audio.delta":
                            if first_audio is None:
                                first_audio = time.perf_counter() - start
                            chunks.append(base64.b64decode(message["delta"]))
                        if message["type"] == "response.done":
                            break
                    elapsed = time.perf_counter() - start
                    waveform = b"".join(chunks)
                    assert len(waveform) > 4800
                    if index:
                        result["tts_websocket"].append({"seconds": elapsed, "first_audio_seconds": first_audio,
                            "samples": len(waveform) // 2, "chunks": len(chunks), "sha256": hashlib.sha256(waveform).hexdigest()})
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
