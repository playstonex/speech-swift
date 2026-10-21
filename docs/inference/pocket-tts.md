# Pocket TTS inference

Use `PocketTTS` for CPU speech synthesis with incremental audio delivery.
The current bundle supports English and the Alba preset, at 24 kHz mono.

## Published and local bundles

The export recipe pins the preconverted CoreML source and verifies upstream
file hashes before packaging compiled `.mlmodelc` graphs, constants, attribution,
and a SHA-256 manifest. Exporting requires macOS, Python 3.11–3.13 and the pinned
CoreML export dependencies. The default download ID is
[`aufklarer/Pocket-TTS-100M-CoreML-INT8`](https://huggingface.co/aufklarer/Pocket-TTS-100M-CoreML-INT8).
The runtime downloads and caches this bundle on first use; Python is not needed
for inference. A local bundle can be selected explicitly.

```bash
speech speak "Hello, welcome back." --engine pocket --stream -o hello.wav
speech speak "Hello, welcome back." --engine pocket \
  --pocket-bundle-dir /path/to/Pocket-TTS-100M-CoreML-INT8 \
  --stream --pocket-seed 42 -o hello.wav
speech speak --engine pocket --list-speakers
```

| Flag | Default | Purpose |
|---|---|---|
| `--pocket-bundle-dir` | unset | Load a local exported bundle |
| `--pocket-model-id` | `aufklarer/Pocket-TTS-100M-CoreML-INT8` | Download published bundle |
| `--pocket-seed` | `42` | Deterministic request-local noise; zero is valid |
| `--pocket-max-frames` | `350` | Per-chunk generation limit; reaching it throws |
| `--stream` | false | Deliver frames during generation |
| `--play` | false | Play the generated audio |
| `--language` | English | Accepts `en` or `english` |

The CLI rejects batch files, voice samples, speaker overrides, and style
instructions for this bundle. Whitespace-only text and text over 4,096 characters
are errors. Each request uses fresh recurrent state. The fixed seed is reused
for each request; it is not incremented between narration segments.

## Swift

```swift
import PocketTTS

let model = try PocketTTSModel.fromBundle(URL(fileURLWithPath: "/path/to/bundle"))
let options = PocketTTSOptions(seed: 42, temperature: 0.7, maxFrames: 350)
let samples = try await model.generate(text: "Hello world.", options: options)

try await model.generateStreaming(text: "Hello world.", options: options) { chunk in
    // Awaited delivery provides backpressure. Each PCM frame has 1,920 samples.
    await consume(chunk.samples)
}
for try await chunk in model.generateStream(text: "Hello world.", options: options) {
    await consume(chunk.samples)
}
```

Exactly one empty final chunk follows all audio frames on success. Frame indices
increase across text chunks. No final marker is emitted after cancellation,
callback failure, invalid PCM, or a generation limit. Previously delivered chunks
are partial audio in those cases. The callback is awaited before producing the
next frame, and the pull stream produces a frame only when requested. Calls on
one model share a bounded inference lane. A request releases admission after
completion or failure; dropping a pull stream releases its request state.

`generate(text:language:)` and `generateStream(text:language:)` conform to
`SpeechGenerationModel`. `try await PocketTTSModel.fromPretrained()` downloads the
default bundle. It also supports custom cache directories and offline loading.

## HTTP and WebSocket

The server aliases `pocket`, `pocket-tts`, and `pocket-tts-100m-coreml-int8` select
this model and download its bundle on first use.
HTTP and WebSocket retain the existing Kokoro default. Generic OpenAI voice names select Alba; unsupported native voice
names or languages fail before model loading.

```bash
curl http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"pocket-tts","input":"Hello world.","voice":"alba","response_format":"pcm","stream":true}' \
  -o hello.pcm
```

Streaming HTTP requires PCM. WAV requests buffer the complete utterance.
WebSocket `/v1/realtime` with `session.model = "pocket-tts"` emits PCM16
`response.audio.delta` messages as frames arrive. HTTP writes and WebSocket
sends apply backpressure. A midstream failure terminates the response; it does
not claim successful completion. Speed changes are unsupported for Pocket.

## Validation and profiling

```bash
swift build -c release --product pocket-tts-bench --disable-sandbox
.build/release/pocket-tts-bench --bundle-dir /path/to/bundle \
  --text "The quick brown fox jumps over the lazy dog." \
  --runs 5 --warmup 1 --seed 42 --output-dir /tmp/pocket-bench

swift test --no-parallel --filter PocketTTSTests --disable-sandbox
POCKET_TTS_BUNDLE=/path/to/bundle swift test --no-parallel \
  --filter E2EPocket --disable-sandbox
```

The benchmark reports model load, first nonempty PCM, synthesis duration,
RTF (wall time / audio duration, lower is faster), current/peak RSS and physical
footprint. Model loading is excluded from synthesis timing. Warmup runs are
explicitly labeled. Use separate processes for Pocket, Kokoro and ASR so memory
figures cannot include another loaded model. See the
[benchmark report](../benchmarks/pocket-tts.md) for measured scope and limitations.
