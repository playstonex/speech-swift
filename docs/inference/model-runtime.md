# Optional model runtime

`SpeechRuntime` is a model-independent Swift Package product providing
`InferenceGate`, `ModelCache`, `ModelRuntime<Model>`, and `ModelDescriptor`.
`Qwen3ASRRuntime` adapts Qwen3-ASR MLX and full CoreML. Existing model products
and synchronous APIs remain available.

```swift
import Qwen3ASRRuntime

let runtime = Qwen3ASRRuntime(descriptor: Qwen3ASRDescriptors.small)
try await runtime.prepare() // optional: exclude loading from inference timing
let text = try await runtime.transcribe(
    audio: samples, sampleRate: 16000, language: "english")
try await runtime.unload()
```

`Qwen3ASRDescriptors.coreML` selects CoreML encoder and decoder loaders. Context
hints require the MLX adapter. The native hybrid encoder/decoder API remains
available independently.

## Ownership

`ModelRuntime` loads lazily, keeps one model, and serializes its operations.
`unload()` waits for active inference before releasing the model and running
its optional cleanup hook. The Qwen adapter clears MLX parameters on unload.

`InferenceGate` admits one operation and queues up to 32 waiting operations by
default. Waiting is FIFO and cancellable; excess admission throws
`InferenceGate.AdmissionError.queueFull`. Hold a permit through consumption of
streaming output. Returning a stream from a closure does not extend that
closure's permit lifetime.

`ModelCache` coalesces loads by concrete type and identity. Failed loads are
removed so a later request can retry. Its actor protects cache metadata, not
model inference. Cached server variants remain resident for the server's
lifetime; admission limits are not a total model-memory budget.

## Cancellation and errors

Managed Qwen3-ASR validates input and checks cancellation before preprocessing,
between model stages, and at token decode boundaries. An already-submitted
device operation completes before cancellation is observed. Native decoding
defaults and token selection remain unchanged. Generic runtime operations
must provide their own cancellation checkpoints inside inference.

Managed CoreML transcription releases autoreleased scratch buffers after every
inference. Repeated calls within one background task do not depend on the
caller returning to a run loop to reclaim these temporary allocations. This
does not impose a total memory limit on loaded models or compiled graphs.

The default `SpeechGenerationModel.generateStream` cancels its producer when
the consumer terminates. A generator must cooperate with cancellation.

## CLI and server

Buffered Qwen3 MLX and full CoreML transcription use the same descriptors and adapter
across CLI, HTTP and WebSocket. Existing CLI flags, defaults and model aliases
are preserved. VAD streaming and hybrid CLI paths retain their native implementations.
The model inventory adds `backend`, `input_sample_rate`, and
`streaming` metadata for Qwen3-ASR; `buffered` denotes complete-utterance output.

The audio server shares one bounded inference lane across native HTTP, OpenAI
audio routes and realtime WebSocket operations. Streaming responses hold their
permit through the final chunk. Lazy model factories coalesce concurrent loads.
Inference failures produce HTTP errors or realtime `error` events; successful
empty transcripts remain successes. HTTP ASR queue overflow returns 503.
Existing OpenAI model-name fallback behavior remains unchanged.

## Regression checks

No-download tests cover shared loading, exclusive execution, cancellation,
load retry, unload ordering, HTTP/WebSocket errors, successful empty transcripts,
and exact playback sample ordering after ring overflow. Real-weight
`E2EQwen3ASRRuntimeTests` check native/managed transcript parity and reloads.
`E2EQwen3ASRCoreMLRuntimeTests` also check bounded scratch-memory growth across
repeated requests from one background task. The standalone Swift 6 Thread
Sanitizer stress check is `python3 scripts/profiling/check_runtime_thread_safety.py`.
Run model-loading suites sequentially in separate processes.
