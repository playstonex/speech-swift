# Voxtral TTS native-runtime validation — 2026-10-17

Voxtral 4B TTS produces 24 kHz speech through the Swift library, CLI, HTTP and
WebSocket server. The tested 4-bit bundle preserves upstream weights, tokenizer
and all 20 preset voice embeddings without requantization. The weights retain
CC BY-NC 4.0.

## Environment and reproducibility

- Apple M5 Pro, 48 GiB unified memory; release build.
- MLX Swift 0.31.4, normal matrix acceleration for speech and performance.
- Original model: `mistralai/Voxtral-4B-TTS-2603` at
  `b81be46c3777f88621676791b512bb01dc1cb970`.
- Quantized source: `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit` at
  `f98fc91b9cb5adc7dab56102c690458276c14c6a`.
- Python reference: `mlx-audio` at
  `441d90d77bc2fa1667bc74cae2e434dcf7427424`, MLX 0.31.2.
- Eight flow time points, guidance 1.2, seed 0. Stream chunks contain 3 codec
  frames, or 240 ms of audio, unless a different size is requested.

This is a shared development machine. An unrelated model workload was observed
while investigating timing variation. These measurements establish observed
latency and correctness; they do not establish an exclusive-GPU throughput
comparison or a performance gain over another implementation.

## Latency and throughput

Each table row is the median of three warm calls. Loading, WAV encoding,
playback and network transfer are excluded. RTF means synthesis time divided
by audio duration; lower is faster. First audio means the first nonempty PCM
chunk reaches the caller. Audible speech onset and speaker playback onset
were not measured.

| Case | Audio seconds | Buffered seconds | Buffered RTF | Stream first audio, ms | Stream seconds | Stream RTF |
|---|---:|---:|---:|---:|---:|---:|
| english-opening | 3.60 | 1.751 | 0.487 | 201 | 1.950 | 0.542 |
| english-fox | 4.08 | 1.992 | 0.488 | 236 | 2.284 | 0.560 |
| french | 2.72 | 1.313 | 0.483 | 183 | 1.471 | 0.541 |

Across two recorded warm profiles, SDK first-audio medians ranged from
146 to
236 ms.
Absolute timings varied with shared-machine activity. Buffered decoding does
not materialize streaming attention history. An alternating three-pair CLI
check around removing that unnecessary work produced identical WAV bytes and
medians of 1.80 seconds before and 1.78 seconds after. That check does not
attribute the larger run-to-run timing variation to the code change.

A localhost French utterance measured first audio at **128 ms over HTTP** and
**125 ms over WebSocket**. These are single transport measurements after warmup,
not medians. HTTP total time was 1.071 seconds; WebSocket total time was
1.068 seconds for 2.72 seconds of speech.

In a new SDK process with locally cached weights, loading took
0.579 seconds. The first PCM chunk arrived
0.760 seconds after loading began. This excludes executable startup and
downloads. A separate CLI stream measured 188 ms from synthesis start to first
chunk, 1.45 seconds of synthesis, and 3.46 seconds total process time.

## Memory

| Measurement | Peak |
|---|---:|
| Streaming SDK process RSS, `getrusage(RUSAGE_SELF)` | 2.50 GiB |
| Streaming MLX allocator | 2.75 GiB |
| Buffered MLX allocator | 2.76 GiB |
| Standalone streaming CLI RSS | 2.46 GiB |
| Standalone buffered CLI RSS | 2.46 GiB |
| Streaming CLI physical footprint | 3.33 GiB |

RSS, physical footprint and MLX allocation measure different aspects of unified
memory. The SDK performance process loaded only TTS. Neither its RSS nor the
allocator peak includes the independent ASR round-trip model. These short
utterances do not establish peak memory for maximum-length input.

## Correctness and regression coverage

| Check | Result |
|---|---|
| CI-equivalent unit suite | 1,687 executed, 42 skipped, zero failures |
| Final focused configuration, CLI, server and playback tests | 54 passed |
| Export and publication tests | 12 passed |
| Isolated real-model functional tests | 11 passed |
| Real HTTP / WebSocket transport test | Passed |
| Independent ASR round trips | Three buffered and two streamed English/French utterances; zero normalized word errors |
| Tekken prompt fixtures | Exact prompts across all nine supported languages |
| Repeated/concurrent calls | Independent request state and deterministic output |
| Active/queued cancellation, delivery errors and frame limits | Explicit failure, admission released, later calls succeed |
| Final callback backpressure | Next request waits until final chunk delivery completes |
| Eight-call memory recovery | Retained active MLX allocation grows by at most 1 MiB in buffered and streamed checks |

Buffered HTTP matches the buffered SDK PCM byte for byte. Streamed HTTP,
WebSocket and CLI match the streaming SDK output byte for byte, including final
samples. Full versus incremental decoding is checked with a numerical tolerance
because matrix shapes can select different accelerated kernels. A disconnected
HTTP client does not prevent the next request from completing.

The existing quantized-layer memory test keeps its fixture alive through the
assertion, avoiding a false release-build failure when memory decreases. No
existing ASR inference implementation was changed. Routing checks preserve
existing server defaults and Kokoro's registry order.

The CI skip list came from `.github/workflows/tests.yml`. Model suites and
benchmarks ran in separate processes, with no other package build/test/benchmark
started alongside them by this validation workflow. The full collection of
model-heavy suites was not accumulated in one process.

## Codec parity and streaming state

A fixed 40-frame code sequence yields exactly 76,800 samples. Streaming tests
split it into chunks of 1, 3, 7, 16 and 40 frames. This covers the opening,
attention-history rollover, convolution overlap and the final samples.

With `MLX_ENABLE_TF32=0` set before both reference and native processes, full
reference parity has maximum absolute error `1.48e-6`. Incremental versus full
native decoding has maximum absolute error `1.12e-5` across all five chunk sizes.
The strict test limit is `1e-4`.

Default M5 acceleration uses approximate matrix products. The fixed-code
single-frame split has maximum absolute error `0.01783` and relative RMS error
`0.00511` against full decoding. The default three-frame split has maximum error
`0.00320` and relative RMS error `0.00108`. For the actual English/French speech
round trips, maximum streamed-versus-buffered sample error was below `0.00037`.
Tests use bounded tolerances in this mode; the runtime does not change global
precision settings.

Codec keys/values and convolution overlap belong to each request. The decoder
retains local history rather than repeatedly decoding the utterance prefix.
The async iterator advances only when asked for another chunk and has no
background producer or PCM backlog. Callback streaming awaits delivery, holding
admission through the final write. An exact chunk multiple ends with an empty
final marker; convolution tails are trimmed without appending extra samples.

## Validation limits

Speech quality checks cover short English and French sentences. Prompt parity
in the other seven languages does not establish their speech quality. All 20
voices, long-form prosody and iOS device inference were not tested. The runtime
uses preset voices and complete input text; reference-audio cloning and streamed
text input are unavailable. CLI playback uses a 160 ms prebuffer; physical
playback onset remains unmeasured. An exclusive-GPU performance comparison is
still unverified.

See [inference and reproduction commands](../inference/voxtral-tts.md).
