# MOSS Transcribe Diarize Inference

## Choose a backend

```text
Audio
  -> 16 kHz mono
  -> native Whisper log-mel frontend
  -> 30-second audio-encoder chunks
  -> one audio-aware MOSS prompt
  -> autoregressive Qwen3 decoder
  -> timestamped speaker segments + plain text
```

Use MLX for the upstream 128K long-context behavior. Use Core ML for short
recordings where Neural Engine execution and its fixed 1,024-token state are
more important.

Both paths are offline. `--stream` is rejected because MOSS needs the complete
recording before it can assemble the global prompt and begin transcript
generation. A live microphone workflow should use a streaming ASR model and a
separate online diarization strategy instead.

## CLI

Core ML INT8 remains the default for compatibility:

```bash
.build/release/speech transcribe recording.wav --engine moss
.build/release/speech transcribe recording.wav \
  --engine moss --backend coreml --model fp16
```

Select the long-context MLX runtime. INT5 is the MLX default:

```bash
.build/release/speech transcribe meeting.wav \
  --engine moss \
  --backend mlx

.build/release/speech transcribe meeting.wav \
  --engine moss \
  --backend mlx \
  --model int8
```

`--model` also accepts a Hugging Face repository ID or a local bundle
directory:

```bash
.build/release/speech transcribe meeting.wav \
  --engine moss \
  --backend mlx \
  --model /path/to/MOSS-Transcribe-Diarize-0.9B-MLX-5bit
```

MLX generates at most 5,120 transcript tokens by default; Core ML defaults to
512. Increase the MLX limit for a dense long recording while keeping the
combined 131,072-token context in mind:

```bash
.build/release/speech transcribe meeting.wav \
  --engine moss \
  --backend mlx \
  --max-tokens 16384
```

## Swift API

```swift
import MossTranscribe

let model = try await MossMLXModel.fromPretrained(variant: .int5)
try model.warmUp()

let options = MossMLXDecodingOptions(
    maxTokens: 5_120,
    encoderBatchSize: 4,
    prefillChunkSize: 512,
    kvCachePrecision: .float16
)
let result = try model.transcribeDetailed(
    audio: samples,
    sampleRate: 48_000,
    options: options
)

print(result.text)
for segment in result.segments {
    print(segment.startTime, segment.endTime, segment.speaker, segment.text)
}
```

Use `.int8` for the MLX quality reference:

```swift
let reference = try await MossMLXModel.fromPretrained(variant: .int8)
```

Load a self-exported directory without network access:

```swift
let local = try await MossMLXModel.fromDirectory(
    URL(fileURLWithPath: "/path/to/moss-mlx")
)
```

The existing `MossTranscribeModel` API remains the Core ML path.

## What “single pass” means

The encoder still operates on Whisper-sized 30-second chunks. The MLX runtime
does not transcribe those chunks independently:

1. It encodes chunks in bounded batches.
2. It concatenates every resulting audio embedding in chronological order.
3. It prefills the decoder with one prompt containing the complete recording.
4. It generates one timestamped, speaker-attributed transcript
   autoregressively.

This provides global speaker context across the recording. It does not mean
all text tokens are produced simultaneously, and it does not provide partial
results while audio is arriving.

## Context and memory

While a MOSS MLX model is loaded, its reusable-buffer cache ceiling is
`min(2 GiB, 25% of physical RAM)`. The ceiling applies before loading weights
and remains in effect during warm-up and transcription. Lower caller limits,
including `MLX.Memory.cacheLimit = 0` to disable caching, are preserved.
MOSS and Qwen3-ASR share ownership of this process-wide setting: the tightest
loaded model's ceiling wins, and releasing the final model restores the prior
caller budget unless the caller changed it in the meantime.

This bounds unused reusable buffers, not total process memory. Live decoder KV
tensors, the waveform, features, and audio embeddings still grow with recording
length; model weights and other runtime allocations are additional.
`Memory.clearCache()` between encoder batches and after transcription releases
cached buffers but cannot replace a decoder-time cache ceiling.

See the [controlled cache-limit validation](../benchmarks/moss-mlx.md#cache-limit-validation-2026-10-24)
for before/after output comparisons, physical footprint, and known limitations.

Audio consumes 12.5 context tokens per second. Ninety minutes is about 67,500
audio tokens before prompt and output tokens. Approximate KV-cache storage at
that audio-token count is:

| KV cache | Estimated storage |
|---|---:|
| FP16 | 7.74 GB |
| INT8 affine | 4.11 GB |

These analytical values cover the 28 decoder layers' key/value tensors and
affine metadata; model weights, audio embeddings, temporary tensors, allocator
overhead, and generated transcript tokens are additional.

Use FP16 cache for highest fidelity and for INT5/INT8 weight comparisons.
Select a lower-memory cache independently:

```bash
.build/release/speech transcribe meeting.wav \
  --engine moss \
  --backend mlx \
  --kv-cache int8
```

An earlier E2E check preserved the structured transcript with INT8 cache.
Validation with MLX Swift 0.31.4 now reproduces missing speaker labels on the
short fixture with both INT5 and INT8 weights, on both the original runtime and
the buffer-cache-limited runtime. Use the default FP16 KV cache for structured
speaker output; the buffer-cache ceiling does not fix quantized KV quality.
INT4 cache is intentionally not exposed: it dropped timestamp structure and
repeated a word on the earlier quality gate. Long-form INT8-cache quality has
not yet been included in the published INT5/INT8 comparison.

## Structured output

`MossTranscription` contains:

- `rawText`: the generated `[start][Sxx]text[end]` form;
- `text`: plain text formed from valid segments;
- `segments`: `[MossTranscriptSegment]`;
- `metrics`: preprocessing, encoder, prefill, decode, token counts, total time,
  RTF, and stop reason.

`diarizedSegments(audioDuration:)` converts anonymous speaker strings to
contiguous integer speaker IDs for `AudioCommon` consumers and the
diarization benchmark.

## Benchmarking

MLX engines are registered in both benchmark binaries:

```bash
.build/release/asr-bench \
  --dataset english.tsv \
  --engines moss-mlx-int5 moss-mlx-int8 \
  --output moss-mlx-asr.json

.build/release/diarization-bench \
  --manifest voxconverse.tsv \
  --engines moss-mlx-int5 moss-mlx-int8 \
  --collar 0.25 \
  --output moss-mlx-diarization.json
```

Run engines in separate processes for publishable peak-memory numbers. Local
bundles can be selected with `MOSS_MLX_INT5_MODEL_DIR` and
`MOSS_MLX_INT8_MODEL_DIR`.

See [MOSS MLX benchmark](../benchmarks/moss-mlx.md) for matched results and
[MOSS model architecture](../models/moss-transcribe-diarize.md) for export
details.

## Current scope

- Greedy decoding only; no beam search or temperature fallback.
- Model-generated timestamps and speaker labels, not a separate diarizer.
- Offline batch inference only; no published streaming checkpoint for this
  MOSS model.
- 131,072 combined prompt/output tokens for MLX.
- 1,024 combined prompt/output tokens for Core ML.
