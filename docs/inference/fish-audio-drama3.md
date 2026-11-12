# Fish Audio Drama 3 inference

Drama 3 is an experimental MLX runtime for Apple Silicon. The pinned 4-bit
checkpoint passed buffered and streamed text-only, reference-cloning and CLI
waveform checks with independent ASR on Apple M5 Pro (48 GiB). Release streaming
first PCM medians were 275 ms for text-only and 730 ms for cloning, including
reference encoding. All 16 benchmark clips passed ASR. See the
[latency and memory report](../benchmarks/fish-audio-drama3-2026-10-17.md).
Other precision variants remain unvalidated. Output is 44.1 kHz mono Float32 PCM. It shares the
`FishAudioTTS` package product with S2 Pro and has its own model class.

## CLI

```bash
speech speak "[excited] Hello from Drama 3." --engine fish-drama3 -o hello.wav
speech speak "Hello while generating." --engine fish-drama3 --stream \
  --fish-drama3-chunk-frames 3 -o streamed.wav
speech speak "Hello again." --engine fish-drama3 \
  --fish-drama3-bundle-dir /path/to/Fish-Audio-Drama-3-MLX-4bit \
  --voice-sample reference.wav \
  --fish-drama3-reference-text "The exact words in the reference recording." \
  --seed 42 -o cloned.wav
```

| Option | Default | Meaning |
|---|---|---|
| `--fish-drama3-model-id` | `mlx-community/fish-audio-drama-3-4bit` | Converted Hugging Face checkpoint |
| `--fish-drama3-bundle-dir` | — | Local bundle, bypasses download |
| `--fish-drama3-reference-text` | — | Required with `--voice-sample` |
| `--fish-drama3-max-frames` | `2048` | Generation limit; approximately 95 seconds |
| `--fish-drama3-temperature` | `1` | Sampling temperature; `0` is greedy |
| `--fish-drama3-top-k` | `30` | Number of candidates |
| `--fish-drama3-top-p` | `0.9` | Nucleus probability |
| `--stream` | `false` | Deliver PCM while codebooks are generated |
| `--fish-drama3-chunk-frames` | `3` | Frames per PCM chunk, 1–128; 3 frames contain about 139 ms of audio |
| `--fish-drama3-memory-mode` | `balanced` | Stage-boundary cache trimming and deferred reference encoder loading; `unrestricted` loads eagerly and skips stage trimming; the lifetime cap still applies |
| `--seed` | `0` | Request-local sampling seed |

The CLI accepts one complete text and writes a WAV. `--stream` decodes and
delivers incremental PCM; add `--play` for playback during generation. It prints
time to the first nonempty PCM chunk, excluding model loading and reference
file I/O. Batch-file mode, preset speakers, and a separate instruction flag are rejected.
Use inline controls such as `[laughs]` and `<|speaker:0|>` in the text. Language
is inferred from the text. Reference audio must be finite, nonempty mono PCM
and no longer than 30 seconds. `--clean-reference` uses the shared optional
reference restoration path.

## Swift

```swift
import FishAudioTTS

let model = try await FishAudioDrama3Model.fromPretrained()
// Or: try await FishAudioDrama3Model.fromBundle(bundleURL)
let pcm = try await model.generate(
    text: "<|speaker:0|> [sighs] Fine. Let's try again.",
    options: FishAudioDrama3Options(maxFrames: 512, seed: 42))

let reference = try await model.encodeReferencePrompt(
    audio: referencePCM, sampleRate: 44_100, text: "The exact reference transcript.")
let cloned = try await model.generate(
    text: "Hello in the reference voice.", references: [reference],
    options: .init(seed: 42))
```

## Memory policy

The default `memoryOptions: .balanced` evaluates intermediate codec stages and
trims idle MLX buffers when the cache exceeds 512 MiB at a stage boundary. This
is a trimming threshold, not a ceiling on total memory or temporary allocations.
Active tensors are preserved. Separately, the model holds the shared
[default MLX cache cap](mlx-memory.md) while loaded, preserving lower caller budgets.
Model loading, prompt prefill, generated frames, codec stages and request
completion provide trimming points. Model admission remains held during cleanup.

Text-only synthesis also defers materializing 794.4 MiB of reference encoder
weights. The first `encodeReferencePrompt` or raw-reference synthesis loads
them from `codec.safetensors` and retains them for subsequent cloning calls.
Keep the local bundle available while using the model. Precomputed reference
codes do not need the encoder.

```swift
let model = try await FishAudioDrama3Model.fromBundle(
    bundleURL, memoryOptions: .balanced)
// Pass .unrestricted for eager loading without stage trimming; the lifetime cap still applies.
```

For a custom policy, pass
`FishAudioMemoryOptions(cacheTrimThresholdBytes: 256 * 1024 * 1024,
lazyReferenceEncoder: true)`. A `nil` threshold disables stage evaluation and
cache trimming. Negative thresholds are rejected before loading files.
The CLI exposes the two presets through `--fish-drama3-memory-mode`.
S2 Pro keeps its existing loading and caching behavior.

## Streaming PCM

```swift
try await model.generateStreaming(
    text: "Hello in the reference voice.", references: [reference],
    options: .init(seed: 42, streamingChunkFrames: 3)
) { chunk in
    // Await playback or transport delivery here to apply backpressure.
    await receiveAudio(chunk.samples, sampleRate: chunk.sampleRate)
}

for try await chunk in model.generateStream(text: "Hello.", options: .init(seed: 42)) {
    await receiveAudio(chunk.samples, sampleRate: chunk.sampleRate)
}
```

Both APIs generate codebook frames and decode only new audio. Convolution
history and windowed attention caches belong to each request. `frameIndex` is
the first codec frame in a chunk; every frame contains 2048 samples at 44.1 kHz.
The final chunk can be empty when EOS follows an exact chunk multiple. Every
successful request has exactly one final marker. Cancellation, frame limits
and delivery failures throw; earlier delivered audio remains partial output.

The callback is awaited before generating another chunk and holds model
admission through the final delivery. The async iterator advances on demand,
without a background producer. Release an abandoned stream and its iterator
to release admission, or cancel the consuming task. Input text must be complete.

Encode a reusable reference once with `encodeReferencePrompt` and pass it via
`references`. The raw-reference `generateStreaming` overload also accepts
`referenceAudio`, `referenceSampleRate` and `referenceText`; that call includes
reference encoding before the first PCM chunk. Streaming first audio means PCM
reaches the caller; audible speech onset and device playback onset are separate.

Multiple references can carry explicit speaker tags. For continuation or speech
revision, pass previous text/codebook pairs through `history`:

```swift
let firstText = "The meeting is on Wednesday."
let first = try await model.generateCodebooks(text: firstText, options: .init(seed: 42))
let prior = try FishAudioReferencePrompt(text: firstText, codes: first.codes)
let revised = try await model.generate(
    text: "The meeting is on Thursday.", history: [prior], options: .init(seed: 42))
```

Text cannot contain structural prompt tokens or noncanonical speaker tags.
Every literal is tokenized separately to preserve the upstream BPE boundaries.
The complete prompt, including references and history, must fit 8192 positions
and the model context. Automatic long-text chunking and streamed text input are
not implemented in this runtime.

## Bundle and validation

A local bundle contains config/tokenizer metadata, `model*.safetensors`,
`codec.safetensors`, LICENSE and NOTICE. Export packages also include a tensor
index and `soniqo_manifest.json` with hashes, pinned source revision, and
validation status. The runtime rejects duplicate, missing, unexpected, and
incorrectly shaped language-model/tower parameters, and mismatched quantization
maps. Invalid rotary dimensions are rejected before weight allocation.
Raw FP8 weights must be converted before loading.

Run heavy checks sequentially. Build the test binary and debug metallib first,
following the package's test instructions. Lightweight tests:

```bash
swift test --no-parallel --filter 'FishAudioDrama3ConfigTests|FishAudioDrama3PromptTests|FishAudioMemoryOptionsTests|FishDrama3SpeakCommandTests'
MLX_ENABLE_TF32=0 swift test --no-parallel --filter E2EFishAudioDrama3MathTests
```

For the real-model checks, use an existing local bundle:

```bash
FISH_DRAMA3_E2E=1 FISH_DRAMA3_BUNDLE=/path/to/bundle \
  swift test --no-parallel --filter E2EFishAudioDrama3Tests/testTextOnlyASRRoundTrip

FISH_DRAMA3_E2E=1 FISH_DRAMA3_BUNDLE=/path/to/bundle \
  FISH_DRAMA3_REFERENCE_AUDIO=/path/to/reference.wav \
  FISH_DRAMA3_REFERENCE_TEXT="The exact reference transcript." \
  swift test --no-parallel --filter E2EFishAudioDrama3Tests/testReferenceCloningASRRoundTrip

FISH_DRAMA3_E2E=1 FISH_DRAMA3_BUNDLE=/path/to/bundle \
  FISH_DRAMA3_CLI="$PWD/.build/release/speech" \
  swift test --no-parallel --filter E2EFishAudioDrama3Tests/testCLIProducesWAVWithCorrectContent
```

The TTS model is released before loading the independent ASR checker. The
full test runner isolates E2E classes in separate processes. Do not run these
checks alongside other Swift builds, benchmarks, or model inference.
Set `FISH_DRAMA3_ARTIFACT_DIR` to retain generated WAV files and ASR transcripts.

Streaming checks use `E2EFishAudioCodecStreamingMathTests`,
`E2EFishAudioCodecStreamingTests`, `E2EFishAudioDrama3StreamingControlTests`, and
the `testStreaming*` methods in `E2EFishAudioDrama3Tests`. Codec parity runs
with `MLX_ENABLE_TF32=0`; speech and performance checks use default inference
precision. Run real-model methods in separate processes, sequentially.
