# Kyutai STT inference

Add the `KyutaiSTT` SwiftPM product. This is a native MLX backend for Apple Silicon, with no Python runtime required for transcription.

```bash
# 1B English/French INT8 export, downloaded on first use.
speech transcribe recording.wav --engine kyutai

# Incremental text output from the model's native stream.
speech transcribe recording.wav --engine kyutai --stream

# A local FP16, INT4, or INT8 export.
speech transcribe recording.wav --engine kyutai --model /models/kyutai-int8

# Precision choices and the original upstream bundle.
speech transcribe recording.wav --engine kyutai --model int4
speech transcribe recording.wav --engine kyutai --model fp16
speech transcribe recording.wav --engine kyutai --model upstream

# Explicitly select the CLI's default reusable-buffer pool limit.
speech transcribe recording.wav --engine kyutai --mlx-cache-limit-mb 256

# Upstream English-only model (larger; requires more memory).
speech transcribe recording.wav --engine kyutai --model 2.6b
```

`--model` accepts `1b` (INT8), `int8`/`8bit`, `int4`/`4bit`, `fp16`, `upstream`, `2.6b`, a compatible HuggingFace repository ID, or an existing local directory. The default shared `0.6B` alias resolves to the Kyutai 1B INT8 model when `--engine kyutai` is selected. `--language` and `--context` are rejected because Kyutai has no prompt mechanism for them. Streaming processes audio continuously; VAD segmentation and `--max-segment` are not used for this backend. `--partial` is accepted with `--stream`, which already emits each available text piece. The 2.6B configuration is supported but real-weight inference has not been validated.

| Precision | Repository |
|---|---|
| INT8 (default) | [aufklarer/Kyutai-STT-1B-MLX-8bit](https://huggingface.co/aufklarer/Kyutai-STT-1B-MLX-8bit) |
| INT4 | [aufklarer/Kyutai-STT-1B-MLX-4bit](https://huggingface.co/aufklarer/Kyutai-STT-1B-MLX-4bit) |
| FP16 | [aufklarer/Kyutai-STT-1B-MLX-FP16](https://huggingface.co/aufklarer/Kyutai-STT-1B-MLX-FP16) |

`--mlx-cache-limit-mb` controls the process-wide pool of reusable MLX buffers. Kyutai CLI commands default to 256 MiB; zero disables that pool. The CLI restores the previous setting on return, including failures. Model APIs now apply the shared default of `min(2 GiB, physical RAM / 4)` while loaded and preserve a lower caller budget; see [MLX memory policy](mlx-memory.md). In the paired 16-clip INT8 profile, a 256 MiB limit reduced peak physical footprint from 4,863 to 1,947 MiB with identical transcripts and 1.6% lower throughput in that single pair. RSS and physical footprint are different measurements; see the [benchmark report](../benchmarks/kyutai-stt.md).

Applications can set the same process-wide policy explicitly before loading models:

```swift
import MLX
Memory.cacheLimit = 256 * 1024 * 1024
```

## Swift

```swift
import AudioCommon
import KyutaiSTT

let model = try await KyutaiSTTModel.load()  // Or load("/models/kyutai-int8")
// Other precisions: KyutaiSTTModel.load(KyutaiSTTVariant.int4.modelId)
let audio = try AudioFileLoader.load(
    url: recordingURL, targetSampleRate: model.inputSampleRate)
let text = try model.transcribe(audio: audio, sampleRate: 24_000)
```

The throwing `transcribe` overload checks the sample rate and rejects nonfinite samples, resamples when needed, pads the last partial frame, and flushes delayed text. Empty audio returns an empty transcript without inference. `SpeechRecognitionModel` conformance supplies the package's nonthrowing `transcribe(audio:sampleRate:language:)` convenience; it returns an empty string on invalid input and does not condition on the language argument.

For live mono PCM, resample the capture stream to 24 kHz before appending samples. Arbitrary chunk boundaries are supported:

```swift
let session = try model.makeSession()
defer { session.cancel() }
for chunk in incoming24kHzPCM {
    for token in try session.append(chunk) {
        print(token.text, terminator: "")
    }
}
for token in try session.finish() {
    print(token.text, terminator: "")
}
let completeTranscript = session.text
```

Call `finish()` even when input ends on a full 1,920-sample frame. The model predicts delayed text, so its last words can arrive during the silence flush. `finish()` is idempotent; appending after finish or cancellation throws. Cancel an abandoned session to allow the model to be reused. Only one session may own a model at a time, and all calls on that model/session must be serialized. Back-to-back sessions reset the encoder, downsampler, and attention state.

`KyutaiSTTToken.timestamp` is an approximate timestamp in seconds, corrected for the actual number of silence-prefix frames and the model's trained delay, and clamped to zero. It is not word-level forced alignment.

The model also conforms to `StreamingRecognitionModel` for shared capture and
file sources. Its adapter emits cumulative text with segment index zero, then a
final update after the delayed-word flush. It has no end-of-utterance head and
ignores language hints. The same ownership and serialization rules apply:

```swift
let source = AudioFileLoader.stream(
    url: recordingURL,
    options: AudioFileStreamOptions(targetSampleRate: 24_000))
let session = try model.makeStreamingSession(language: nil)
for try await chunk in source {
    for update in try session.push(chunk) {
        print(update.text, update.isFinal)
    }
}
for update in try session.finish() {
    print(update.text, update.isFinal)
}
```

## Input levels

The runtime passes PCM levels through without automatic gain control. Very quiet
recordings can produce large errors or empty transcripts. Preserve Float32 samples
when preparing evaluation audio; converting quiet source audio to PCM16 adds
quantization noise.

For an offline file, a constant gain selected from the complete recording can
improve recognition. This example targets a peak of 0.9 and caps gain at 60 dB:

```swift
let peak = audio.map { abs($0) }.max() ?? 0
let gain: Float = peak > 0 ? min(0.9 / peak, 1_000) : 1
let text = try model.transcribe(
    audio: audio.map { $0 * gain }, sampleRate: 24_000)
```

Whole-file peak normalization requires the complete recording. Live streaming
needs a causal gain policy; this experiment does not validate an automatic
streaming gain controller. See the [quality investigation](../benchmarks/kyutai-stt.md#input-level-quality-investigation)
for paired scores, original PyTorch checks, and published-reference comparisons.

## Validation

Run configuration, tensor mapping, cache, and tokenizer tests without downloading models:

```bash
swift test --no-parallel --filter KyutaiSTTTests --skip E2E --disable-sandbox
```

Run the real-weight regression class separately after building tests and the MLX metallib:

```bash
KYUTAI_STT_MODEL_PATH=/models/kyutai-int8 \
  swift test --skip-build --no-parallel --filter E2EKyutaiSTTTests
```

The tests check known speech including its final word, identical file/stream transcripts across arbitrary input chunk boundaries, session ownership, reset, and cancellation. `E2EKyutaiSTTRoundTripTests` synthesizes two English sentences with cached Kokoro weights and accepts a separate French synthetic fixture; `E2EKyutaiSTTLongStreamTests` crosses both attention windows and checks bounded state and the final word. Run each class in its own process, with `KYUTAI_STT_MODEL_PATH` set. Never overlap model-loading tests with another model process or benchmark. See [architecture and weights](../models/kyutai-stt.md) for the artifact contract and supported precisions.

On macOS with the French Thomas voice installed, prepare the known-transcript round-trip fixture:

```bash
say -v Thomas --data-format=LEF32@24000 --file-format=WAVE \
  -o /tmp/kyutai-french-roundtrip.wav 'Bonjour, je voudrais un café avec du lait.'
KYUTAI_STT_MODEL_PATH=/models/kyutai-int8 \
KYUTAI_STT_FRENCH_ROUNDTRIP_AUDIO_PATH=/tmp/kyutai-french-roundtrip.wav \
  swift test --skip-build --no-parallel --filter E2EKyutaiSTTRoundTripTests
```

For independent parity checks, the optional `KYUTAI_STT_AUDIO_PATH` selects a
mono 24 kHz fixture, `KYUTAI_STT_REFERENCE_PATH` selects the reference JSON,
and `KYUTAI_STT_CODEC_REFERENCE_PATH` selects its adjacent safetensors file.
The suite compares text token IDs with the reference decoder and continuous
codec activations with original PyTorch (absolute tolerance 0.001). Later
residual codebook decisions can differ at floating-point boundaries.

`E2EKyutaiSTTDecoderReferenceTests` isolates text decoding from those codebook
decisions. Set `KYUTAI_STT_DECODER_REFERENCE_PATH` to matching-precision reference
JSON and run that class separately. It supplies reference audio codes and text
history and compares every frame's predicted token. For codec diagnostics,
`KYUTAI_STT_CODEC_CODES_REFERENCE_PATH` optionally selects the same JSON while
running `testMimiReferenceStages`; the test reports codebook differences alongside
the numeric feature checks. Quiet-input transcript differences are recorded in
the validation report.

[Recorded 1B validation results](../benchmarks/kyutai-stt.md) cover FP16, INT4, INT8, and upstream-layout loading.
