# Breeze TTS 2 inference

Add the `BreezeTTS` SwiftPM product. The default bundle is
`aufklarer/Breeze-TTS-2-MLX-8bit`. The converted model and self-hosted audio are for
research and non-commercial use under the upstream model license.

```swift
import BreezeTTS

let model = try await BreezeTTSModel.fromPretrained()
let samples = try await model.generate(
    text: "The quick brown fox jumps over the lazy dog.",
    instruction: "A clear, calm English narrator.")

try await model.generateStreaming(
    text: "你好，欢迎使用语音合成。",
    instruction: "温柔清晰的普通话女声。"
) { chunk in
    // chunk.samples is mono Float32 PCM at chunk.sampleRate (24,000 Hz).
    // An empty final chunk can mark completion when no samples remain.
}
```

`fromBundle(_:)` loads a local exported directory without network access.
`fromPretrained(modelId:cacheDir:offlineMode:downloadOptions:progressHandler:)` uses the package's
Hugging Face cache. One model instance permits one generation at a time; an
overlapping request fails. Cancellation and a non-EOS frame cap propagate errors.
The shared `SpeechGenerationModel` protocol supports plain synthesis and streaming.

```bash
speech speak "The quick brown fox jumps over the lazy dog." --engine breeze \
  --instruct "A clear, calm English narrator." --stream -o design.wav

speech speak "Hello, world!" --engine breeze \
  --breeze-model-id aufklarer/Breeze-TTS-2-MLX-bf16 -o bf16.wav

speech speak "The weather is beautiful today." --engine breeze \
  --voice-sample reference.wav --breeze-reference-text "The exact reference transcript." \
  --breeze-seed 42 -o clone.wav

speech speak "We need to discuss what happened." --engine breeze \
  --voice-sample reference.wav --breeze-reference-text "The exact reference transcript." \
  --instruct "Speak slowly, with a serious tone." --stream -o directed.wav
```

Input is English or Chinese; the model infers language from text. An optional
`--language` validates English/Chinese aliases and does not insert a language
token. Match the voice instruction to the text's language. A reference clip must
be nonempty, finite, no longer than 30 seconds, and accompanied by its complete
transcript. The CLI loads/resamples audio; the Swift API expects 24 kHz PCM.
`--clean-reference` optionally restores the reference with Sidon before encoding.

| Flag | Default | Meaning |
|---|---|---|
| `--breeze-model-id` | `aufklarer/Breeze-TTS-2-MLX-8bit` | Converted Hugging Face bundle |
| `--breeze-bundle-dir` | unset | Local bundle, overrides model ID |
| `--breeze-cache-limit-mib` | unset | Process-wide reusable MLX cache budget in MiB; 0 disables caching |
| `--instruct` | unset | Voice design or reference voice direction |
| `--breeze-reference-text` | unset | Exact transcript of `--voice-sample` |
| `--breeze-temperature` | `0.9` | Temperature; zero is greedy |
| `--breeze-top-k` | `50` | Top-k; zero disables |
| `--breeze-top-p` | `1` | Nucleus threshold in `(0,1]` |
| `--breeze-cfg-scale` | `4` | Instruction guidance; one disables CFG |
| `--breeze-max-frames` | `750` | 80 ms frames; reaching the cap before EOS throws |
| `--breeze-seed` | `42` | Local sampling seed, including zero |
| `--first-chunk-frames` | `3` | Frames before first waveform chunk |
| `--chunk-frames` | `25` | Subsequent chunk size |
| `--stream` | off | Emit audio chunks and print first-audio latency |

For `--stream --play`, the CLI buffers at least 160 ms or one output chunk,
whichever is larger, before playback.
Use `--chunk-frames 3` to reduce that playback buffer. The default 25-frame
chunks favor throughput; generation TTFA measures PCM delivery and excludes
playback buffering.

Inline vocal events follow upstream text syntax, for example `(sigh)` in English
or `[笑]` in Chinese. Batch files, preset speaker IDs, and speaker listing are
unsupported. Partial streamed audio may precede a later cancellation or frame-limit
error; consumers should only mark an utterance complete after the stream succeeds.

BF16 projections with two input vectors use a batched matrix-vector operation.
This targets the small guidance batches in the depth decoder without changing
weights or sampling settings. Floating-point rounding can change seeded audio.
Set `SPEECH_SWIFT_BREEZE_GEMV=0` before loading a model to compare the original
projection path. Other precisions retain their existing projection path. See
the [benchmark report](../benchmarks/breeze-tts.md) for measured results and
the isolated performance and regression checks.

```bash
python3 scripts/export_breeze_mlx.py --bits 8 --output out/Breeze-TTS-2-MLX-8bit
python3 scripts/export_breeze_mlx.py --bits 0 --output out/Breeze-TTS-2-MLX-bf16
```

The exporter requires `mlx`, `numpy`, and `huggingface_hub`. `export_config.json`
records source revision, removed tensors, quantized layers, and file checksums.
Use the [benchmark procedure](../benchmarks/breeze-tts.md) before publishing a
converted bundle. Upstream NVIDIA latency claims do not describe this Swift port.

For lower playback startup latency, request one initial frame and two frames
per later chunk:

```bash
speech speak "The quick brown fox jumps over the lazy dog." --engine breeze \
  --instruct "A clear, calm English narrator." \
  --first-chunk-frames 1 --chunk-frames 2 --breeze-cache-limit-mib 1024 \
  --stream -o fast-start.wav
```

On the measured M5 Pro, warm first PCM with default 3 / 25 chunks was 178 ms
for 8-bit (RTF 0.535), and 272 ms for BF16 (RTF 0.868). BF16 with one initial
frame reached first PCM at 138 ms (RTF 0.862). TTFA excludes model loading and
speaker playback; 25-frame follow-ups still require a larger playback buffer.
See the [benchmark report](../benchmarks/breeze-tts.md) for matched baseline
controls, memory peaks, and continuous-playback estimates.

The package's shared model policy caps reusable MLX cache at the smaller of
2 GiB and one quarter of physical memory, while honoring a tighter application
budget. The optional `--breeze-cache-limit-mib` sets that process-wide budget
before loading; zero disables caching. It can tighten the package ceiling and
does not limit live model tensors. Smaller chunks add codec work and can retain
more reusable Metal buffers.

On the 9.04-second 8-bit paragraph, 1 / 2 chunks with a 1 GiB cache budget gave
103 ms first PCM, RTF 0.761, and 6.10 GiB peak physical footprint. The estimated
CLI prebuffer-ready time was 190 ms, without additional buffering required by
the observed chunk schedule. These chunk-arrival estimates exclude render
callback and output-device latency. Results are specific to the measured
hardware, text, and settings.

## Regression checks

The E2E suite loads the published default bundle when `BREEZE_E2E_BUNDLE` is
unset. It checks English and Chinese streaming against independent ASR, official
PyTorch numerical fixtures, frame-limit errors, and overlapping-request rejection
with recovery after callback cancellation. The CLI E2E test also uses the published
default unless a local bundle override is supplied.

After building tests and the debug metallib, run the model suites in isolation:

```bash
SPEECH_CLI="$PWD/.build/release/speech" E2E_ONLY_FILTER=Breeze E2E_SKIP_UNIT=1 \
  scripts/test_e2e_isolated.sh
```

Build the release `speech` product first for the CLI check. Set
`BREEZE_REFERENCE_WAV` and `BREEZE_REFERENCE_TEXT` to a reference file and its
exact transcript to include cloning and direction. Set `BREEZE_E2E_BUNDLE` to a
local BF16 or 8-bit export when testing a specific conversion. Run these checks
sequentially with other model tests and benchmarks.

`E2EBreezeTokenizerTests` compares every retained added token, representative
unused placeholders, and mixed Unicode inputs against the original tokenizer when
`BREEZE_E2E_BUNDLE` is set. Ordinary text uses the compact matching expression; literal `<unused…>`
inputs retain all original added-token rules through the full-tokenizer fallback.
