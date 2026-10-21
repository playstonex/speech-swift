# Kyutai STT benchmark and validation — 2026-10-15–16

Validated the 1B English/French checkpoint and all three exports on Apple M5 Pro, 48 GB, macOS 26.6.2. Corpus accuracy uses a Swift release build with compiled MLX Metal resources. Each language/precision ran in sequential batches of ten clips, with a fresh process and first-clip warmup per batch and a 256 MiB MLX cache limit. Clip errors are summed across all 80 clips before WER division. INT8 is the default. The tables report accuracy, speed and memory across precisions.

## Corpus accuracy

The fixed [FLEURS](https://huggingface.co/datasets/google/fleurs) subset contains 80 English clips (753.14 s) and 80 French clips (849.30 s), selected as the first 80 test rows per language in parquet order at revision `168de341b3db6859a9bac1c50a2ef5e3b47647e0`. FLOAT WAV preserves the original decoded samples and levels. These are subset results, not full-test leaderboard scores. Both references and hypotheses use speech-swift's existing normalizer: lowercase, punctuation removal and common English contractions, with no general spoken-number normalization. Empty transcripts count as deletions.

| Precision | English WER ↓ | Change, pp | French WER ↓ | Change, pp |
|---|---:|---:|---:|---:|
| FP16 | 23.85% | -0.18 | 16.44% | +0.00 |
| INT8 | 23.74% | -0.18 | 17.13% | +0.60 |
| INT4 | 24.03% | -0.53 | 17.23% | +0.00 |

Changes compare the earlier release build with the current build on `main` base `511307a` (runtime source `f9a3e0a`). The current Metal library uses Metal 3.2 and the macOS 15 deployment target. Dataset files, weights and scoring are unchanged; the cause of individual transcript changes has not been isolated. Lower WER is better: the default INT8 English score improved, while French increased by 0.60 percentage points. Changed transcripts out of 160: FP16 6, INT8 9, INT4 10.

Corrected corpus runs check accuracy. Speech models and builds remain sequential, but non-model rendering may overlap. Corpus timing fields are retained as diagnostics and are not used as isolated performance measurements. The performance and latency profiles below were measured separately, without observed competing heavy work.

## Memory and performance iterations

The earlier paired INT8 profile uses PCM16 copies of the first eight English and first eight French clips (157.7 s total). These copies are retained to compare the runtime/cache change; use the Float32 corpus above for current accuracy. All three implementations below produced identical transcripts on those clips.

| Runtime / policy | WER ↓ | Throughput ×RT ↑ | Peak RSS MiB | Peak physical footprint MiB |
|---|---:|---:|---:|---:|
| Full Mimi module | 15.72% | 7.44 | 1607 | 5000 |
| Encoder-only Mimi | 15.72% | 7.75 | 1452 | 4863 |
| Encoder-only, 256 MiB cache | 15.72% | 7.62 | 1449 | 1947 |

The runtime now constructs only Mimi's encoder, transformer, downsampler and codebooks. It omits unused audio-generation modules while retaining the original checkpoint on disk. This reduced peak resident memory without changing these transcripts. Codec and decoder work share one synchronization per frame. Decoder/quantizer graph-compilation experiments did not improve measured throughput and were removed.

`--mlx-cache-limit-mb 256` bounds MLX's reusable buffer pool; it is not a cap on total process memory. The Kyutai CLI uses this 256 MiB limit by default and restores the previous process-wide setting on return. Model APIs leave this policy to the caller. The paired cache comparison is one run per policy, so small timing differences should not be treated as statistically established.

The encoder must process its two 25 Hz frames together before downsampling. Correcting an early one-frame-at-a-time implementation changed the fixed mixed-subset INT8 WER from 20.88% to 15.72%. The accepted implementation matches the original codec's within-chunk attention and context rollover.

## CLI default-cache profiles

These current-build profiles use PCM16 copies of the first eight English and eight French clips (157.7 s) with the 256 MiB reusable-buffer cache limit. The INT8 profile preserves all 16 transcripts from the earlier build. FP16 and INT4 each change one transcript. Accuracy here is for that mixed 16-clip selection; use the Float32 corpus table for the 80-clip-per-language accuracy comparison. Model APIs leave the global cache policy to the caller.

A slower post-restart run is retained only as a diagnostic: subsequent sampling found active indexing and browser GPU work. Accepted profiles began after three baseline GPU samples at or below 3%; observed competing work caused retries.

| Precision | Profile WER ↓ | RTF ↓ | Throughput ×RT ↑ | Peak RSS MiB | Peak physical footprint MiB |
|---|---:|---:|---:|---:|---:|
| FP16 | 17.78% | 0.172 | 5.82 | 2185 | 2678 |
| INT8 | 15.72% | 0.130 | 7.72 | 1452 | 1951 |
| INT4 | 19.33% | 0.110 | 9.12 | 1060 | 1556 |

Throughput change versus the earlier build: FP16 -3.4%, INT8 +1.3%, INT4 -4.5%. Each is a single pass; these differences do not establish a statistically significant speed regression or improvement.


## Isolated CLI latency profile

Medians below use the separate 16-clip PCM16 profile with the 256 MiB cache limit.
Input is unpaced; first-text compute time is not live microphone latency.
Frame p95 is the median of the per-clip percentiles, for 80 ms audio frames.

| Precision | First text compute ms | Input consumed s | Frame p95 ms | Final flush ms |
|---|---:|---:|---:|---:|
| FP16 | 233.1 | 1.60 | 12.1 | 245.4 |
| INT8 | 179.3 | 1.60 | 9.1 | 186.6 |
| INT4 | 142.5 | 1.60 | 7.8 | 159.9 |

## Long-stream check

This earlier-build timing comparison uses a synthetic 80 s recording that repeats the shared known-speech fixture four times. It checks performance after both attention windows fill; it is not a natural long-form accuracy benchmark. All four sentences and the final word were preserved. The current-build E2E checks repeated the rollover/output assertions; the table retains the earlier timing and memory measurements.

| INT8 policy | Audio s | Inference s | WER ↓ | Frame p95 ms | Peak RSS MiB | Peak physical footprint MiB |
|---|---:|---:|---:|---:|---:|---:|
| API with MLX default | 80.0 | 10.33 | 0.00% | 13.5 | 1456 | 18936 |
| CLI default: 256 MiB cache | 80.0 | 9.91 | 0.00% | 10.7 | 1463 | 2098 |

## Regression and round-trip coverage

The [hosted package CI run](https://github.com/soniqo/speech-swift/actions/runs/38921004122)
passed 1,588 unit checks with 46 skips and no failures. Separate PersonaPlex/Hibiki
configuration, mapping, and cache checks passed 44 tests with six unavailable
PersonaPlex fixture checks skipped. The full local 147-class model matrix was
not completed: its unit phase was interrupted when a graphics render began.
The equivalent unit selection completed on the hosted runner; real-model
coverage below is specific to Kyutai and its Kokoro round trips.

After integrating `main` at `511307a`, all 19 Kyutai E2E checks passed in
separate class processes: seven INT8, six FP16, and six INT4. The two
matching-precision decoder-reference checks for FP16/INT4 were intentionally
skipped because the independent decoder reference is INT8. These runs include
the shared `StreamingRecognitionModel` adapter, rate validation, cumulative
partial/final updates, idempotent finish, abandoned-session release, known
speech, codec features, English/French round trips, and attention-window rollover.

- Before integrating `main`, 101 targeted Swift unit/logic tests passed; six existing PersonaPlex fixture tests were skipped because their fixtures were unavailable. The current-base CI and shared-module checks above supersede those counts. Coverage includes existing transcription arguments, shared Mimi defaults, PersonaPlex/Hibiki configuration and cache logic, strict tensor completeness, UTF-8 decoding, lifecycle rules, and memory-option bounds.
- 31 exporter/release-preparation tests passed, including source-sample preservation, audio-format guards, all precision layouts, rejected architectures, verified provenance, dataset identity and artifact checksums.
- The earlier 15 real-weight E2E executions were repeated after integrating `main`, with the shared streaming adapter bringing the current total to 19 above. Earlier diagnostic runs also passed three INT8 decoder checks with fixed reference audio codes and two INT8 codec-feature checks on difficult clips. Each E2E class ran in its own process. Attention caches retained bounded lengths while absolute positions continued to advance.
- The French round-trip fixture used macOS's French Thomas voice and the exact text “Bonjour, je voudrais un café avec du lait.” English round trips used cached Kokoro weights. Natural French quality is evaluated separately by the FLEURS table.
- Release CLI transcription passed for all three precisions with streaming and the default 256 MiB MLX cache limit; an explicit limit also passed. The original upstream BF16 layout also passed the known-speech CLI check.
- The corpus run exposed an existing empty-hypothesis crash in the WER scorer. A regression test now verifies full deletions, and all completed corpus scores include empty outputs.

The shared fixture transcribed as “Can you guarantee that the replacement part will be shipped tomorrow?” including its final word, with identical batch and uneven-chunk stream transcripts. Runtime bundle sizes, including the original Mimi file, tokenizer and configuration: FP16 2.201 GiB, INT8 1.483 GiB, INT4 1.100 GiB. Decoder files are 1,886.77 / 1,151.78 / 759.78 MiB respectively. Only encoding tensors from Mimi are loaded.

## Independent reference

Source: [Kyutai stt-1b-en_fr](https://huggingface.co/kyutai/stt-1b-en_fr), pinned at `1c34c6b4f7e9299bb61985f145052ff131005dde`. Weights and FLEURS data are CC-BY-4.0; attribution: Kyutai and the FLEURS contributors.

The reference uses unmodified [Kyutai Moshi](https://github.com/kyutai-labs/moshi) revision `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`: original PyTorch Mimi plus the independent MLX text decoder with matching INT8 weights. The same mono 24 kHz fixture was supplied to both runtimes. Emitted text token IDs and the complete transcript matched. Encoder, transformer and downsampler activations at frames 0, 1, 2, 3, 124, 125, 126 and 249 passed absolute tolerance 0.001; the largest observed difference was 0.000376884. This includes the codec's 10 s context boundary and the final partial frame.

The original PyTorch codec uses causal attention and exact GELU. The upstream MLX codec example differs in those settings, so it is not used as the audio-feature oracle. Later residual codebook IDs can vary at floating-point decision boundaries; continuous features are compared numerically.

Six earlier INT8 diagnostics used PCM16 copies resampled to identical mono 24 kHz FLEURS input. Four complete transcripts matched the independent reference, including both empty English outputs and the empty French output. Two quiet nonempty cases produced different text. With fixed reference audio codes, the native decoder matched every reference frame token for both differing clips and the known-speech fixture. Codec features passed the 0.001 numeric tolerance; codebook 0 matched every audio frame, while later residual-codebook decisions differed. Numeric feature tolerance therefore does not establish bit-exact audio codes or complete transcript parity on quiet speech. These diagnostics predate the current-build corpus refresh.

The diagnostic audio was resampled with ffmpeg for both runtimes; the main corpus run uses AudioFileLoader's resampler. Very quiet recordings can also be sensitive to that preprocessing difference. The two empty English recordings peak below 0.5% of full scale. The corpus benchmark preserves original levels and includes recognition failures.

## Limits

These are single-machine subset and fixture measurements. Timings are single-pass observations on a workstation with desktop applications open. They establish the listed checks and expose precision tradeoffs; they do not establish absence of regressions across every model or every recording. Natural audio longer than 60 s was not evaluated for accuracy. The 2.6B English configuration/export path has unit coverage, but real-weight 2.6B inference was not run. Semantic VAD extra-head checkpoints are unsupported.


## Input-level quality investigation

### Paired gain experiment

Original source Float32 samples were preserved. The fixed corpus is the first 80
test rows per language at FLEURS revision `168de341b3db6859a9bac1c50a2ef5e3b47647e0`.
For each complete recording, multiply all samples by `0.9 / max(abs(samples))`;
leave digital silence unchanged. No clipping, trimming, denoising or text editing
was applied. Both arms use the existing speech-swift normalizer and the same
ten-clip sequential process batches with a 256 MiB MLX cache limit.

| INT8, 80 clips each | Original levels WER ↓ | Peak 0.9 WER ↓ | Improved / unchanged / worsened clips | Empty transcripts before → after |
|---|---:|---:|---:|---:|
| English | 23.74% | 7.81% | 45 / 29 / 6 | 2 → 0 |
| French | 17.13% | 13.40% | 34 / 36 / 10 | 1 → 1 |

Whole-file peak selection uses future audio. It is not a causal streaming gain
controller, and the worsened clips remain included. No automatic normalization
was added to the model or CLI. See [input-level handling](../inference/kyutai-stt.md#input-levels).

### Preparation and scoring correction

The first benchmark prepared PCM16 copies of FLOAT sources. Very quiet recordings
therefore acquired quantization noise and bias. The preparation helper now writes
FLOAT WAV. All 160 prepared recordings were compared sample for sample with the
decoded pinned source and matched exactly; two regression cases cover quiet FLOAT
and PCM16 input. Release preparation rejects mismatched audio-format metadata.
On the earlier build, preserving source samples changed INT8 English WER from 24.50% to 23.91%. The current-build comparison above reruns both original and normalized levels. Preparation explains a small part of the error, rather than the
large improvement from gain. Historical PCM16 measurements are summarized separately from the source-preserving corpus scores.

The official Whisper English normalizer was also tested, pinned at
`86098128c0b4f24f0e2aa2994de830614b474227`. On corrected INT8 English output it gives
24.82% at original levels and
9.46% after gain on the current build. The
speech-swift score and normalizer are retained as the primary paired comparison.

### Original runtime and published references

Eight diagnostic clips were selected before reviewing the gain result: two ordinary
English clips, two quiet empty English clips, two other difficult English clips,
and the empty and wrong-script French examples. Each original-level/normalized
24 kHz Float32 WAV was supplied identically to native INT8, native FP16, and
unmodified PyTorch Mimi plus PyTorch LMGen. The source checkpoint is pinned at
`1c34c6b4f7e9299bb61985f145052ff131005dde`, and Moshi at
`e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`.
The reference used the original BF16 decoder on MPS, the original Float32 codec on
CPU, greedy generation, and the same 20-frame flush. All 48 inferences completed.
This is a diagnostic selection, not another representative WER benchmark.

The original PyTorch model reproduces both empty quiet-English outputs and the
wrong-script French failure. Gain restores the quiet-English speech there too.
Some transcripts differ across runtimes and precisions, including an empty FP16
French result after gain. Complete pipeline parity is not established for all
inputs; the earlier fixed-code decoder and continuous-feature tests isolate those
components separately. These diagnostic runs predate the current-build corpus refresh.

Kyutai's [DSM paper, Table 1](https://arxiv.org/html/2509.08753v1#S4.SS3) reports
6.4% average WER for its 2.6B English model across eight short-form OpenASR datasets,
with the Whisper English normalizer. Its [evaluation script](https://github.com/kyutai-labs/delayed-streams-modeling/blob/main/scripts/stt_evaluate_on_dataset.py)
resamples and pads audio without peak normalization. This is a different model
and dataset mix from the present 1B FLEURS subset. An official 1B FLEURS English
score was not found in the paper and model cards checked.

An [independent exporter's reference Rust/candle measurement](https://huggingface.co/stephvax/kyutai-stt-1b-en_fr-candle-gguf)
reports 17.18% WER for the original f32 1B model on 200 French FLEURS clips. It is a
different subset and implementation, so it provides context rather than an exact
regression threshold.

## Published artifacts and runtime checks

The following initial public revisions were verified on 2026-10-16. All nine runtime and
validation artifacts in each repository passed size and SHA256 checks. Each
repository includes the root configuration required for download tracking.

| Precision | Verified public revision |
|---|---|
| FP16 | [31f1207](https://huggingface.co/aufklarer/Kyutai-STT-1B-MLX-FP16/tree/31f1207b99fe89ae0d88328e9d27ab19a57f4c8e) |
| INT8 | [d0dcdfe](https://huggingface.co/aufklarer/Kyutai-STT-1B-MLX-8bit/tree/d0dcdfed4d7d74fccdf518a89e6ae2036dd8c57d) |
| INT4 | [700b676](https://huggingface.co/aufklarer/Kyutai-STT-1B-MLX-4bit/tree/700b6761c841bf10c28ca2dff0ade02d568ba383) |

The artifacts are `config.json`, `model.safetensors`, `mimi.safetensors`,
`tokenizer_en_fr_audio_8000.model`, `export.json`, `README.md`, `VALIDATION.md`,
`benchmark.json`, and `SHA256SUMS`. Published [benchmark metadata](https://huggingface.co/aufklarer/Kyutai-STT-1B-MLX-8bit/blob/d0dcdfed4d7d74fccdf518a89e6ae2036dd8c57d/benchmark.json)
and [checksums](https://huggingface.co/aufklarer/Kyutai-STT-1B-MLX-8bit/blob/d0dcdfed4d7d74fccdf518a89e6ae2036dd8c57d/SHA256SUMS)
record the initial release contract. Current corpus accuracy and CLI profiles are reported above; runtime weights, tokenizer, configuration and export provenance remain unchanged. Current per-run JSON, transcripts, logs, and profiler output
are retained in scratch storage rather than committed to this package.

Post-publication runtime checks passed six CLI runs: batch and streaming for each
published precision, all with the exact shared-fixture transcript. INT8 used the
default model selection and completed a native cold download from HuggingFace.
FP16 and INT4 used normal model caches populated with the prepared exports after
their sizes and SHA256 hashes were verified against the public repositories.
Each run used the canonical model ID and the default 256 MiB MLX buffer-cache
limit. These are functional checks, with no new performance claim.

## Reproduction

Build `asr-bench` in release mode with compiled MLX Metal resources, then run each engine/language in a separate process. A manifest is a relative WAV path, tab, and reference text; select the pinned first 80 test rows per language described above and preserve Float32 samples.

To match the reported corpus isolation, split each manifest into ten-row TSV files in the same directory and run them sequentially. Sum substitutions, insertions, deletions and reference words across batches before calculating WER; do not average batch percentages. The commands below show the full-manifest form.

```bash
asr-bench --dataset fleurs-en_us-80.tsv --engines kyutai-stt-mlx-int8 --mlx-cache-limit-mb 256 --output english-int8.json
asr-bench --dataset fleurs-fr_fr-80.tsv --engines kyutai-stt-mlx-int8 --mlx-cache-limit-mb 256 --output french-int8.json
# Separate memory-policy profile:
asr-bench --dataset fleurs-en-fr-16.tsv --engines kyutai-stt-mlx-int8 \
  --mlx-cache-limit-mb 256 --output cache256-int8.json
```

Use `kyutai-stt-mlx-fp16` or `kyutai-stt-mlx-int4` for the other precisions. Local paths can override downloads through `KYUTAI_STT_FP16_MODEL_PATH`, `KYUTAI_STT_INT8_MODEL_PATH` and `KYUTAI_STT_INT4_MODEL_PATH`. Keep model processes and Swift builds sequential. Isolated performance runs must also avoid rendering jobs; the accuracy-only corpus sweeps explicitly permit non-model rendering. Timing runs interrupted by observed competing work or system sleep were discarded and retried.
