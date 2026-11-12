# MOSS MLX INT5 vs INT8 Benchmark

## Recommendation

Use `moss-mlx-int5` by default. Against the matched INT8 export it reduced the
two weight files from 1,200.1 MiB to 987.0 MiB, lowered peak RSS by about
211–217 MiB, and ran faster. English aggregate WER increased by 0.28
percentage points. Aggregate diarization error on the five-file slice was
effectively unchanged: 28.05% DER for INT5 versus 28.02% for INT8.

Use `moss-mlx-int8` when the small measured ASR-quality advantage matters more
than decoder size and memory.

## Controlled setup

- Date: 2026-07-24
- Machine: MacBook Pro, Apple M5 Pro (18 cores), 48 GB unified memory
- Build: release, compiled MLX Metal library
- Source checkpoint:
  `OpenMOSS-Team/MOSS-Transcribe-Diarize` at
  `e6d68cdfcddbdad1a7e8454f0cb859cad76e2502`
- Audio encoder and VQ adaptor: identical FP16 weights
- Decoder: affine group-64 INT5 or INT8
- Decode: greedy
- KV cache: FP16 for both variants
- Each engine: separate process, one benchmark at a time

The comparison isolates decoder weight precision. It does not compare
quantized KV-cache quality. A separate short-fixture integration check found
that INT8 cache preserved the exact structured transcript. INT4 cache dropped
timestamp structure and repeated a word, so the runtime does not expose it.

## ASR

The ASR fixture contains the first 80 English-US test clips from
[`google/fleurs`](https://huggingface.co/datasets/google/fleurs), 759.56
seconds of audio and 1,813 reference words.

| Variant | Aggregate WER | CER | Overall ×RT | Elapsed | Peak RSS |
|---|---:|---:|---:|---:|---:|
| INT5 | 8.55% | 6.36% | 35.2× | 21.60 s | 1,196 MiB |
| INT8 | **8.27%** | **5.98%** | 31.6× | 24.04 s | 1,407 MiB |

INT5 changed the aggregate error counts from 150 to 155 word errors. On this
fixture, the 213 MiB smaller decoder is a measurable speed and memory win with
a small ASR-quality cost.

## Speaker diarization

The diarization fixture contains five recordings from the CC-BY-4.0
[`ggfox00000/dia-voxconverse-test`](https://huggingface.co/datasets/ggfox00000/dia-voxconverse-test)
mirror of VoxConverse v0.3 (`aepyx`, `aggyz`, `aiqwk`, `aorju`, `auzru`),
totaling 2,346.56 seconds. The files contain 4, 13, 7, 12, and 8 reference
speakers. Scoring uses a 0.25-second collar and 10 ms resolution.

This is not the Community-1 five-file development slice used in
[`moss-coreml.md`](moss-coreml.md); the two DER figures are not directly
comparable.

| Variant | DER | JER | Miss | False alarm | Speaker error | Count accuracy | Overall ×RT | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INT5 | 28.05% | 26.56% | 22.96% | **0.93%** | 4.16% | **40%** | **15.5×** | **1,281 MiB** |
| INT8 | **28.02%** | **25.79%** | 23.14% | 1.25% | **3.62%** | 20% | 11.8× | 1,498 MiB |

| Reference speakers | INT5 DER | INT8 DER |
|---:|---:|---:|
| 4 | 6.95% | 6.97% |
| 7 | **24.32%** | 30.52% |
| 8 | **7.81%** | 8.11% |
| 12 | 46.55% | **45.40%** |
| 13 | 3.18% | 3.18% |

The pooled DER difference is 0.03 percentage points, which is not meaningful
on five files. Quantization changed individual decoding trajectories: INT5 was
better on the 7-speaker recording and worse on the 12-speaker recording. The
12-speaker, 20-minute file dominates both aggregates through missed speech.

This slice validates multi-chunk global-context execution but is too small for
a general diarization-quality claim. A full VoxConverse run and multilingual
speaker-attributed transcription evaluation remain needed.

## Weight export

| Artifact | INT5 | INT8 |
|---|---:|---:|
| `audio_encoder.safetensors` | 596.0 MiB | 596.0 MiB |
| `decoder.safetensors` | 391.0 MiB | 604.1 MiB |
| Combined | 987.0 MiB | 1,200.1 MiB |
| Validation cosine | 0.99897182 | 0.99998456 |
| Validation max absolute error | 0.02246094 | 0.00268555 |

The self-exported INT8 bundle was compared tensor by tensor with the earlier
published INT8 bundle: all 373 audio tensors and all 704 exported decoder
tensors were equal. File hashes differ because safetensors header ordering and
metadata serialization differ.

## Reproduce

Prepare the FLEURS manifest:

```bash
python3.11 scripts/prepare_fleurs_asr_manifest.py \
  --fleurs-dir /path/to/fleurs/en_us \
  --output /path/to/fleurs-en-us-80.tsv \
  --limit 80
```

Run each engine separately:

```bash
MOSS_MLX_INT5_MODEL_DIR=/path/to/int5 \
.build/release/asr-bench \
  --dataset /path/to/fleurs-en-us-80.tsv \
  --engines moss-mlx-int5 \
  --output moss-int5-asr.json

MOSS_MLX_INT8_MODEL_DIR=/path/to/int8 \
.build/release/diarization-bench \
  --manifest /path/to/voxconverse.tsv \
  --engines moss-mlx-int8 \
  --collar 0.25 \
  --resolution 0.01 \
  --output moss-int8-diarization.json
```

Repeat the commands with the other engine. Do not overlap model-loading
benchmarks; separate processes are required for comparable peak memory.


## Cache-limit validation (2026-10-24)

This follow-up validates the allocator-cache fix for issue #519, separately
from the July weight-precision comparison above. It used an M5 Pro with
48 GiB unified memory, release builds, compiled Metal shaders, MLX Swift
0.31.4, greedy decoding, and FP16 KV caches. Each model ran in its own
process, with model-loading runs strictly sequential.

Against the original runtime at `6e6fd02`, both MOSS weight variants produced
exactly the same text, raw output, speaker segments, timestamps, prompt/output
token counts, and stop reasons on 80 English FLEURS clips: 160 identical
structured outputs. The shared-coordinator refactor also preserved all 80
Qwen3-ASR 0.6B INT4 transcripts.

| Short-clip workload | Original transcription time | Capped runtime | Output comparison |
|---|---:|---:|---|
| MOSS INT5, 80 clips | 21.79 s | 21.29 s | 80/80 exact |
| MOSS INT8, 80 clips | 22.72 s | 23.07 s | 80/80 exact |
| Qwen3-ASR 0.6B INT4, 80 clips | 12.37 s | 10.54 s | 80/80 exact |

These single-run timings are regression checks, not a statistically established
speedup. This FLEURS subset is separate from the July quality benchmark.

For long inputs, the uncapped original exceeded a 28 GiB physical-footprint
watchdog on the 1,200.064-second `aorju` recording and was stopped. The recorded
peak was 28.91 GiB, including about 25.65 GiB of reusable MLX cache. Its active
MLX memory peaked at about 3.46 GiB.

The positive control set `MLX.Memory.cacheLimit = 2 * 1024 * 1024 * 1024` before
loading the original model, matching the workaround in the issue. Both this
control and the fixed runtime completed the following inference calls:

| Input | Original with manual 2 GiB cap: peak footprint | Fixed runtime: peak footprint | Structured output |
|---|---:|---:|---|
| `aorju`, 20.00 min | 6.38 GiB | 6.58 GiB | 283 segments, 8,933 output tokens; exact match |
| Five concatenated VoxConverse files, 39.11 min | 9.01 GiB | 9.01 GiB | 624 segments, 16,384 output tokens; exact match |

The longer input concatenated `aepyx`, `aggyz`, `aiqwk`, `aorju`, and `auzru`
in that order, totaling 2,346.56 seconds. Both runtimes emitted EOS with the
last parsed segment ending at 1,437.03 seconds. This is a pre-existing early
termination on the stitched workload, not evidence that its complete audio
was transcribed. No new long-form accuracy or completeness claim is made.

The 39-minute inference took 452.38 seconds in the manually capped original
and 465.97 seconds with the fix. The first fixed 20-minute timing overlapped
an external Swift job and is excluded from performance conclusions.

### Reproduce the inference controls

Use the INT8 bundle with `fromDirectory` → `warmUp` → `transcribeDetailed`.
Load 16 kHz mono Float32 audio and keep these options identical:

```swift
let options = MossMLXDecodingOptions(
    maxTokens: 32_768,
    encoderBatchSize: 4,
    prefillChunkSize: 512,
    kvCachePrecision: .float16
)
```

Measure physical footprint with `TASK_VM_INFO`, including its recorded
high-water mark, rather than RSS alone. Sample active and cached MLX memory
separately. Compare raw output and every structured segment in separate fresh
processes; leave the original cache limit unset for the watchdog control, and
set it to 2 GiB before loading for the positive control. The 2 GiB setting
bounds reusable buffers and does not cap live KV tensors or total footprint.

The current real-weight checks also reproduce missing speaker labels with
INT8 KV cache for both weight variants, identically before and after this fix.
The earlier short-fixture INT8 KV result above does not establish current
structured-output fidelity. Use FP16 KV for speaker-attributed output.
