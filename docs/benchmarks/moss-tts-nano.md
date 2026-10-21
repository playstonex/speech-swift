# MOSS-TTS-Nano CoreML measurements

Apple M5 Pro, 48 GB RAM, macOS 26.6.2. Each backend ran alone in a fresh Python
process using CoreMLTools 9.0, NumPy 1.26.4 and SentencePiece 0.2.1. Neither PyTorch
nor ASR was loaded. The first request followed model loading; warm values are
medians of five requests. All requests used fresh state. OS/CoreML disk caches
were not cleared. These are Python runtime results, not Swift or iPhone results.

| Text-only: `Hello from Core ML.`, seed 0 | CPU | CPU+GPU |
|---|---:|---:|
| Peak process RSS | 658 MiB | 636 MiB |
| Model loading | 0.98 s | 1.18 s |
| First audio token, first / warm | 62 / 44 ms | 87 / 21 ms |
| First 80 ms audio frame, first / warm | 82 / 57 ms | 178 / 47 ms |
| Warm time / audio duration (lower is better) | 0.289 | 0.422 |

Cloning uses the saved 2.72-second stereo `hello-gpu.wav` reference, text
`The weather is nice today.`, seed 1. Uncached measurements include file reading,
resampling/padding, codec-encoder loading and reference encoding. Cached
measurements reuse encoded reference codes. Loading the three synthesis graphs
is excluded from request latency. RSS covers the complete worker while all four
graphs are exercised; codec-encoder loading is included in reference preparation.

| Voice cloning | CPU | CPU+GPU |
|---|---:|---:|
| Peak process RSS | 779 MiB | 731 MiB |
| First audio token, first / warm uncached | 542 / 203 ms | 649 / 225 ms |
| First audio token, warm cached | 52 ms | 24 ms |
| First audio frame, first / warm uncached / warm cached | 563 / 216 / 65 ms | 741 / 245 / 48 ms |
| Warm reference preparation | 150 ms | 206 ms |
| Warm time / audio duration, uncached / cached | 0.324 / 0.270 | 0.471 / 0.400 |

The first token is the first sampled codebook ID; audio playback must wait for
all 16 codebooks and codec decoding. Small backend numerical differences can
change sampled output length (37 CPU frames versus 30 GPU frames in the cloning
run). Compare normalized time/audio duration for throughput. Warm GPU first-token
latency is lower with cached references; CPU is faster for complete utterances.
The likely contributors to first-request GPU cost are runtime specialization and
many small sequential predictions, but operation placement was not profiled.

The bundle includes `performance.json`, `cloning_performance.json`, the runnable
`benchmark_coreml.py`, and numerical/ASR smoke reports. Source-graph parity and
short ASR round trips validate conversion and conditioning. They do not establish
general speech quality or speaker identity similarity. The saved GPU greeting
transcribes its acronym differently; the CPU greeting and original cloned sample
match their text after normalization. See the full model card for exact samples.

## Native Swift release CLI

One fresh CLI process per backend, CPU then CPU+GPU, using the same reference,
text and seed as the Python cloning run. Both runs used the public model bundle.
OS/CoreML disk caches were not cleared. First-token/audio latency includes
reference reading, padding, codec-encoder loading and encoding, while loading the
three synthesis models is excluded. RSS is the whole `speech` process maximum
from `/usr/bin/time -l`; ASR weights were loaded in separate verification
processes. These are single-request smoke measurements, not warm medians.

| Metric | CPU | CPU+GPU |
|---|---:|---:|
| Peak process RSS | 670.7 MiB | 602.5 MiB |
| First audio token, including reference | 537 ms | 570 ms |
| First decoded audio frame, including reference | 560 ms | 645 ms |
| Request / generated audio duration | 1.38 / 2.96 s | 1.65 / 2.40 s |
| Time/audio duration | 0.466 | 0.688 |
| Independent ASR | The weather is nice today. | The weather is nice today. |

The measured WAVs were byte-identical to the independently ASR-checked release
outputs. Matched-host E2E tests also reproduce Python codec tokens on both
backends. Swift warm/cached-reference timings and iPhone performance remain
unmeasured. Full metadata: [native result JSON](moss-tts-nano-swift.json).

```bash
make build
/usr/bin/time -l .build/release/speech speak "The weather is nice today." \
  --engine moss-nano --voice-sample reference.wav --moss-nano-seed 1 -o cpu.wav
/usr/bin/time -l .build/release/speech speak "The weather is nice today." \
  --engine moss-nano --voice-sample reference.wav --moss-nano-seed 1 \
  --moss-nano-compute-units gpu -o gpu.wav
```
