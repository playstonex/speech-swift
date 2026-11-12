# Fish Audio Drama 3 export, streaming and memory validation — 2026-10-17

The pinned 4-bit export produces 44.1 kHz speech through buffered and streaming
Swift APIs and the CLI. All 16 release benchmark clips passed independent ASR
with zero normalized word errors. Warm streaming first PCM was **275 ms for
text-only speech** and **730 ms for reference cloning**, including reference
encoding on every cloning call.

## Environment and export

- Apple M5 Pro, 48 GiB unified memory, macOS 26.6.2; Swift 6.4 release build
  with the compiled MLX Metal library and default inference precision.
- Checkpoint: `mlx-community/fish-audio-drama-3-4bit` at
  `cc639e27aba7edb73429d3e3806f471dcd79d316`.
- Export: 22.15 GiB; 1,841 model tensors and 538 Fish DAC codec tensors.
  Names, shapes, dtypes, mixed quantization and tokenizer consistency passed
  validation. All eight large-file hashes match the pinned source. Every
  exported file's manifest checksum was rechecked before benchmarking.
- Text: “The quick brown fox jumps over the lazy dog.” Seed 42, temperature 1,
  top-k 30, top-p 0.9, maximum 512 audio frames.
- Cloning uses the upstream model-card
  [reference recording](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav)
  and exact transcript: “We asked over twenty different people, and they all
  said it was his.” The raw report records the reference WAV hash.

## Latency and throughput

Each of the four modes runs in a fresh process: load once, measure the first
synthesis, then three warm calls. Full synthesis measures completion of the
entire waveform. It excludes model loading, reference file I/O and WAV writes.
Cloning includes reference encoding on every call. Local weights were cached;
these are not disk-cold load measurements.

| Mode | Load, s | First full synthesis, s | Warm full median, s | Warm range, s | Audio, s | Warm median RTF |
|---|---:|---:|---:|---:|---:|---:|
| Buffered text-only | 2.640 | 3.431 | 2.306 | 2.296–2.323 | 3.483 | 0.662 |
| Buffered reference cloning | 3.482 | 4.163 | 2.681 | 2.617–2.683 | 3.204 | 0.837 |
| Streaming text-only | 3.502 | 3.557 | 2.529 | 2.518–2.536 | 3.483 | 0.726 |
| Streaming reference cloning | 3.529 | 4.386 | 2.860 | 2.825–2.913 | 3.204 | 0.893 |

RTF is synthesis wall time divided by audio duration; lower is faster.
Streaming preserves the generated sample count. Each mode's four seeded
repetitions produced identical WAV bytes within that mode. The timing sample
covers three repetitions of one short English sentence on this machine.

## Streaming first PCM

| Mode | First call, ms | Warm median, ms | Warm range, ms |
|---|---:|---:|---:|
| Streaming text-only | 1318.1 | 275.3 | 271.5–283.2 |
| Streaming reference cloning | 2309.5 | 730.2 | 700.1–730.8 |

First PCM is the first nonempty callback at the caller. Three codec frames
produce 6,144 samples, or 139.3 ms of audio, per default chunk. The text-only
utterance has 25 nonempty chunks; cloning has 23. Both end with an empty final
marker because EOS follows an exact chunk multiple.

These timings exclude model loading and include reference encoding for
cloning. Reusing an `encodeReferencePrompt` result removes that preparation
from later calls, but cached-reference TTFA was not measured. Audible speech
onset and device playback onset were not measured. CLI playback uses a 160 ms
prebuffer. The iterator advances on demand and the callback is awaited before
another chunk is generated, retaining admission through final delivery.

## Process memory

| Mode | Peak RSS, GiB | Peak physical footprint, GiB | Physical after loading, GiB | Physical after all calls, GiB |
|---|---:|---:|---:|---:|
| Buffered text-only | 14.457 | 24.220 | 22.007 | 22.952 |
| Buffered reference cloning | 15.064 | 24.789 | 22.013 | 23.067 |
| Streaming text-only | 15.571 | 22.321 | 22.014 | 21.918 |
| Streaming reference cloning | 14.676 | 23.885 | 22.013 | 22.503 |

The peaks are kernel-recorded high-water marks across loading and all four calls,
including retained MLX allocations and caches. RSS uses Mach resident accounting
and `getrusage`; physical footprint uses `TASK_VM_INFO`. Physical footprint is
the more useful unified-memory sizing metric for these MLX weights. ASR runs
in a separate process after all TTS processes exit. Maximum-context and
maximum-length output memory were not measured.

## Memory optimization comparison

The default `balanced` policy evaluates codec intermediates between layers and
trims idle MLX buffers above 512 MiB at stage boundaries. It also defers
794.4 MiB of reference encoder weights until the first reference encoding.
Those weights stay loaded after cloning begins. Global allocator limits, model
weights, quantization and inference precision are unchanged. `unrestricted`
opts out of both changes. The threshold is not a total-memory limit.

The comparison below reruns the preserved pre-optimization release binary and
then the optimized binary on the same checkpoint and reference. Each mode uses
one first call and three warm calls in its own process. The optimized measurements
also passed a preceding independent run; its peak physical footprints were
24.241 / 24.788 GiB buffered and 22.322 / 23.906 GiB streaming.

| Mode | Previous peak physical, GiB | Balanced peak physical, GiB | Reduction | Peak RSS before → after, GiB | Warm full synthesis, s |
|---|---:|---:|---:|---:|---:|
| Buffered text-only | 28.490 | 24.220 | 15.0% | 12.689 → 14.457 | 2.219 → 2.306 |
| Buffered reference cloning | 31.127 | 24.789 | 20.4% | 16.396 → 15.064 | 2.436 → 2.681 |
| Streaming text-only | 23.504 | 22.321 | 5.0% | 16.657 → 15.571 | 2.383 → 2.529 |
| Streaming reference cloning | 26.545 | 23.885 | 10.0% | 15.850 → 14.676 | 2.650 → 2.860 |

Warm streaming first PCM changed from 268.7 to 275.3 ms for text-only and from
670.4 to 730.2 ms for cloning. Cache trimming reduces memory at some cost to warm latency; these
small repeated samples do not establish a latency distribution across workloads.
Peak physical footprints reproduced closely across optimized runs. RSS varied
more widely between fresh processes, so the table reports both measurements.

All 16 corresponding WAVs are byte-identical between the previous runtime and
both optimized runs. Every run passed 16 independent ASR checks. Deferred and
eager reference encoding also produced exactly the same codebook values.
The lazy codec remains decoder-only after synthesis and materializes its encoder
on first use; repeated encoding preserves the same codes.

## Correctness and regression coverage

- 51 targeted Swift checks passed: configuration, prompt/sampling, CLI routing,
  S2 Pro regressions, upstream numerical fixtures, released tokenizer, waveform
  and ASR checks, streaming decoder math, codec parity, deferred loading, cache
  policy and stream lifecycle.
- 13 exporter tests and six profiling tests passed.
- Independent `aufklarer/Qwen3-ASR-0.6B-MLX-4bit` recovered all nine requested
  words from each benchmark WAV: zero errors across 144 reference words.
  Streamed SDK text-only, cloned SDK output, buffered CLI and streamed CLI
  also passed isolated real-model ASR checks.
- Full versus incremental Fish DAC decoding was checked on 137 fixed frames
  with chunk sizes 1, 3, 17 and 128. This crosses the 128-frame attention window
  and covers convolution overlap and final trimming. With
  `MLX_ENABLE_TF32=0`, maximum sample error was `8.52e-6`; RMS error stayed below
  `3.18e-7`. The exact output count was 280,576 samples in every case.
- Default M5 inference uses the library precision policy. Real speech had
  maximum sample error `0.003845` and relative RMS difference `0.4263%` for
  text-only, and `0.002360` / `0.3583%` for cloning versus buffered decoding.
  The sample counts matched and independent ASR recovered the complete sentence.
- Active and queued cancellation, callback errors, explicit frame-limit errors,
  final-callback backpressure, protocol dispatch and abandoned-stream cleanup
  passed. A later request succeeds after abandoning partial output.
- Builds, tests, shader compilation and model processes ran sequentially. The
  profiler rejects overlapping package build/test/benchmark processes and
  discards a measurement if a conflicting job starts.

ASR establishes sentence content; speaker similarity was not measured. 8-bit,
BF16, multilingual quality and long-form generation remain unvalidated.

## Reproduction and artifacts

From a separate worktree, with the ASR checkpoint already cached:

```bash
python3 scripts/profiling/profile_fish_drama3.py --build --delivery both --memory-mode balanced \
  --bundle /path/to/Fish-Audio-Drama-3-MLX-4bit \
  --reference /path/to/reference.wav \
  --reference-text "We asked over twenty different people, and they all said it was his." \
  --output-dir /path/to/drama3-benchmark --warm-repeats 3 --chunk-frames 3 --jobs 2

python3 -m unittest discover -s scripts/tests -p test_profile_fish_drama3.py -v
```

The command builds an isolated release profiler, restores the package manifest,
and runs TTS and ASR sequentially. The output directory retains WAVs, raw
profiles, transcripts and `benchmark.json`. Raw run data stays in the profiling
output directory; this file is the summary report.

To measure a preserved baseline binary, replace `--build` with
`--binary /path/to/preserved/fish-drama3-profile` and
`--runtime-sources-sha256 <recorded-source-hash>`, and select
`--memory-mode unrestricted` with a separate output directory.

Runtime source SHA-256: `c0f986760c2c9945b8de9ead6a19d671af58b56ccc0d8d0b035bad2c7d73fcab`.
Release profiler SHA-256: `d4bfd94d64768a95131c8f7ec2d0f586d13ce16c963fc8d141b2534581e4a32c`.

See the [inference guide](../inference/fish-audio-drama3.md) for CLI and Swift
streaming usage.

Preserved baseline runtime source SHA-256: `d1fe5c530450f11ab065a04bff6e87a1b111dd07e21274245560791b58eb58d5`.
Baseline release profiler SHA-256: `7d86aea7bed8596acd64e60c686d7f46927907e850683aa638d64ad5db51ed9f`.

Measured runtime snapshot: `bf3bccfd9a219df94b29b2290b951b4bdf194b80`.
Subsequent integration retains the Drama 3 implementation while adding the
current package targets and CLI routes from main.
