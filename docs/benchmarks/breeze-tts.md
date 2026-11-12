# Breeze TTS 2 benchmark

Measure the release Swift runtime with compiled MLX Metal shaders. Run builds,
tests, synthesis, and independent ASR sequentially, with one model process at a
time. Raw traces, timing JSON, and WAVs stay outside the committed report.

## Matched runtime comparison — 2026-10-30

Apple M5 Pro, 48 GiB unified memory, macOS 26.6.2. The baseline is the Breeze
runtime from `d53bcb333882ebbd7444ae059b270127565594a0`, built with the same
profiler as the optimized runtime. Both use the current shared 2 GiB reusable
MLX cache ceiling, cached local bundles, seed 42, CFG 4, and default 3 / 25
frame chunks. Model loading and WAV writing are outside TTFA and RTF. Memory
values are process peaks including loading and all requests; physical footprint
includes GPU allocations that RSS can undercount.

Each row runs in a fresh process: one initial request, then three warm requests.
Warm timings are medians. Opening and closing baseline controls bracket each
optimized short-sentence case. Other builds and model jobs were excluded from
the measurement window.

English voice design: “The quick brown fox jumps over the lazy dog.”
Instruction: “A clear, calm English narrator.”

| Runtime | MLX | First PCM p50 | RTF | Peak RSS | Peak physical footprint |
|---|---|---:|---:|---:|---:|
| Baseline, opening | 8bit | 332 ms | 0.706 | 3.92 GiB | 7.07 GiB |
| Optimized | 8bit | 178 ms | 0.535 | 3.92 GiB | 7.23 GiB |
| Baseline, closing | 8bit | 328 ms | 0.706 | 3.92 GiB | 7.08 GiB |
| Baseline, opening | bf16 | 459 ms | 1.258 | 6.05 GiB | 9.18 GiB |
| Optimized | bf16 | 272 ms | 0.868 | 6.06 GiB | 9.22 GiB |
| Baseline, closing | bf16 | 474 ms | 1.303 | 6.05 GiB | 9.20 GiB |

RTF is generation wall time divided by generated audio duration; values below
one indicate generation faster than playback on this workload. All 8-bit short
clips are 3.20 s and byte-identical across baseline and optimized builds. BF16
changes floating-point accumulation order: its seeded clip changes from 3.68 s
to 3.20 s, so this comparison does not represent identical output frames.
Weights, sampling parameters, and default chunk sizes are unchanged.

### Longer speech and first audio

The paragraph is: “Before we start the meeting, please check that your microphone
is working. We will review the project timeline, answer questions, and agree on
the next steps.” Paragraph values are medians of two warm requests after an
initial request, with the same instruction and default chunks.

| BF16 runtime | Audio duration | First PCM | RTF | Peak RSS | Peak physical footprint |
|---|---:|---:|---:|---:|---:|
| Baseline | 10.80 s | 491 ms | 1.280 | 6.07 GiB | 9.48 GiB |
| Optimized | 9.20 s | 272 ms | 0.853 | 6.06 GiB | 9.94 GiB |

With `--first-chunk-frames 1`, optimized BF16 short-sentence first PCM is
138 ms and RTF is 0.862, with a 9.51 GiB peak physical footprint. Keeping
25-frame follow-ups still delays the estimated CLI prebuffer until 1.78 s.
First PCM measures delivery of a decoded chunk, excluding model loading,
playback buffering, render callbacks, and output-device latency.

For the 8-bit paragraph, one initial frame, two-frame follow-ups, and a 1 GiB
cache budget give 103 ms first PCM, RTF 0.761, 3.93 GiB peak RSS, and 6.10 GiB
peak physical footprint. The 9.04 s output reaches the CLI prebuffer at an
estimated 190 ms, with no additional buffering required by the recorded chunk
arrivals. These are medians of two warm requests, not a speaker-latency measurement.

```bash
speech speak "The quick brown fox jumps over the lazy dog." --engine breeze \
  --instruct "A clear, calm English narrator." --first-chunk-frames 1 \
  --chunk-frames 2 --breeze-cache-limit-mib 1024 --stream -o fast-start.wav
```

The continuous-playback lower bound is
`max(chunkArrival - precedingAudioDuration)` over nonempty chunks. The CLI
prebuffer is `max(0.16, chunkFrames / 12.5)` seconds. The profiler records both
estimates and any extra initial delay needed to avoid gaps in the observed
schedule. A short clip can start when generation finishes if it never reaches
the prebuffer threshold. These estimates assume immediate PCM scheduling.

A 10 ms RMS activity check at −50 dBFS finds the short clip's first active window
20 ms into its waveform; the paragraph is active in its first window. Both are
present in the first PCM chunk. This measures waveform activity, not VAD accuracy.

`--breeze-cache-limit-mib` optionally sets a process-wide reusable MLX cache
budget before loading. Omission preserves the application budget; the package's
shared model policy caps reusable cache at `min(2 GiB, physicalMemory / 4)` and
honors tighter limits. Zero disables reusable-buffer caching. The flag can
tighten the package ceiling; it does not limit live model tensors. The profiler
uses `--mlx-cache-limit-mib` for the same experiment.

### Cloning and direction

The reference is the verified baseline BF16 English sentence above, with its
exact transcript. Each case measures one warm request after an initial request,
saying “The weather is beautiful today.” Direction adds “Speak clearly and warmly.”

| Mode | MLX | First PCM | RTF | Audio duration | Peak RSS | Peak physical footprint |
|---|---|---:|---:|---:|---:|---:|
| Reference clone | 8bit | 156 ms | 0.470 | 1.76 s | 4.10 GiB | 6.26 GiB |
| Directed clone | 8bit | 204 ms | 0.549 | 1.84 s | 4.09 GiB | 6.35 GiB |
| Reference clone | bf16 | 245 ms | 0.743 | 2.16 s | 6.22 GiB | 8.64 GiB |
| Directed clone | bf16 | 303 ms | 0.887 | 1.92 s | 6.24 GiB | 8.61 GiB |

## What changed

The compact tokenizer removes 6,241 unused Gemma placeholders from ordinary-text
matching. Literal `<unused…>` inputs load the original tokenizer lazily and
retain every added-token rule. All 187 retained added tokens, representative
placeholders, and mixed Unicode inputs match the exported original tokenizer.

Depth-codebook samples remain integer tensors until the complete frame is read
once by the host. Each depth step submits asynchronous evaluation; backbone
hidden states and caches are evaluated together. Sampling masks, RNG advances,
and guidance settings retain their original behavior.

Unquantized BF16 projections with exactly two input vectors keep the vectors in
the batch dimension, allowing Metal's batched matrix-vector kernel. Other
precisions, shapes, and quantized layers use the original projection path.
Set `SPEECH_SWIFT_BREEZE_GEMV=0` before loading a model to disable this projection
change. Different accumulation rounding can change BF16 seeded waveforms and
duration; determinism within a fixed runtime and chunk schedule is preserved.

## Regression validation

The current-main integrated release build completed 1,933 CI-equivalent unit tests
with 42 explicit skips and no failures. The focused run passed 41 checks,
including sampling, compact-tokenizer fallback, projection shapes and precision,
cache ownership, CLI parsing, waveform activity, and playback estimates.
Three Python build-helper checks passed.

Fifteen isolated real-model checks passed with no skips: official numerical
fixtures, English and Chinese independent-ASR round trips, cloning and direction,
frame-limit errors, overlap rejection and cancellation recovery for both BF16
and 8-bit; streaming CLI output for both variants; and the real exported
tokenizer comparison. Full-model fixture cosines remain above 0.99:

| MLX | Text features | Backbone | LM logits | Depth logits |
|---|---:|---:|---:|---:|
| BF16 | 0.999701 | 0.999957 | 0.999985 | 0.999938 |
| 8bit | 0.999503 | 0.999937 | 0.999977 | 0.999952 |

The final matched series generated 45 WAVs representing 11 distinct waveforms.
All repeated requests within each case are byte-identical. The 12 baseline and
optimized 8-bit default-chunk WAVs share the same SHA-256. Independent Qwen3-ASR
reports zero normalized WER and CER for all 11 distinct waveforms, covering the
short sentence, paragraph, alternate chunks, reference clones, and direction.
This checks speech content, not general acoustic quality, speaker identity,
instruction fidelity, or arbitrary-text real-time performance.

## Earlier profiling — 2026-10-24

Earlier measurements preceded the shared cache ceiling and can have larger
physical footprints. The original published short-sentence baseline was
329 ms / RTF 0.707 / 3.94 GiB RSS / 8.22 GiB footprint for 8-bit, and
462 ms / RTF 1.258 / 6.06 GiB RSS / 10.75 GiB footprint for BF16.
The initial tokenizer and host-read optimization gave 179 ms / 0.545 for 8-bit
and 326 ms / 1.137 for BF16, before the BF16 projection change.

An isolated 8-bit Metal System Trace attributed about 65% of generation time
to depth decoding and sampling, 23% to backbone decoding, 7% to waveform
conversion, and 4% to prompt work. Target GPU command intervals were active for
93% of the traced generation span; this is not shader occupancy. CPU samples
showed scheduler backpressure in asynchronous evaluation. Full-frame compilation,
projection fusion, guidance-prefill batching, wider matrix batches, and backend
command-buffer sweeps did not provide a suitable consistent improvement and
were not retained.

One-frame follow-ups generated the earlier paragraph at RTF 1.040 and needed
279 ms of extra initial buffering; two-frame follow-ups provided a better
startup/throughput balance. Earlier byte-identical baseline and optimized output
for “The meeting starts at three o'clock this afternoon. Please bring the report.”
omitted the initial article in independent ASR (WER 1/13, CER 4.84%). This known
smoke-test discrepancy was present before the optimization.

A locally converted FP16 experiment measured about 0.897 RTF on a short clip,
but both original and candidate paths reached the paragraph's 250-frame cap.
It remains unpublished; the retained projection optimization applies only to BF16.

[Earlier timing and transcripts](breeze-tts-2026-10-24.json) and the
[2026-10-23 archive](breeze-tts-2026-10-23.json) remain historical records. The
latter includes a discontinued 4-bit export that is no longer published.

## Reproduce

```bash
python3 scripts/profiling/build_breeze_profile.py
.build/release/breeze-tts-bench --bundle out/Breeze-TTS-2-MLX-8bit \
  --output /tmp/breeze-8bit.json --warm-repeats 3
.build/release/breeze-tts-bench --verify /tmp/breeze-8bit.json \
  --output /tmp/breeze-8bit-asr.json
```

The synthesis JSON records model-load time, a cold request and warm repeats,
time to the first nonempty decoded audio chunk, total generation latency, audio
duration, RTF (wall seconds / audio seconds), RSS, kernel peak RSS, physical
footprint, kernel peak physical footprint, and MLX active/cache/peak bytes.
Model-load time excludes downloads. First-audio latency includes prompt encoding,
prefill, frame generation, and waveform decoding; it excludes model loading.
WAV writing follows the timing interval. The first request is cold for execution
and codec kernels; warm requests reuse the loaded model.

It also records each chunk's arrival time and sample offset, first waveform
activity at −50 dBFS RMS in 10 ms windows, and the minimum continuous-playback
start estimate described above. Raw profiler output stays in a scratch directory;
this report summarizes the results.

`--reference` and `--reference-text` measure reference cloning; add `--instruction`
for direction. Use `--instruction ""` for cloning without guidance. Reference-file
read/resampling occurs before request timing; reference codec encoding is inside
each measured request. `--first-chunk-frames`, `--chunk-frames`, `--cfg-scale`,
`--temperature`, `--seed`, and `--max-frames` specify the measured workload.

Round-trip verification uses `aufklarer/Qwen3-ASR-0.6B-MLX-4bit`, normalized WER
for English and CER for Chinese. It verifies speech content, not voice identity,
instruction adherence, acoustic quality, or absence of boundary artifacts. A
short-sentence smoke benchmark is not a general multilingual quality evaluation.
