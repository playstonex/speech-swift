# MOSS-VoiceGenerator validation

This module targets the original BF16 language checkpoint and FP32 synthesis
decoder. No INT4 result is represented as a supported configuration.

## Measured runtime

Measured on Apple M5 Pro with 48 GiB unified memory, 2026-10-29, using a release
build and compiled MLX Metal shaders. One English sentence and one Mandarin
sentence were each generated four times in a process, with seed 42 and three
audio frames per codec chunk. Each language produced identical WAV bytes across
its four requests. Warm values are medians of the final three requests.

| Input | Audio duration | First PCM, initial request | First PCM, warm | Generation, warm | Real-time factor, warm | Peak process physical footprint |
|---|---:|---:|---:|---:|---:|---:|
| English, nine words | 3.44 s | 561 ms | 470 ms | 2.098 s | 0.610 | 8.448 GB |
| Mandarin, two sentences | 4.88 s | 1,308 ms | 457 ms | 2.793 s | 0.572 | 8.619 GB |

Lower real-time factor is faster; both examples finished faster than their audio
duration. These short measurements cover two scripts and do not establish an
instruction-following score or performance on long utterances. First PCM and
generation time exclude model loading. Weight files were hashed immediately
before loading, so their filesystem cache was warm; loading took 0.635 s for the
English process and 1.383 s for Mandarin. Integrity verification is a separate
measurement.

The complete local bundle contains 7,793,146,620 bytes. MLX active model memory was
7,777,465,356 bytes before unload; explicit unload reduced live MLX memory to 12
bytes in both processes. Reusable spare buffers can remain within the restored
caller/shared cache budget. Loading and model lifetime request a coordinated
512 MiB spare-buffer budget, without process-wide cache flushes. This allocator
setting is not a total process-memory limit; cached bytes can transiently exceed
it by buffer granularity. The process physical-
footprint column above is the measured memory value, rather than the smaller
allocator counter. All eight final WAVs are byte-identical to the earlier
profiles generated before this cache-ownership change.

File-hashing buffers are released after each chunk before model memory is
measured.

## Correctness and lifecycle checks

- Five CPU-only cache-ownership checks passed, covering overlapping loader/model
  leases, other model owners, tighter caller limits, restoration and failed loads.
- Six numerical reference checks passed, including Float32/BF16 language output
  and full versus cached continuation.
- Seven real-model checks passed, covering synthesis, tokenizer IDs, cancellation
  before and during generation, recovery, concurrent-request rejection and
  explicit/concurrent unloading.
- Input-boundary checks cover UTF-8 byte limits, normalization and rendered
  token counts. The real synthesis test rejects oversized raw and tokenized
  prompts without emitting audio, then completes a normal request on the same
  instance. Each field is limited to 32,768 bytes before preprocessing, and
  the complete rendered prompt is limited to 4,096 tokenizer positions before
  allocating its dense attention mask.
- The native CLI generated a separate nine-word English script. After that
  synthesis child exited, an independent Parakeet child transcribed the saved WAV
  and matched all nine normalized words exactly.
- All four Mandarin timing outputs matched the expected toneless pinyin sequence.

The English timing clip has a recognizer disagreement: Parakeet inserts "and",
while Qwen3-ASR returns the exact nine-word script. Both results are retained.
The separate CLI content check passed, but these results do not establish zero
word error on every recording.

The existing-model regression screen retained identical WAV bytes for all sixteen
seeded before/after pairs, and 59 existing runtime source files are unchanged.
All 24 planned timing runs have completed. Some latency results exceeded the
check's thresholds, so the existing-model performance comparison remains
inconclusive; absence of latency regression has not been established. Identical
audio does not establish unchanged performance.

Model quality remains experimental. These checks verify native execution and
specific content/lifecycle cases. This model has not yet been scored on
InstructTTSEval.

## Follow-up regression checks, 29 October 2026

A same-executable control was stable in a quieter machine window. Qwen English
generation differed by less than 2% between the archived old/new executables in
both run orders. A separate 24-profile matched screen retained identical audio
in all twelve pairs, but several timing/load comparisons and one Breeze physical-
memory comparison exceeded the original thresholds. That memory increase was
8.6%. These results do not certify unchanged latency or memory across all cases;
the original measurements and threshold failures remain part of the evidence.

A sequential process kept Qwen3-TTS loaded while MOSS was loaded, used, actively
cancelled and unloaded. All twelve fixed-seed Qwen WAVs matched exactly. Its cache
setting returned from 512 MiB to 2 GiB, and the original caller setting was restored
after both models unloaded. Peak process physical memory was 14.147 GB, falling
to 5.098 GB immediately after MOSS unload with Qwen still loaded. Qwen's warm
real-time factor increased 8.0% while MOSS was resident and 10.5% afterward;
first-audio latency increased 13.3% and 12.0%. This checks functional recovery and
cache ownership for one English script. Concurrent generation, other scripts
and other model pairs remain outside this check.

The opt-in `E2EMossVoiceGeneratorCoexistenceTests` regression case checks complete
Qwen audio and cache restoration with explicit local bundles. It makes no timing
assertion. The documented public model-ID CLI commands for both new engines also
produced complete WAVs; independent Parakeet transcription matched every word of
their fixed five-word and nine-word English scripts.

Runtime measurements must use a release build with its compiled MLX metallib and
run one model at a time. Record model-load time separately from request time,
first token, first audio, output duration, real-time factor and process physical
footprint. Read the physical footprint while generation runs; allocator counters
and resident memory do not replace that measurement.

The numerical fixtures use tiny deterministic CPU reference tensors. They check
Qwen3 attention and embedding math, speech transformer windows and cached chunks,
rotary positions, patch ordering and signed weight normalization. The codec
fixture specifically includes context wraparound, where a faulty cache could
pass a short-utterance check.

Model checks use `MOSS_VOICEGEN_BUNDLE` to select local weights and
`MOSS_VOICEGEN_TEST_OUTPUT` to save the generated WAV and timing JSON. Run an
isolated synthesis test, release its process, then transcribe its saved recording
and compare the complete script. Also inspect the first and last words, silence,
non-finite samples and waveform clipping. Cancellation and concurrent-request
checks must leave the model usable.

An instruction-following score requires the fixed pilot recordings and the same
saved cloud rubric used for other models. The ability to synthesize a voice is
separate from agreement with an audio judge. No style score is claimed here.
