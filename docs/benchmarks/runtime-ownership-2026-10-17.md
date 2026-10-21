# Runtime ownership regression and performance verification

The initial ownership changes were compared with `53510fb99661942b952184c1a744dcedc0a8fdb3`
on an Apple M5 Pro with 48 GiB of memory, macOS 26.6.2 and Swift 6.4.
The initial candidate is `88fce9a`. The current-main integration is verified
separately below. No weights,
quantization settings or decoding defaults were changed.

## Issues found during verification

- Strict Swift 6 code generation rejected the inference gate's private waiter
  in the cancellation handler. Its mutable fields are protected by the gate's
  lock; an explicit `@unchecked Sendable` declaration documents that invariant.
  The production sources now compile and run under Thread Sanitizer.
- Repeated managed CoreML calls from one background task retained autoreleased
  scratch buffers. A 16-request probe grew from 879 MiB after warm-up to
  2,495 MiB. The adapter now wraps each synchronous inference in an
  `autoreleasepool`, matching the existing native CoreML benchmark's policy.
  A real-weight regression test checks memory growth over 12 repeated requests
  after three warm-up requests, with a 256 MiB allowance for allocator noise.

## Measurement method

Each model run uses its own process, cached weights in offline mode and a
release build with the compiled MLX metallib. Heavy builds, tests and model
measurements run sequentially. Both variants execute on a background task at
`userInitiated` priority. Native CoreML calls use the same per-utterance
autorelease-pool policy as `EngineQwen3CoreML` in `asr-bench`.

The dataset contains the first 20 English and first 20 French clips from the
cached `kyutai-fleurs-float-80` manifests: 40 unique utterances and 427.280
seconds of audio. Audio is decoded to 16 kHz before timing. Both paths use
automatic language detection and their existing default decoding parameters.

After two warm-up transcriptions, each process transcribes the dataset twice.
Timing statistics use the final pass. The order is baseline A, managed A,
managed B, baseline B, then a native control built from the changed sources.
Each baseline/managed group therefore contains 80 warm timing samples and
160 exact transcript comparisons across both passes. WER uses the repository's
`Normalizer` and `WER` implementations. Weighted RTF is total inference wall
time divided by total audio duration; lower is faster.

The first exploratory batch showed substantial timing drift after compilation
and exposed the native caller's autorelease-pool behavior. Those files are
retained under `conditioning/` and excluded from the final comparison.

## Results

No transcript regression or material single-request slowdown was observed in
the measured paths. Both backends produced identical text in all 320 paired
baseline/managed comparisons. The native control built from the changed
sources retained the same WER as the original build.

| Model | Baseline weighted RTF | Managed weighted RTF | Change | Baseline / managed WER |
|---|---:|---:|---:|---:|
| Qwen3-ASR 0.6B MLX INT4 | 0.01378 | 0.01334 | -3.2% | 11.936% / 11.936% |
| Qwen3-ASR full CoreML | 0.13174 | 0.13134 | -0.3% | 7.237% / 7.237% |

The MLX native control's weighted RTF was 0.01380, and the CoreML native
control's was 0.13198. Small differences do not establish an inference
optimization or performance guarantee on other hardware. The two WER values
describe different existing backends on this subset; this is not a quality
ranking of the backends.

Peak physical footprint was 7,834 MiB before and 7,856 MiB after for MLX
(+0.28%). CoreML peaks were 1,354 MiB and 1,257 MiB respectively. Current
footprint at the end of the two 80-request managed runs was 705 MiB and
1,022 MiB. Those readings include allocator and framework-cache variation;
the kernel-recorded peak remained 1,257 MiB across the managed runs.

The same 16-request background probe that exposed the memory issue was
repeated after the fix. Its footprint growth fell from 1,616 MiB to 102 MiB,
with identical transcripts. The fixed probe's footprint rose and fell within
the workload rather than increasing with every request; model/graph caches
still account for part of the process footprint.

| Live API check | Baseline median | Changed median | Samples per version |
|---|---:|---:|---:|
| HTTP MLX transcription | 95.48 ms | 95.36 ms | 12 |
| WebSocket MLX transcription | 97.14 ms | 97.22 ms | 12 |
| HTTP Kokoro synthesis | 86.43 ms | 87.30 ms | 4 |
| WebSocket Kokoro synthesis | 85.09 ms | 85.65 ms | 4 |

Both Kokoro APIs returned byte-identical audio before and after the changes,
with 73,800 samples per response. The TTS timing sample is small; its observed
median increase was below 1 ms. CoreML HTTP and WebSocket requests also
transcribed correctly through the explicit CoreML descriptor. The first
CoreML HTTP request included 13.8 seconds of cached model loading; subsequent
HTTP/WS calls were approximately 0.59–0.60 seconds on the same fixture.

Four simultaneous HTTP requests completed successfully at approximately 97,
191, 284 and 378 ms, with 378 ms total wall time. This is the intended serial
admission behavior. The baseline's concurrent mutable-model execution was
not used as a throughput target. Health requests during inference remained
responsive: 0.53 ms median before and 0.57 ms after.

Uncontended gate admission/release measured 0.21 microseconds median; a warm
generic runtime operation measured 0.33 microseconds. In the concurrent
playback stress, all 28,800,000 samples were consumed and the longest buffer
read was 0.120 ms. A 512-frame callback at 24 kHz has a 21.33 ms period.
The 60-second overflow probe preserved all 1,440,000 samples. Read medians
and p99 values were near the clock's resolution, so they do not support a
precise relative CPU-speed comparison.

The aggregate numbers and input/binary hashes are retained with the local
artifacts as `benchmark-results/runtime-ownership-20261017/published-summary.json`.

## Regression coverage

- CI-style unit selection: 1,427 tests executed, 42 skipped, zero failures.
- Targeted ownership, HTTP/WebSocket, CLI, decoding and playback rerun after
  the Swift 6 fix: 163 tests, zero failures.
- Real MLX inference: native/managed transcript parity, unload/reload,
  cancellation and recovery passed again.
- Real CoreML inference: backend selection and repeated background-request
  memory regression tests passed.
- Live HTTP/WebSocket ASR, CoreML backend discovery/routing, four concurrent
  HTTP requests, and Kokoro HTTP/WebSocket output checks passed. MLX and full
  CoreML CLI smoke tests transcribed the fixture correctly.
- Swift 6 Thread Sanitizer: 16,000 exclusive operations, 64 concurrent cache
  requests, 3,200 cancellation/grant races and 1,966,080 exactly ordered
  playback samples passed without sanitizer reports.
- The model-independent `SpeechRuntime` sources compiled for iOS 18 with
  strict Swift 6 code generation. This is not a full iOS package/app build.

## Current-main integration

The change was then integrated with main at
`839127e237314d01fab392727261b9d9d3ba5951`; the production candidate is
`c1ba66451ab264a01030e87755a023d758af56c1`. The adapter delegates to the
existing cancellation-aware decoder and preserves main's MLX cache cleanup,
checkpoint initialization and realtime server VAD handling.

Validation after integration passed:

- Hosted CI build and unit tests on `c1ba664`, plus all CodeQL checks.
- Local CI-style selection: 1,665 tests executed, 42 skipped, zero failures.
- Focused ownership, HTTP/WebSocket, VAD, CLI and playback selection:
  182 tests, zero skips or failures. A new regression checks VAD-triggered
  transcription sharing the HTTP inference lane while retaining automatic
  language detection. The new server tests use the shared readiness/shutdown
  fixture.
- Isolated real MLX and CoreML runtime suites, including unloaded-model
  errors, reloads, cancellation recovery and bounded repeated-call memory.
- Five native decoder memory-cleanup tests, the real HTTP and WebSocket
  Qwen3-ASR tests, and the standalone Thread Sanitizer stress check.
- Debug CLI smoke checks for MLX and full CoreML returned the expected fixture
  transcript.

The release comparison was repeated against the integrated main revision
using the same 40 English/French clips. This follow-up uses one measured pass
per process after two warm-ups, in the same ABBA order followed by the native
control. Each baseline/managed group has 80 samples per backend, with 160 exact
transcript comparisons overall. Both builds use the same Metal 3.2/macOS 15
shader library; the baseline build script verified its compiler/source hash.

| Backend | Baseline weighted RTF | Managed weighted RTF | Change | Baseline / managed median | WER, both |
|---|---:|---:|---:|---:|---:|
| MLX INT4 | 0.01561 | 0.01580 | +1.2% | 145.32 / 143.78 ms | 11.936% |
| Full CoreML | 0.13101 | 0.13212 | +0.8% | 1,301.00 / 1,307.72 ms | 7.237% |

All 160 transcripts matched exactly. MLX p95 was 279.64 ms before and
312.35 ms after; the native control built from the candidate measured
311.87 ms. The native controls' weighted RTFs were 0.0161 for MLX and 0.1399
for CoreML, showing that native timings also varied during the session.

MLX peak physical footprint was approximately 2,536 MiB in both groups.
CoreML's managed peak was 1,327 MiB versus 1,240 MiB in the baseline, an
86 MiB increase in this run. Peak footprint includes allocator and framework
caches; the repeated-call memory regression test still passed. The absolute
memory readings include main's newer initialization/cache fixes and should
be assessed against this fresh baseline.

| Live API check | Current-main median | Managed median | Samples per version |
|---|---:|---:|---:|
| HTTP MLX transcription | 105.62 ms | 105.16 ms | 12 |
| WebSocket MLX transcription | 106.39 ms | 106.64 ms | 12 |
| HTTP Kokoro synthesis | 86.27 ms | 87.72 ms | 4 |
| WebSocket Kokoro synthesis | 86.13 ms | 86.45 ms | 4 |

Kokoro audio remained byte-identical for all four repetitions on each
transport, with 73,800 samples per response. The largest observed TTS median
increase was 1.45 ms; four timing samples provide limited tail-latency evidence.
Four concurrent HTTP ASR requests completed at 109, 212, 315 and 419 ms,
with 419.49 ms total wall time. Health-request medians were 0.585 ms before
and 0.581 ms after. Explicit CoreML discovery and HTTP/WebSocket routing
passed; its first HTTP call included 17.29 seconds of cached model loading,
and subsequent calls took approximately 602–607 ms.

Follow-up artifacts are retained in the local run's `current-main/` directory.

## Reproduction

Create a separate baseline worktree at the revision above. From the changed
worktree, build each harness sequentially:

```bash
python3 scripts/profiling/build_runtime_profile.py ../speech-swift-runtime-baseline-verify
python3 scripts/profiling/build_runtime_profile.py .
BUILD_DIR="$PWD/.build/index-build" ./scripts/build_mlx_metallib.sh release
```

Run the metallib step in the baseline worktree as well. The build helper adds
a temporary profiling target and restores `Package.swift` in a `finally`
block. It does not add production package products.

For the same input selection, construct the manifest from the existing cache:

```python
from pathlib import Path
cache = Path.home() / "Library/Caches/qwen3-speech/benchmarks/kyutai-fleurs-float-80"
rows = []
for language in ("en_us", "fr_fr"):
    for line in (cache / f"fleurs-{language}-80.tsv").read_text().splitlines()[:20]:
        audio, reference = line.split("\t", 1)
        rows.append(f"{cache / audio}\t{reference}")
Path(".build/runtime-profile-40.tsv").write_text("\n".join(rows) + "\n")
```

```bash
python3 scripts/profiling/profile_runtime_models.py \
  --baseline ../speech-swift-runtime-baseline-verify/.build/index-build/release/runtime-ownership-profile \
  --current .build/index-build/release/runtime-ownership-profile \
  --manifest .build/runtime-profile-40.tsv \
  --output-dir benchmark-results/runtime-ownership-20261017
python3 scripts/profiling/summarize_runtime_profiles.py benchmark-results/runtime-ownership-20261017
python3 scripts/profiling/check_runtime_thread_safety.py
```

For live transports, run these sequentially as well (the Python client uses
`numpy`, `soundfile` and `websockets`):

```bash
python3 scripts/profiling/profile_runtime_transports.py \
  --binary ../speech-swift-runtime-baseline-verify/.build/index-build/release/runtime-ownership-profile \
  --audio Tests/Qwen3ASRTests/Resources/test_audio.wav \
  --output benchmark-results/runtime-ownership-20261017/transports-baseline.json --tts
python3 scripts/profiling/profile_runtime_transports.py \
  --binary .build/index-build/release/runtime-ownership-profile \
  --audio Tests/Qwen3ASRTests/Resources/test_audio.wav \
  --output benchmark-results/runtime-ownership-20261017/transports-current.json \
  --tts --coreml --concurrent 4
```

The manifest is a TSV containing absolute audio paths and reference text.
It is constructed from the two cached language manifests described above.
Raw per-utterance results, binary/manifest hashes, memory snapshots and logs
are local artifacts under `benchmark-results/runtime-ownership-20261017/`.
After verification they are archived locally at
`~/Library/Logs/speech-swift/runtime-ownership-20261017/`.

## Limits

This is a regression comparison of Qwen3-ASR 0.6B MLX INT4 and full CoreML on
one machine and a small fixed bilingual subset. It is not the full model
quality or E2E matrix. Download times, uncached installation, the 1.7B variant
and full iOS builds were not measured.

The shared server lane intentionally serializes mutable model access across
protocols and engines. Concurrent request latency includes queueing; concurrent
throughput is not claimed to match the previous unsafe shared-model behavior.
Cached variants and queued playback audio still consume memory without a
global residency/queue-byte budget. Buffer timing measurements exercise the
storage and synchronization code, not physical audio-device deadlines.
