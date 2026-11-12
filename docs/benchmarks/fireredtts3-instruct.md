# FireRedTTS3 Instruct validation

This implementation must pass numerical and waveform checks before benchmark
scores or performance claims are published. The initial weights retain their
source Float32 precision; the official CUDA backbone uses BF16 autocast.
Compare acoustic content and instruction behavior separately from random
waveform identity.

`E2EFireRedTTS3NumericalTests` uses tiny synthetic CPU fixtures from the pinned
official PatchEncoder, DiT, RedAE decoder class bodies, and Transformers Qwen3.
It covers full and cached backbone steps, a sliding decoder with head width
different from hidden-width divided by head count, patch encoding, flow
velocity, and final waveform decoding. A separate NumPy inverse FFT fixture
tests frame-loop overlap-add, one-frame boundaries, periodic windows, and
imaginary DC/Nyquist behavior. Source revisions and file hashes are retained
with the fixtures. No pretrained weights are included in tests.

Run the tests sequentially after building the package and its MLX metallib:

```bash
swift test --no-parallel --filter FireRedTTS3ConfigurationTests
swift test --no-parallel --filter E2EFireRedTTS3NumericalTests
FIREREDTTS3_E2E_BUNDLE=/path/to/bundle \
  swift test --no-parallel --filter E2EFireRedTTS3BundleTests
```

The real-bundle test checks English and Chinese planning, valid waveform
length, cancellation recovery, and explicit unload. Independently transcribe
fixed generated clips before judging instruction following: finite audio and a
plausible plan do not establish that the complete script was spoken correctly.

For each fixed pilot clip report the exact source text and instruction, seed,
weight revision and precision, options, output duration, load time, first audio
time, planning/acoustic/decoding times, wall time divided by audio duration,
peak process physical footprint (`TASK_VM_INFO.ledger_phys_footprint_peak`),
plus resident process memory and peak active MLX memory. Run a warm repeat and
state whether model loading is included. Keep raw logs outside the repository.

Run existing Qwen, Breeze, voice cloning, and memory/cache checks in separate
processes before and after integration. Do not overlap model loading, builds,
or benchmark processes. Shared runtimes and public defaults are unchanged by
the FireRed module itself; the CLI integration must validate explicit engine
selection and unsupported controls separately.

## Local validation, 29 October 2026

The Float32 export passed five reference numerical tests, both real-bundle tests
(English/Chinese synthesis, cancellation recovery and unload), and the CLI test
with an independent Parakeet check of every word. The fixed English and Chinese
measurement scripts both transcribed with zero normalized error. All four seeded
repetitions per language produced identical WAVs.

Release measurements on Apple M5 Pro with 48 GiB unified memory used the compiled
MLX metallib and `MLX_ENABLE_TF32=0`. Each language had a fresh process, one first
request and three warm repetitions of one short script. Checksums ran before the
timed model load with bounded temporary buffers. Filesystem caches were warm.

| Script | Audio duration | Median warm complete PCM | Median warm RTF | Peak process physical footprint |
|---|---:|---:|---:|---:|
| English | 2.72 s | 4.50 s | 1.65 | 10.82 GB |
| Chinese | 5.28 s | 11.22 s | 2.13 | 11.70 GB |

Model loading took 0.60 s and 0.57 s respectively, excluded from request timings.
Complete PCM latency includes voice planning, acoustic generation and decoding;
this implementation buffers the complete waveform. Physical memory is the
kernel process high-water value across loading and all four requests, in decimal
GB. MLX active weight memory returned to approximately zero after explicit unload;
allocator buffers and operating-system residency are separate measurements.

These are short-script measurements on a shared machine. They are not a language
coverage study or an instruction-following score. Existing-model seeded WAVs
matched before and after integration; speed comparisons require comparable host
load and are reported separately. No InstructTTSEval score is claimed for this
new runtime yet.
