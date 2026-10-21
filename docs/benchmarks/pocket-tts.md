# Pocket TTS CPU benchmark

Measured 2026-10-17 with the native release runtime on Apple M5 Pro (18 CPU
cores, 48 GiB), macOS 26.6.2, Xcode 27 / Swift 6.3. All inference graphs use
`.cpuOnly`, including Kokoro's grapheme-to-phoneme models. CoreML chooses its
CPU thread count. Each engine runs in its own process; ASR starts only after
TTS exits. Model loading is excluded from synthesis latency.

Pocket adds earlier audio delivery on CPU. Kokoro completes the full utterance
faster and uses a smaller steady physical footprint. The voices and durations
differ; comparing listener preference requires a separate evaluation.

## Same-text comparison

Text: “The quick brown fox jumps over the lazy dog.” Pocket uses Alba, seed 42,
one warmup and five measured iterations. Kokoro uses `af_heart`, two warmups
and seven measured iterations. RTF = synthesis wall time / audio duration;
lower is faster. First audio means the first nonempty PCM callback, before
playback buffering. Kokoro returns its complete waveform in one call.

| Metric | Pocket TTS CoreML INT8 | Kokoro CoreML CPU |
|---|---:|---:|
| First PCM, median | 78.9 ms | 241.2 ms |
| First PCM, measured range | 69.4–94.4 ms | 236.7–263.0 ms |
| Full synthesis, median | 943.4 ms | 241.2 ms |
| Output duration | 2.800 s | 3.075 s |
| RTF, median | 0.337 | 0.078 |
| Peak process RSS, including loading | 866.8 MiB | 1570.1 MiB |
| Largest post-run physical footprint | 343.7 MiB | 87.0 MiB |

The Pocket process loaded in 1430.9 ms; its first synthesis
callback took 207.5 ms after loading
(1638.3 ms combined). These are
observations from a fresh process with existing CoreML/filesystem caches.
Initial model compilation and cold storage can take longer. Earlier runs
observed Pocket loads from 245 to 3,456 ms. An earlier Kokoro run had 444–1,427 ms
synthesis latency; the repeat above is included alongside its measured range.
The host was not reserved exclusively, so these samples describe this machine
and do not establish a device-wide latency guarantee.

RSS includes mapped model pages and loading temporaries. Physical footprint is
the task's currently charged memory after synthesis. Neither is the bundle's
on-disk size. The benchmark records baseline/current RSS, peak RSS and physical
footprint separately in raw JSON.

## Repeated requests and export regression

Fifteen sequential “Hello world.” requests used three warmups and twelve measured
iterations. Physical footprint grew by
0.578 MiB from the first to
last measured request. A regression test also checks eight requests after three
warmups against a 64 MiB growth allowance. The initial implementation retained
about 30 MiB of conditioning temporaries per request; explicit autorelease pools
around conditioning and per-frame predictions removed that growth.

The pinned upstream Mimi export also returned eleven state tensors under the
same names as its inputs, triggering a
[CoreML BNNS compilation diagnostic](https://github.com/FluidInference/FluidAudio/issues/294).
The exporter omits only those redundant outputs; the host keeps their unchanged
input tensors. Every recurrent input, graph operation and learned weight is
preserved. The compiled Mimi weight file is byte-identical to upstream. The fox
waveform differs from the earlier export by at most one PCM16 step, with 97.6 dB
signal-to-error ratio and correlation above 0.9999999999. The final bundle loads
without that diagnostic.

## Intelligibility and transport coverage

- Twelve phrases, 150 reference words, seed 42: **0 word errors** after lowercase
  and punctuation normalization, using independent Parakeet-TDT transcription.
- Each phrase generated four identical WAVs (one warmup plus three measured
  requests). The corpus includes short commands, multiple sentences and a long
  sentence crossing the text-chunk boundary. All final words were retained.
- Five real-model tests passed: state isolation/completion, frame-limit and
  cancellation recovery, bounded memory, CLI WAV generation, and HTTP/WebSocket
  sample preservation with HTTP disconnect recovery.
- HTTP PCM and WebSocket concatenated audio match the SDK's PCM16 bytes exactly.
  The debug transport probe observed first audio at 100 ms over HTTP and 117 ms
  over WebSocket; those single observations are separate from the release SDK
  timing table.
- The full CI unit selection executed 1,701 tests: 1,660 passed, 41 skipped,
  zero failures. Export/graph-patch tests and benchmark scoring tests passed.

The corpus is a small regression sample. Naturalness, speaker preference,
physical iPhone/iPad performance and end-to-end speaker playback remain
unmeasured. The initial bundle supports English and Alba only.

## Reproduce

```bash
swift build -c release --product pocket-tts-bench --disable-sandbox
.build/release/pocket-tts-bench --bundle-dir /path/to/pocket-bundle \
  --runs 5 --warmup 1 --seed 42 --output-dir /tmp/pocket-cpu
.build/release/pocket-tts-bench --engine kokoro \
  --bundle-dir /path/to/kokoro-cache --runs 7 --warmup 2 \
  --output-dir /tmp/kokoro-cpu
```

Run model inference one process at a time. The companion exporter benchmark
uses a fixed 12-phrase corpus, saves raw results outside the repository, verifies
fixed-seed waveform equality, and fails when normalized ASR word error rate
exceeds the requested threshold. Only this summary is committed.
