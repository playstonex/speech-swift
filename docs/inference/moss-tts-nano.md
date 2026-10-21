# MOSS-TTS-Nano inference

Add the `MossTTSNano` product to a Swift 6, macOS 15+ / iOS 18+ target. The runtime loads
compiled CoreML graphs; it does not compile `.mlpackage` files on device.

```swift
import AudioCommon
import MossTTSNano

let model = try await MossTTSNanoModel.fromPretrained() // CPU by default
let reference = try await model.encodeReference(url: referenceURL)
let result = try await model.synthesize(
    text: "The weather is nice today.", reference: reference,
    options: MossTTSNanoOptions(maxFrames: 300, seed: 1))
try WAVWriter.writeStereo(left: result.left, right: result.right,
    sampleRate: 48000, to: outputURL)
```

Omit `reference` for text-only synthesis. Keep the returned reference codes to
avoid loading/encoding audio for every request. `encodeReference(channels:sampleRate:)`
also accepts planar mono/stereo Float32 PCM and resamples both channels together.
Mono is duplicated; the last reference frame is zero-padded to 80 ms. References
must be finite, nonempty and fit the context. Use authorized reference recordings.

```swift
for try await chunk in model.synthesizeStream(text: "Hello world.", reference: reference) {
    // chunk.left/right contain 3,840 samples per channel.
    // The final marker has empty channels and isFinal == true.
}
```

The `SpeechGenerationModel` methods `generate(text:language:)` and
`generateStream(text:language:)` return the mean of left and right channels as
mono PCM. Pass `language: nil`; the model infers language from text. The stereo
methods retain both channels. Cancelling a stream stops generation at the next
CoreML call boundary. A frame-limit error can follow chunks already delivered.

```bash
speech speak "Hello world." --engine moss-nano -o hello.wav
speech speak "The weather is nice today." --engine moss-nano \
  --voice-sample reference.wav --moss-nano-seed 1 -o cloned.wav
speech speak "Hello world." --engine moss-nano --stream --play -o streamed.wav
speech speak "Hello world." --engine moss-nano \
  --moss-nano-compute-units gpu --moss-nano-max-frames 150 -o gpu.wav
```

| Option | Default | Meaning |
|---|---|---|
| `--moss-nano-model` | `aufklarer/MOSS-TTS-Nano-100M-CoreML` | Repository ID or local compiled bundle directory |
| `--moss-nano-compute-units` | `cpu` | `cpu` or `gpu` (CoreML CPU+GPU permission) |
| `--moss-nano-max-frames` | `300` | Maximum 80 ms frames; generation must reach natural EOS |
| `--moss-nano-seed` | `0` | Swift sampling seed |
| `--voice-sample` | none | Reference audio for cloning; no transcript required |
| `--clean-reference` | off | Optional Sidon cleanup of the reference, using mono audio |
| `--stream` | off | Consume 80 ms chunks as generated |
| `--play` | off | Play a mono downmix; the saved WAV remains stereo |

The CLI always saves 48 kHz stereo WAV after successful completion. The complete
prompt, reference rows and requested frame budget must fit the 1,024-row context.
Reduce text/reference length or the requested frame budget on overflow. Hitting
`maxFrames` before EOS throws an error instead of returning truncated speech.
`--language`, `--speaker`, `--instruct`, batch files and speaker listing are
unsupported for this engine.

Use `MossTTSNanoModel(modelDirectory:computeUnits:)` for a local bundle, or pass
`.cpuAndGPU` to `fromPretrained(computeUnits:)` for GPU permission. CPU is the
default because measured Python throughput on the M5 Pro was higher. Those
[measurements](../benchmarks/moss-tts-nano.md) are not Swift/iPhone benchmarks.

Validation:

```bash
swift test --no-parallel --filter 'MossTTSNanoTests.MossTTSNanoTests|MossNanoCommandTests'
MOSS_NANO_MODEL_DIR=/path/to/bundle swift test --no-parallel --filter E2EMossTTSNanoTests
```

Unit tests include upstream multilingual tokenizer fixtures, exact prompt rows,
sampling, configuration, code ranges and CLI arguments. E2E tests exercise real
synthesis, cloning, streaming, state isolation, truncation and ASR round trips.
On the matched M5 Pro export host, add `MOSS_NANO_VERIFY_CODES=1` to check exact
CPU/GPU codec-token equality with the published Python runtime. Those fixed
goldens are opt-in because CoreML numerical results can differ across hardware.
