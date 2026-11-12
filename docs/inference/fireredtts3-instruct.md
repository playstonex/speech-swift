# FireRedTTS3 Instruct inference

Add the `FireRedTTS3TTS` SwiftPM product and load a compatible local export:

```swift
import FireRedTTS3TTS

let model = try await FireRedTTS3Model.fromBundle(bundleDirectory)
let result = try await model.generateWithPlan(
    text: "The weather is beautiful today.",
    instruction: "A calm, clear female voice, speaking warmly.")
// result.samples: mono Float32 PCM at model.sampleRate (24,000 Hz).
// result.voiceDescription: the checkpoint's generated voice attributes.

let chinese = try await model.generate(
    text: "你好，欢迎使用语音合成。",
    instruction: "一位声音清晰、语气亲切的普通话女声。")
model.unload()
```

The CLI requires an explicit compatible bundle and an instruction:

```bash
speech speak "The weather is beautiful today." --engine fireredtts3 \
  --fireredtts3-bundle-dir /path/to/exported-bundle \
  --instruct "A calm, clear female voice, speaking warmly." --seed 2 -o speech.wav
```

Use `--fireredtts3-model-id owner/bundle` instead of the local directory to load
a compatible published export. `--fireredtts3-max-patches` and
`--fireredtts3-flow-steps` control the audio limit and diffusion steps.
The CLI rejects `--stream` because this runtime returns complete audio.

`fromBundle` reads local files only. `fromPretrained(modelId:cacheDir:offlineMode:)`
accepts an explicitly supplied identifier for a compatible export. No default
published bundle is assumed. A bundle contains `config.json`,
`model.safetensors`, `decoder.safetensors`, `tokenizer.json`, and
`tokenizer_config.json`; distribution should retain the source license and
export manifest as well.

The text is preserved exactly, including punctuation and whitespace. English
and Chinese language aliases are accepted as validation; language is inferred
from the script. Give the instruction in the script's language. Text and
instruction must be nonempty and must not contain reserved `<|...|>` tokens.
The combined prompt is limited to 4,096 tokenizer positions.

`FireRedTTS3Options` provides request-local seed and sampling controls:

| Setting | Default | Meaning |
|---|---:|---|
| `seed` | 2 | Planning and diffusion random state |
| `flowSteps` | 10 | Euler steps per acoustic patch |
| `guidanceScale` | 1.2 | Flow instruction guidance; zero disables the negative branch |
| `stopThreshold` | 0.5 | Probability required to stop |
| `minAudioPatches` | 6 | Minimum of 960 ms generated audio |
| `maxAudioPatches` | 400 | Maximum of 64 seconds before a limit error |
| `maxPlanningTokens` | 200 | Limit before a voice-plan error |
| `textTemperature` | 0.7 | Planning temperature; zero selects the maximum logit |
| `textTopK` | 20 | Planning top-k; zero disables |
| `textTopP` | 0.8 | Planning nucleus threshold |

One instance admits one generation at a time. Cancellation is checked during
planning, every acoustic patch, every flow step, and around waveform decoding.
Plan or audio limit exhaustion throws; a truncated waveform is not returned as
a completed utterance. `generateStream` supplies one buffered final chunk.
It does not deliver incremental waveform chunks. First audio latency therefore
includes planning, acoustic generation, and full decoding.

`unload()` rejects new generation requests and releases model-owned weights.
An already admitted request retains its weights until it completes. The runtime
participates in the shared spare-buffer cache policy without changing the
process-wide wired memory limit or clearing another model's cache.
