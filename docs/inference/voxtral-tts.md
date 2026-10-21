# Voxtral TTS inference

`VoxtralTTS` synthesizes speech with the checkpoint's preset voices, either as
a complete waveform or as PCM chunks during generation. Input text is supplied
in full before generation starts.
It produces mono Float32 PCM at 24,000 Hz. The model weights use CC BY-NC 4.0.

## CLI

```bash
speech speak "The quick brown fox jumps over the lazy dog." \
  --engine voxtral --voxtral-voice casual_male -o speech.wav

speech speak "Bonjour, nous testons la synthèse vocale." \
  --engine voxtral --voxtral-voice fr_female -o bonjour.wav

speech speak --engine voxtral --list-speakers

speech speak "Hello, welcome to Voxtral." --engine voxtral \
  --stream --voxtral-chunk-frames 3 --play
```

| Option | Default | Purpose |
|---|---|---|
| `--voxtral-model-id` | `aufklarer/Voxtral-4B-TTS-2603-MLX-4bit` | Published MLX bundle |
| `--voxtral-bundle-dir` | — | Load a local bundle instead of downloading |
| `--voxtral-voice` | `casual_male` | Preset voice; an explicit language selects its male preset when this flag is omitted |
| `--voxtral-max-frames` | 500 | Maximum generated audio frames; 500 frames is 40 seconds |
| `--voxtral-seed` | 0 | Per-request random seed |
| `--voxtral-chunk-frames` | 3 | Codec frames per streaming chunk; 3 frames is 240 ms of audio |
| `--stream` | false | Decode and deliver PCM during frame generation; report first-audio latency |
| `--list-speakers` | false | List presets without loading weights |
| `--play` | false | Play the generated utterance |

The CLI rejects batch synthesis and reference-audio cloning for this
engine. It also rejects unsupported languages, unknown voices, empty text and
invalid limits before loading weights. Buffered synthesis returns no partial
utterance on a frame-limit error. A stream can fail after earlier chunks were
delivered; it then throws without a successful final marker. The CLI writes a
WAV only after successful completion. Increase the limit or split long text at
sentence boundaries.

With `--stream --play`, playback starts during generation using a 160 ms
prebuffer. The playback path preserves quiet leading chunks. Without `--play`,
the CLI collects the chunks and writes a complete WAV after synthesis.

## Swift

```swift
import Foundation
import VoxtralTTS

let model = try await VoxtralTTSModel.fromPretrained()
let samples = try await model.generate(
    text: "Bonjour, nous testons la synthèse vocale.",
    voice: "fr_female",
    options: VoxtralTTSOptions(maxFrames: 500, seed: 0)
)

for try await chunk in model.generateStream(text: "Hello, welcome to Voxtral.", voice: "casual_male") {
    print(chunk.frameIndex, chunk.samples.count, chunk.isFinal)
}

// Local export:
let local = try VoxtralTTSModel.fromBundle(URL(fileURLWithPath: "/path/to/bundle"))
```

`fromPretrained` downloads the config, tokenizer, backbone weights and all 20
preset files in `voice_embedding/`. With `offlineMode: true`, every bundle file
must already be cached; missing files are reported before model weights load.

The model conforms to `SpeechGenerationModel`. Its protocol `generate(text:language:)`
selects a language's default male preset. Explicit voice synthesis uses the overload
above. Both `generateStream` overloads deliver new PCM during generation. Each
chunk starts at `frameIndex` and normally contains 3 codec frames (240 ms).
`VoxtralTTSOptions.streamingChunkFrames` accepts 1...125. Exactly one final chunk
marks successful completion; it can be empty when speech ends on a chunk boundary.
No convolution tail or extra silence is appended at the end.

`generateStreaming(text:voice:options:onChunk:)` awaits each callback before
generating more frames. Use it for transport backpressure or delivery errors.
Throwing from the callback releases model admission. The `AsyncThrowingStream`
adapter advances generation only when the consumer asks for the next chunk.
It has no background producer or PCM backlog. Cancelling the consuming task
while awaiting a chunk aborts generation. Drop an unfinished stream when ending
iteration so its request state and admission permit are released.

`VoxtralTTSOptions` also exposes `denoisingSteps` (default 8) and `guidance`
(default 1.2). Defaults follow the reference inference implementation. Random
keys are local to each request and do not reset another model's random generator.

## Presets

| Language | Voices |
|---|---|
| English | `casual_male`, `casual_female`, `cheerful_female`, `neutral_male`, `neutral_female` |
| French | `fr_male`, `fr_female` |
| Spanish | `es_male`, `es_female` |
| German | `de_male`, `de_female` |
| Italian | `it_male`, `it_female` |
| Portuguese | `pt_male`, `pt_female` |
| Dutch | `nl_male`, `nl_female` |
| Arabic | `ar_male` |
| Hindi | `hi_male`, `hi_female` |

This runtime uses the released preset embeddings. It does not accept custom
reference audio or provide a reference-audio encoder.

## HTTP and WebSocket

The server recognizes `voxtral`, `voxtral-tts`, `voxtral-tts-4bit` and the canonical
name `voxtral-4b-tts-2603-mlx-4bit`.

```bash
curl http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"voxtral-tts","input":"Bonjour, nous testons la synthèse vocale.","voice":"fr_female","response_format":"wav"}' \
  -o bonjour.wav
```

`/speak` accepts `model: "voxtral-tts"`, `text` and an optional `voice`. For
`/v1/realtime`, select the model and voice with `session.update`, then send
`response.create` with text instructions. The server sends ordered 24 kHz PCM16
deltas as codec chunks become available. HTTP and WebSocket share the existing
bounded inference lane and model cache. Unknown voices and invalid text return
client errors; model failures remain explicit server errors.

Generic OpenAI voice names select the engine default; native names such as
`fr_female` pass through unchanged. Speed adjustment is unavailable for Voxtral.

For streamed HTTP output, request `stream: true` with `response_format: "pcm"`:

```bash
curl http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"voxtral-tts","input":"Hello, welcome to Voxtral.","voice":"casual_male","stream":true,"response_format":"pcm"}' \
  -o hello.pcm
```

This returns raw 24 kHz mono signed PCM16 little-endian audio. Streamed WAV is
rejected because its final sample count is not yet known. Streaming HTTP is
currently supported for Voxtral only; other engines retain buffered responses.
The server holds admission through body delivery and awaits each write. Loading
and input errors return an HTTP error before audio headers. An inference or
connection failure after headers aborts the stream instead of reporting success.


## Validation

Run the focused unit tests and isolated model tests sequentially:

```bash
swift test --no-parallel --filter 'VoxtralTTS|VoxtralSpeakCommand|VoxtralRouting' --skip E2E
VOXTRAL_TTS_BUNDLE_DIR=/path/to/bundle \
  swift test --no-parallel --filter E2EVoxtralTTSTests
VOXTRAL_TTS_BUNDLE_DIR=/path/to/bundle \
  swift test --no-parallel --filter E2EVoxtralTransportTests
MLX_ENABLE_TF32=0 VOXTRAL_TTS_BUNDLE_DIR=/path/to/bundle \
  swift test --no-parallel --filter E2EVoxtralTTSTests/testDecoderMatchesReferenceAcrossAttentionTilesAndFinalSamples
```

The model suites check buffered and streamed English/French ASR round trips, speech openings, prompt
parity in nine languages, deterministic repeated/concurrent requests, cancellation
recovery, explicit frame-limit errors, codec parity across chunk boundaries and
transport output against the corresponding buffered or streaming SDK API.
The decoder fixture uses full FP32 matrix products. Setting `MLX_ENABLE_TF32=0`
before process launch matches that reference precision; default M5 acceleration
is checked with a bounded waveform-error tolerance. Performance measurements use
the normal MLX settings.
