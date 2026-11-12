# p99 turn-1-mini

`P99TurnModel` runs p99lab's 5.3M-parameter audio-only turn-completion model
with Core ML. It accepts 16 kHz mono PCM. The compiled graph contains the
required 4 kHz filtering, log-mel frontend, transformer and streaming state.
The upstream model targets English.

```swift
import SpeechVAD

let model = try await P99TurnModel.fromPretrained()
let session = try model.createSession()
let frames = try session.pushAudio(pcm16k)
for frame in frames {
    print(frame.probability, frame.availableAfterSeconds)
}
try session.reset()
```

Use one session per conversation and serialize calls to it. Arbitrary input
chunk sizes are buffered into 2,560-sample steps, each returning eight
probabilities. An incomplete step remains buffered until more audio arrives.
`reset()` clears that pending audio, model history and timestamps.

`frameEndSeconds` identifies a 20 ms model frame. `availableAfterSeconds`
records the end of the 160 ms step that made it available. Use the latter for
live decision timing. The caller selects a response policy from the probabilities.

For an existing local compiled bundle, use `P99TurnModel.load(from:)`.
The loader validates the manifest and the graph's input/output names, shapes
and Float32 types before creating a session.

The Float32 export was compared with the upstream streaming ONNX graph on
noise, silence, clean tones and quiet tones. Native checks also compare a
fixed reference across chunk sizes and reset. These are numerical checks;
the TurnBench submission's event scores use its own published policy.

Source: [p99lab/turn-1-mini](https://huggingface.co/p99lab/turn-1-mini),
revision `076c86d1744a231d3a4cce31f3bbbd1211b4c281`. Model and code:
Apache 2.0; retain the accompanying licence and notice.
