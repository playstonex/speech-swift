# Voice Activity Projection

`VAPModel` runs the otoSpeech VAP checkpoint with Core ML. Supply synchronized
user and agent audio on separate channels. The graph uses their latest
20 seconds at 16 kHz, with leading silence for a shorter request. It returns
the most recent frame's near-term and later speaker forecasts and current
voice-activity probabilities.

```swift
import SpeechVAD

let model = try await VAPModel.fromPretrained()
let forecast = try model.forecast(userAudio: userPcm, agentAudio: agentPcm)
print(forecast.pNow)          // [user, agent]
print(forecast.pFuture)       // [user, agent]
print(forecast.voiceActivity) // [user, agent]
```

An inactive agent uses a zero channel with the same length as the user audio.
Each forecast recomputes the context window; choose a request cadence that
fits the device. The caller combines the forecasts with its turn and
interruption policy. Loading the model leaves the existing VAD and Smart Turn
pipeline settings in place.

`pNow` aggregates the first two projection bins and `pFuture` the last two.
The checkpoint's four bin durations are 0.2, 0.4, 0.6 and 0.8 seconds. Both
arrays keep the channel order `[user, agent]`. The loader rejects a manifest
with reversed channel order, incompatible geometry or an incompatible graph.

Use `VAPModel.load(from:)` for an existing compiled local bundle. Model checks
compare the original checkpoint with both exports over all 1,000 frames of
20-second test windows, and check the native API's channel order and outputs.
These checks establish numerical preservation on those inputs. TurnBench
event scores also depend on the chosen forecasting context and decision policy.

Weights: [viks66/VAP_checkpoints](https://huggingface.co/viks66/VAP_checkpoints),
`oto.ckpt` at `b9aa0ba1718221153e04ef343c7f3b8cf84bfbc3`, CC BY 4.0.
Architecture: [VoiceActivityProjection](https://github.com/ErikEkstedt/VoiceActivityProjection),
MIT. Keep the model attribution with redistributed weights.
