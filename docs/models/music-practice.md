# Music passage export

The unreleased `speech practice` command exports a selected passage as a WAV
file. Speed and pitch are independent: slowing a passage preserves its key,
and a semitone shift preserves the requested playback speed. Repetitions are
baked into the completed file rather than stored as player-only settings.

```sh
speech practice song.wav --start 20 --duration 12 --speed 0.8 \
  --semitones -2 --loops 3 --output practice.wav --json
```

This source-build example exports three repetitions of a 12-second selection,
played at 80% speed and shifted down two semitones. Its completed duration is
45 seconds. `--json` writes one schema-versioned export report to standard
output; it does not play audio through the speakers or download a model.

## Native API

```swift
import MusicTools

let report = try MusicPracticeExporter.export(
    input: inputURL,
    output: outputURL,
    options: MusicPracticeOptions(
        startSeconds: 20, durationSeconds: 12,
        speed: 0.8, semitones: -2, loops: 3))
```

The implementation uses AVFoundation offline rendering and
`AVAudioUnitTimePitch` on macOS and iOS. It reads and writes bounded 4096-frame
PCM buffers. A float PCM scratch passage is rendered once, then streamed into
the repeated 16-bit WAV. Scratch files sit beside the destination and are
removed after success or ordinary thrown errors. This requires temporary disk
space for the transformed passage plus the completed WAV; it does not retain
the full repeated recording in memory. At the maximum sample rate, channel
count and permitted durations, those files can approach 5.53 GB in total.
The API is synchronous and does not provide cancellation. The caller must
provide access to sandbox-protected URLs and choose an appropriate execution
context. Desktop verification does not establish iOS device behavior.

## Selection and export contract

- Inputs are mono or stereo formats decodable by AVFoundation, at 8–192 kHz.
- Start is nonnegative and before the input ends. An explicit duration must
  fit inside the recording; omitted duration selects the remaining input.
- A selected passage is at most 600 seconds. Speed is 0.25–2.0; pitch is
  −12 to +12 semitones; loops is the total repetition count, from 1 to 100.
- The completed output is at most one hour. Duration rounds up to a whole
  output frame per repetition. Sample rate and channel count are preserved.
- Each repetition has a 5 ms fade at both ends by default. `--fade-ms` accepts
  0–50 ms. Fades do not remove frames or add a gap between repetitions.
- A single shared gain attenuates peaks above 1 without boosting quiet audio.
  The JSON report records that gain, actual selected range, output frames,
  output duration, speed, pitch, and repetition count.
- Output is a `.wav` file. Existing destinations are rejected unless
  `--overwrite` is supplied. Source identity is checked through resolved paths
  and file identity, including symbolic links. A completed staged WAV is
  published only after processing succeeds.

Offline rendering retries transient no-progress statuses a bounded number of
times and fails instead of looping indefinitely. Time-pitch latency is removed
using the audio unit's reported latency before writing the requested passage
frame count. Synthetic tone tests check pitch, speed, duration, and audible
content near both ends. A sparse-burst fixture checks that complete beginning
and final phrases survive at 0.25× and 2×, including transposition. These checks
permit transient smearing and partial transient loss; they are not claims
about perceptual quality on every recording.

The command does not estimate BPM, detect keys, separate stems, or create MIDI.
Those are distinct operations. This command is not in the 0.0.29 release.
