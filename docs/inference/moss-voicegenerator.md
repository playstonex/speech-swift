# MOSS-VoiceGenerator inference

MOSS-VoiceGenerator accepts a script and a description of the desired voice. A
reference recording is unnecessary. Use a converted BF16 language and FP32 codec
bundle; the upstream language folder alone does not contain the synthesis codec.

```bash
speech speak "The little boat crossed the quiet lake before sunset." \
  --engine moss-voicegenerator \
  --moss-voicegenerator-bundle-dir /path/to/moss-voicegenerator-bundle \
  --instruct "A warm adult female voice, speaking slowly and gently." \
  --seed 42 --stream --output speech.wav
```

```swift
import Foundation
import MossVoiceGeneratorTTS
import AudioCommon

let model = try await MossVoiceGeneratorModel.fromBundle(
    URL(fileURLWithPath: "/path/to/moss-voicegenerator-bundle"))
let result = try await model.generate(
    text: "The little boat crossed the quiet lake before sunset.",
    instruction: "A warm adult female voice, speaking slowly and gently.",
    options: .init(maxFrames: 150, seed: 42))
try WAVWriter.write(samples: result.samples, sampleRate: result.sampleRate,
                    to: URL(fileURLWithPath: "speech.wav"))
```

`fromPretrained(modelId:)` downloads an already converted compatible bundle chosen
by the caller. It does not convert the upstream model or choose an unpublished
default repository. `offlineMode: true` requires all files in the local cache.

Text and instruction must each fit 32,768 UTF-8 bytes, checked before cleanup or
tokenization. The complete rendered prompt, including processor and instruction
tokens, is limited to 4,096 tokenizer positions before tensor allocation. These
native safety limits are separate from the checkpoint's 40,960-position context;
longer inputs throw instead of allocating an unbounded prefill attention mask.

`generateStreaming` emits `AudioChunk` values while preserving decoder state. It
returns the same full result and timing fields as `generate`. One request may run
per model instance. Cancellation releases request caches, and exceeding the frame
limit throws instead of returning an unfinished utterance.

The model implements `ModelMemoryManageable`. `isLoaded` reports availability and
`memoryFootprint` estimates weight storage. `unload()` is idempotent and immediately
rejects new requests. An active request stops at its next safe boundary, then
releases its weights and cache lease. Create a new instance to load it again.

The model and its loader request a 512 MiB spare-buffer budget through the shared
cache coordinator. This temporarily sets the spare-buffer target for cooperating models in
the same process; stricter caller limits remain in force. Other owners' budgets
are restored after unload. MOSS does not flush the process-wide cache during
generation or unload, so cached reusable buffers may remain after live weights
are released. The allocator can transiently exceed the requested spare-buffer
budget by buffer granularity; it is not a total process-memory limit.

```swift
let result = try await model.generateStreaming(
    text: "Good morning. Welcome home.",
    instruction: "A relaxed adult male voice with a soft, friendly tone.") { chunk in
        // Consume mono PCM at chunk.sampleRate.
        consume(chunk.samples)
    }
```

The default frame limit is 750, about 60 seconds at 12.5 frames per second. The
default codec chunk contains three frames, or 240 ms of audio. The model first
fills its sixteen-codebook delay, so the first audio arrives later than the first
generated token. The result exposes both timings and total generation time.

Text and audio temperatures default to 1.5, top-k to 50, text top-p to 1.0, audio
top-p to 0.6 and audio repetition penalty to 1.1, following the upstream generator.
A zero temperature selects greedily. `normalizeInputs: false` preserves exact
benchmark scripts and descriptions. Enabling it removes bracketed metadata and
adjusts punctuation as the upstream processor does.

Language, age, gender, accent and delivery requirements belong in the instruction.
The generic `SpeechGenerationModel` entry point uses a neutral instruction and
rejects a separate language override. Use the explicit instruction API for voice
design and instruction-following measurements.

Keep baseline comparisons on the same scripts, instructions, normalization and
source languages. Record the seed, sampling settings, bundle hashes, audio code
count, generated duration, first audio, total runtime and physical memory. Content
checks should transcribe saved audio after releasing the synthesis process.
