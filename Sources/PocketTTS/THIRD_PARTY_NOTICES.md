PocketTtsLayerKeys.swift and PocketTtsMimiKeys.swift are adapted from FluidAudio,
Copyright FluidInference, licensed under Apache License 2.0:
https://github.com/FluidInference/FluidAudio/tree/2d6c42ca5f3accc0faf9fee9e0fff5b9b2cf6042/Sources/FluidAudio/TTS/PocketTTS/Pipeline

Changes: removed conditional compilation; reject unrecognized Mimi state inputs;
carry explicitly declared unchanged state when redundant outputs are omitted.
A copy of the license is in Apache-2.0.txt.

The separately downloaded Pocket TTS weights are CC BY 4.0, by Kyutai;
CoreML conversion by FluidInference. The Alba voice is performed by Alba MacKenna,
CC BY 4.0: https://huggingface.co/kyutai/tts-voices#alba-mackenna.
Retain the model bundle NOTICE when redistributing weights.
