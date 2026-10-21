# Voxtral 4B TTS 2603

`VoxtralTTS` implements [mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)
with native MLX Swift operations. It is separate from the `VoxtralASR` module.
The model produces 24 kHz mono speech with 20 preset voices across English,
French, Spanish, German, Italian, Portuguese, Dutch, Arabic and Hindi.
The checkpoint license is CC BY-NC 4.0.

## Architecture

| Stage | Configuration |
|---|---|
| Language model | 26 causal transformer layers, width 3072, FFN 9216, 32 query / 8 KV heads, head dimension 128 |
| Positional encoding | Interleaved RoPE, theta 1,000,000 |
| Acoustic transformer | 3 bidirectional layers, width 3072, FFN 9216, 32 query / 8 KV heads |
| Frame generation | Semantic argmax plus 36 acoustic FSQ codes, 8 time points / 7 Euler updates, guidance 1.2 |
| Audio decoder | Four convolution/transformer stages, strides 1/2/2/2, two transformer layers per stage |
| Decoder attention | Causal ALiBi, windows 2/4/8/16, normalization of projected queries and keys |
| Audio output | 1920 samples per codec frame, 12.5 frames/s, 24,000 Hz |

The semantic codebook contains 8192 entries; each acoustic codebook has 21
levels. Code values reserve indices 0 and 1 for empty/end markers. Audio-code
feedback uses offsets based on the codebook sizes **including** these markers.
The audio embedding table is padded independently of the logical offsets.

## Speech prompt and frame zero

The Tekken tokenizer supplies text byte-pair ranks, control token IDs and the
number of conditioning embeddings for each voice. The prompt is:

```text
<s> [BEGIN_AUDIO] [AUDIO] × voice_frames
[NEXT_AUDIO_TEXT] text [REPEAT_AUDIO_TEXT] [BEGIN_AUDIO]
```

The voice embeddings replace the prompt's `[AUDIO]` embeddings. The last
prompt hidden state predicts the first generated frame. Inserting another
`[AUDIO]` step before generating frame zero shifts the sequence and damages
opening words. Exact prompt fixtures cover all nine languages.

## Weight format

The default bundle is `aufklarer/Voxtral-4B-TTS-2603-MLX-4bit`. It preserves the
MLX reference's affine quantization and parameter names without requantization.
The audio decoder and audio embeddings stay in floating point. Loading verifies
the complete parameter tree and shapes; it does not silently retain randomly
initialized parameters when weights are missing.

Required files are root `config.json`, `model.safetensors`, `tekken.json` and
`voice_embedding/<preset>.safetensors` for all 20 presets. The exported bundle
also includes the original parameters, tensor index, source revisions, file
hashes, model card and measured validation results.

## Ownership and memory

Each synthesis call owns its language KV cache, audio frames, codec state and random keys. One model
admits calls through `SpeechRuntime.InferenceGate` with at most 32 pending
requests. Cancellation is checked during admission, between frames and Euler
updates, and throughout audio decoding. Failed and cancelled requests do not
leave recurrent generation state on the model.

Streaming caches each codec layer's local keys and values and retains convolution
history per request. Transposed convolution overlap is carried into the next
chunk, while its final tail is trimmed exactly as in complete decoding. Chunk
boundaries do not restart normalization or introduce padding. Full and incremental
matrix shapes can produce small floating-point differences on accelerated kernels.

The decoder evaluates local causal attention in tiles. Its attention workspace
depends on the tile and local window rather than the square of the complete
utterance length. It produces the same local attention as full-sequence decoding.

The Python MLX reference is pinned at
[`441d90d`](https://github.com/Blaizzy/mlx-audio/tree/441d90d77bc2fa1667bc74cae2e434dcf7427424/mlx_audio/tts/models/voxtral_tts).
Its MIT notice is included in the module's `LICENSE-reference` resource.

See [inference and API usage](../inference/voxtral-tts.md).
