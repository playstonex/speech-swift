# MOSS-VoiceGenerator

MOSS-VoiceGenerator synthesizes speech from an exact script and a free-form
description of the desired voice and delivery. It needs no reference recording.
It is a different model from MOSS-Audio, which analyzes audio, and MOSS-TTS-Nano,
which uses a different speech representation and decoder.

The `MossVoiceGeneratorTTS` module implements the language network, delayed audio
codebooks and synthesis decoder in MLX Swift. Inference does not invoke Python.

## Supported checkpoint

| Component | Pinned source | Precision |
|---|---|---|
| Voice generator | [OpenMOSS-Team/MOSS-VoiceGenerator](https://huggingface.co/OpenMOSS-Team/MOSS-VoiceGenerator/tree/97521ec2b6f3ec5026ac1f5751f8fc302d82c2d4) | Original BF16 |
| Audio tokenizer synthesis decoder | [OpenMOSS-Team/MOSS-Audio-Tokenizer](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer/tree/3cd226ba2947efa357ef453bcad111b6eafba782) | Original FP32 |

Both upstream components use Apache 2.0. The conversion retains source tensor
values. The encoder and unused final sixteen codebooks are omitted because
reference-free synthesis never calls them. INT4 is not supported. INT8 requires
separate conversion and validation and is not accepted by the BF16 loader.

## Language network and processor

The backbone follows Qwen3: 28 layers, width 2,048, 16 query heads, eight key/value
heads, per-head query/key normalization, split-half rotary positions and SwiGLU.
Seventeen independent output heads predict text and sixteen audio codebooks.
The text output head is untied. Every input position sums the text embedding and
all sixteen audio embeddings, including learned padding embeddings.

The processor creates the upstream reference-free `UserMessage` and ChatML
assistant prefix. Unused reference, quality, language, sound-event and token-count
fields contain the literal `None`. The script and instruction remain unchanged
by default. `normalizeInputs` optionally applies the upstream punctuation and
metadata cleanup; benchmark runs must record that setting.

The native processor rejects text or instruction fields above 32,768 UTF-8 bytes
before rendering or normalization. It admits at most 4,096 tokens in the complete
rendered prompt before creating tensors. This bounds the initial dense attention
mask independently of the checkpoint's larger total context.

Audio codebooks have 1,024 values plus a padding value. Codebook number `n` starts
`n` generation steps after the first book. A complete audio frame becomes
available only after all sixteen values arrive. The native scheduler drains the
delayed tail before accepting an audio-end token; incomplete output throws.

Sampling follows the pinned implementation: temperature, repetition penalty,
top-k filtering, then nucleus filtering. The pinned Python generator flattens
heads 1–15 before applying its repetition helper, so those heads share their
prior token IDs; head zero has a separate history. Request-local MLX random keys
avoid changing another model's random state. A seed is reproducible within this
runtime; it does not promise the same random draws as PyTorch.

## Speech decoder

Sixteen learned eight-dimensional codebooks project to width 512 and sum before
a projection to width 768. The one-by-one projections fold the original weight
normalization without reducing precision. Four causal transformer stages alternate
with time expansion by factors 2, 2, 2 and 240. Each input frame produces exactly
1,920 samples at 24 kHz: 12.5 frames per second, mono Float32 PCM.

The decoder uses adjacent-pair rotary positions, LayerNorm, exact GELU and learned
residual scales. These differ from the language network and are implemented
separately. Per-layer caches retain ten seconds of context and absolute positions.
Chunked decoding preserves the upstream offline causal result. A tiny reference
fixture documents an upstream ring-cache issue: writing a whole multi-token chunk
can overwrite history required by early queries. The native cache retains that
history rather than reproducing the overwrite.

## Bundle and validation

`config.json`, `model.safetensors`, `codec_config.json`, `codec.safetensors`,
`tokenizer.json`, `tokenizer_config.json` and `mvg_export.json` are required.
The loader checks the family, control IDs, dimensions and storage precision
before generation. An unconverted upstream tokenizer folder is not this layout.

Explicit unloading marks the model unavailable under the same lock used for
request admission. If a request is active, resource release waits until it exits;
the generation loop checks availability before further inference and audio
delivery. This avoids clearing weights while an active request uses them.

Loading and the loaded model register a 512 MiB spare-buffer ceiling with the
shared cache coordinator. A tighter caller limit is preserved, and other owners'
budgets are restored when MOSS unloads. The generation loop and unloading avoid
global cache flushes. Unloading releases this model's live parameters; reusable
allocator buffers may remain within the restored caller/shared budget.

Unit checks cover the processor, padding, end-of-speech schedule and invalid
configuration. Small numerical fixtures compare transformer output, cached
chunks, rotary positions, patch ordering and weight-normalized projections with
the pinned upstream code. Model tests require an explicitly chosen local bundle;
they emit audio and timings for separate content and memory checks. Measured
performance belongs in the [benchmark report](../benchmarks/moss-voicegenerator.md).
