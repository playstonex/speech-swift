# Kyutai STT

`KyutaiSTT` implements Kyutai's delayed-stream speech recognition on Apple Silicon with MLX. The default checkpoint is [aufklarer/Kyutai-STT-1B-MLX-8bit](https://huggingface.co/aufklarer/Kyutai-STT-1B-MLX-8bit), derived from [kyutai/stt-1b-en_fr](https://huggingface.co/kyutai/stt-1b-en_fr), for English and French. The same configuration reader supports [kyutai/stt-2.6b-en](https://huggingface.co/kyutai/stt-2.6b-en), for English; real-weight 2.6B inference has not been validated.

| Property | 1B English/French | 2.6B English |
|---|---|---|
| Transformer width | 2,048 | 2,048 |
| Layers / attention heads | 16 / 16 | 48 / 32 |
| SwiGLU intermediate width | 5,632 | 5,632 |
| Text vocabulary | 8,000 | 4,000 |
| Attention window, including current frame | 750 | 375 |
| Trained text delay | 0.5 s | 2.5 s |
| Silence prefix | 0 frames | 13 frames (1.04 s) |
| Audio | Mono Float32 PCM, 24 kHz | Same |
| Mimi frame size / rate | 1,920 samples / 12.5 Hz | Same |
| Audio codebooks | 32 × 2,048 entries | Same |
| Weight license | CC-BY-4.0 | CC-BY-4.0 |

A causal Mimi encoder produces 32 audio codes every 80 ms. The text decoder sums their embeddings with the previous text token's embedding, applies interleaved RoPE attention, float32 RMS normalization, and SwiGLU layers, and predicts one text token per frame. There is no depth transformer or generated audio stream. At step zero, text and audio use their initial-token embeddings. Each later decoder step consumes the **previous** audio frame, matching Kyutai's `LMGen` scheduling.

The module reuses `PersonaPlex`'s Mimi encoder, transformer, downsampler, codebooks, and temporal decoder layers. Its encoder-only codec omits audio-generation modules and their weights. The existing `Mimi.sanitize(weights:)` delegates to `Mimi.sanitizeWeights(_:)`, which maps checkpoint keys without constructing a codec. `MimiConfig.moshiko(numCodebooks:padMode:)` accepts a padding mode; Kyutai selects constant zero padding while the existing default remains edge padding. The codec uses exact GELU and causal attention as in the original PyTorch implementation, processing the two 25 Hz encoder frames together before downsampling. Codec and text-decoder work are submitted together with one synchronization per 80 ms block. It owns its STT configuration, checkpoint mapping, bounded attention caches, SentencePiece decoding, and session lifecycle. Each session retains recent attention state while preserving absolute RoPE positions. It does not retain all historical audio frames.

## Checkpoints

The loader accepts original PyTorch-layout safetensors from the two upstream repositories and native exports with `weight_layout: speech_swift_v1`, `format_version: 1`. Required runtime files are:

- `config.json`: original architecture and STT timing, plus optional quantization metadata.
- `model.safetensors`: complete STT decoder weights.
- The `mimi_name` file: original Mimi tensor layout, loaded in Float32.
- The `tokenizer_name` file: SentencePiece vocabulary matching `text_card`.

Original `transformer.layers.*`, `.in_proj_weight`, and RMSNorm `.alpha` keys map to the Swift parameter tree. Missing, unused, or incorrectly shaped decoder weights fail loading rather than leaving randomly initialized parameters. Unknown architectures, nonzero codebook delays, invalid timing, and unsupported quantization are rejected before model allocation.

Exports support FP16, affine INT4, and affine INT8 (group size 64). Only transformer attention and feedforward matrices are quantized. Embeddings, text output projection, and normalization remain FP16; Mimi is copied without quantization. `export.json` records the upstream repository, pinned revision, precision, and file sizes.

`KyutaiSTTVariant` selects FP16, INT4, or INT8 exports. The default is INT8; `KyutaiSTTModel.upstreamModelId` selects the original 1B layout. Local exports can also be selected by directory. Runtime file sizes include the complete upstream Mimi file, but only encoding tensors are loaded; missing or incorrectly shaped encoding tensors fail strict loading, and unknown tensor prefixes are rejected.

## Scope

Text language is inferred from audio; language prompts and context biasing are unavailable. Token timestamps are approximate frame positions corrected for the silence prefix and trained text delay, rather than forced word alignments. Semantic VAD extra-head checkpoints are not supported. The 2.6B configuration and exporter are supported; real-weight validation results must be reported separately from the 1B checks.

See [inference](../inference/kyutai-stt.md) for API and CLI examples. Upstream implementation: [delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling) and [Moshi](https://github.com/kyutai-labs/moshi).
