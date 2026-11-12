# Breeze TTS 2

`BreezeTTS` implements Breeze TTS 2 natively with MLX Swift. It produces English
and Chinese speech at 24 kHz, supports reference-free voice design, and clones or
directs a reference voice when supplied with its complete transcript.

The runtime contains a 26-layer T5Gemma2 text encoder (hidden size 1,152), a
28-layer Qwen3 backbone (hidden size 2,048), and a 12-layer depth decoder (hidden
size 1,024). Each 80 ms frame contains 16 codebook IDs in `0..<2048`. The backbone
predicts the first codebook or EOS (2051); the depth decoder predicts the other
15 codebooks using a fresh KV cache for each frame. The backbone cache persists
through the utterance. Instruction guidance evaluates positive and negative
prompts; the depth branches are batched together.

The text encoder uses bidirectional local attention and a full-attention layer
every sixth layer. Full attention applies the checkpoint's linear RoPE scaling
(factor 8). The depth decoder uses Llama 3 frequency scaling. These settings are
read from the checkpoint rather than inferred from model names.

Audio decoding and reference encoding reuse `Qwen3TTS`'s speech tokenizer.
Breeze configures its two pre-upsampling stages with the bundled kernel size of
2, preserving exactly 1,920 samples for each generated frame and at every chunk
boundary.
Reference audio is mono Float32 PCM at 24 kHz. Streamed chunks use an overlapping
codec window (25 left-context frames by default); this is bounded recomputation,
and chunk boundaries can differ from decoding the whole utterance at once.
`generate()` assembles the same chunks into a complete waveform. Reaching the
frame cap before EOS throws instead of returning a truncated success.

The exporter in `scripts/export_breeze_mlx.py` pins the upstream revision, keeps
the audio tokenizer unchanged, removes unused legacy codec/text embeddings, and
stores tied audio embeddings once. Published bundles use BF16 or affine group-64 8-bit Linear
weights are supported. Embeddings and the depth codebook head retain their
upstream precision. The published bundles are
[Breeze TTS 2 MLX 8-bit](https://huggingface.co/aufklarer/Breeze-TTS-2-MLX-8bit)
and [Breeze TTS 2 MLX BF16](https://huggingface.co/aufklarer/Breeze-TTS-2-MLX-bf16).

Weights, derivatives, and self-hosted outputs are governed by the
[BreezeBlue Research and Non-Commercial License](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE).
Converted bundles preserve the full license, required `NOTICE`, and provenance.
The unchanged Qwen3 audio tokenizer also includes its separate Apache-2.0
license and pinned source attribution in `audio_tokenizer/LICENSE`.
The Swift configuration and text encoder include MIT attribution in
`ThirdPartyNotices/BreezeTTS.txt`.

Sources: [official implementation](https://github.com/breezeblue-ai/breeze-tts),
[pinned checkpoint](https://huggingface.co/BreezeBlue/Breeze-TTS-2/tree/3e28c5151381a722f1d8661b4118c298caa77aa4).
See [usage](../inference/breeze-tts.md) and [benchmark procedure](../benchmarks/breeze-tts.md).
