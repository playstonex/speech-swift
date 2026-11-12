# Fish Audio Drama 3

Experimental native MLX implementation of Fish Audio Drama 3 in `FishAudioTTS`.
Built with Fish Audio. The model's [Community License](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE)
and NOTICE accompany exported bundles.

| Component | Format |
|---|---|
| Source | `fishaudio/drama-3` |
| Default MLX checkpoint | `mlx-community/fish-audio-drama-3-4bit` |
| Swift API | `FishAudioDrama3Model` |
| Backbone | 40 layers, 2048 hidden channels, approximately 35B total / 3B active parameters |
| Attention | 30 gated DeltaNet layers and 10 gated full-attention layers |
| Experts | 256 routed experts per layer, top 8, plus a gated shared expert |
| Audio tower | 4 shared enrichment layers, 9 residual codebook layers, temporal KV caches |
| Codec | Fish DAC, 10 codebooks, 44.1 kHz mono, 2048 samples per frame |
| Prompt | `reference-bracket-v1`, speaker tags 0–15, inline emotion markers |

Drama 3 is a separate architecture from S2 Pro. `FishAudioTTSModel` continues to
load S2 Pro; `FishAudioDrama3Model` loads converted Drama 3 bundles. Both reuse
the Fish DAC implementation and codebook/reference value types.

## Weight format

The runtime accepts the converted `fish_drama3` format, including floating-point
and per-module mixed affine quantization. Raw `fish_omni` FP8 checkpoints are
rejected before allocating the network.

- `backbone.model.*` holds embeddings, hybrid attention, MoE, and final norm.
- `semantic_head.*` contains only the 4096 semantic rows followed by EOS.
- `tower.codebook_embeddings.*`, `tower.shared_layers.*`,
  `tower.residual_layers.*`, and `tower.codebook_heads.*` hold the temporal tower.
- `codec.safetensors` holds the raw FP32 Fish DAC parameter names.
- `config.json` lists every quantized module's bit width and group size.

Shifted normalization weights store `1 + gamma` in FP32. The runtime evaluates
these norms and partial RoPE in FP32. DeltaNet's output norm uses its ordinary
learned weight; recurrent state remains FP32. Tower Q/K normalization is always
enabled, including for configs whose `attention_qk_norm` is false.

The 4-bit checkpoint uses 4-bit experts and mostly 8-bit dense projections.
Embeddings, routing, and decay projections remain floating point. The variant
name alone does not describe every tensor's precision.

## Generation

Each prompt position contains a primary token and 10 raw codebook ids. On
semantic positions the backbone adds the configured first residual codebooks
to the text embedding and scales the sum. The remaining codebooks enter the
first shared tower layer.

Prompt prefill runs in 128-position chunks. Tower position `p` is teacher-forced
with the codes at `p + 1`; the final prompt position is filled during the first
generation step. All tower layers attend over time and retain their caches
across generated frames. S2 Pro's per-frame fast-decoder cache does not apply.

Sampling selects semantic tokens or EOS, then predicts the 9 remaining
codebooks. It follows upstream's top-k/top-p crossing-token rule and a 16-frame
repetition-aware retry. The seed and caches belong to each request; an inference
permit serializes calls on the same model. Cancellation is checked between
prefill chunks and generated frames. Reaching the frame limit throws without
returning a truncated waveform.

Buffered and streaming synthesis share a request-local frame iterator. Streaming
decodes new frames through Fish DAC with causal convolution history,
transposed-convolution overlap and per-layer attention caches. Attention retains
127 past frames and applies the original 128-frame window with global rotary
positions. State is released at request completion, cancellation or delivery
failure. The final convolution tail is trimmed exactly as in full decoding.
Callbacks apply backpressure; the async iterator decodes on demand.

The tower uses a 4,096-token vocabulary for each codebook. Buffered and streaming
Fish DAC decoding both clamp residual codes to the codec's 1,024-entry residual
codebooks. Returned codebooks retain the tower's original token values.

## Memory management

Drama 3 uses `FishAudioMemoryOptions.balanced` by default. It evaluates codec
intermediates between layers so temporary buffers can be reused or released,
and trims idle MLX allocations above a 512 MiB threshold at stage boundaries.
It leaves live tensors intact. The threshold does not bound allocations made
within a stage. The model also holds the shared
[default MLX cache cap](../inference/mlx-memory.md) for its lifetime.

The codec loads decoding weights first. Encoder, quantizer pre-transformer
and downsampling weights are materialized only on the first reference encoding,
saving 794.4 MiB for text-only requests or precomputed reference codes. Those
weights remain loaded after cloning begins. The bundle's codec file must remain
available for that first encoding. `memoryOptions: .unrestricted` restores eager
loading and disables stage trimming; the lifetime cap still applies. The shared S2 Pro codec loader
retains its previous behavior.

## Validation

The checked-in tiny fixture is generated by
`scripts/fish_drama3_reference.py` from pinned upstream PyTorch code, without
downloading model weights. It covers hybrid-backbone prefill, stepwise cache
continuation, semantic logits, teacher-forced tower state, residual codebook
logits, and feedback into the backbone.
The comparison disables reduced-precision FP32 matrix multiplication with
`MLX_ENABLE_TF32=0`, including on M5 GPUs. Production inference keeps MLX's
default precision policy.

Configuration, prompt, and sampling tests require no model or GPU. Real-model
waveform, voice-cloning, CLI, and independent ASR checks are opt-in; see the
[inference guide](../inference/fish-audio-drama3.md).

On 2026-10-17, the pinned 4-bit checkpoint
(`cc639e27aba7edb73429d3e3806f471dcd79d316`) passed buffered and streamed
text-only, reference-cloning and CLI waveform checks on Apple M5 Pro with
48 GiB unified memory. All 16 release benchmark WAVs passed independent ASR.
The 22.15 GiB bundle's large-file hashes match the pinned upstream revision.

With the default balanced memory policy and three warm calls per mode,
streaming first PCM was 275 ms for text-only and 730 ms for cloning, including
reference encoding. Streaming RTF was 0.726 / 0.893; peak RSS was
15.571 / 14.676 GiB and physical footprint was
22.321 / 23.885 GiB respectively. All 16 WAVs were byte-identical to the
previous runtime. See the
[benchmark report](../benchmarks/fish-audio-drama3-2026-10-17.md) for before/after
memory, latency, allocator behavior and numerical parity. 8-bit and BF16 remain
unvalidated.

Sources: [upstream implementation](https://github.com/fishaudio/fish-speech/tree/541566932abbaebcb0a40e562a1f846738383f38/fish_speech/models/text2semantic/fish_omni),
[MLX conversion](https://github.com/Blaizzy/mlx-audio/tree/d35a8038131f0de33b7ae6da3f2d5b69523b1c3e/mlx_audio/tts/models/fish_drama3).
