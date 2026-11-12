The codec fixture executes selected, unchanged classes from the Apache 2.0
[MOSS-Audio-Tokenizer](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer/tree/3cd226ba2947efa357ef453bcad111b6eafba782)
source. It contains generated toy parameters, not pretrained weights. Its JSON
records the pinned source hash, dependency versions and tolerances. The model and
codec configurations are copied from their pinned public upstream checkpoints.

`codec-tiny.json` covers exact GELU, nonuniform residual scales, adjacent-pair
rotary positions at a nonzero offset, a strict causal context window, single-token
and multi-token cached inference, patch channel ordering, and signed weight
normalization. The diagnostic upstream multi-token ring result differs from the
offline reference because old keys are overwritten before early queries use them.
The native runtime must match the offline and concatenated-cache references.

`qwen-tiny.json` uses Transformers Qwen3Model and generated toy parameters for
sixteen additional learned audio embeddings and seventeen independent heads.
It includes Float32 and BF16 weights, full and cached outputs, and all-padding
audio positions whose embeddings must still contribute to the language input.
