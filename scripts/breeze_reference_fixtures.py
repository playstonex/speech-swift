#!/usr/bin/env python3
"""Generate numeric fixtures with the official CPU PyTorch implementation.

Use a checkout of https://github.com/breezeblue-ai/breeze-tts and install its
torch/transformers dependencies. Run separately from Swift model tests.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tiny", action="store_true", help="Small deterministic CPU text-encoder fixture, no checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream.resolve()))
    import torch
    torch.set_num_threads(4)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.upstream, text=True).strip()
    if args.tiny:
        from models.t5gemma2_compat import T5Gemma2TextConfig, T5Gemma2TextEncoder
        config = {"vocab_size":32,"hidden_size":16,"intermediate_size":32,"num_hidden_layers":6,
                  "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"query_pre_attn_scalar":8,
                  "sliding_window":4,"eoi_token_index":31,
                  "layer_types":["sliding_attention"] * 5 + ["full_attention"],
                  "rope_parameters":{"sliding_attention":{"rope_type":"default","rope_theta":10000},
                                     "full_attention":{"rope_type":"linear","rope_theta":1000000,"factor":8}}}
        c = T5Gemma2TextConfig(**config)
        c._attn_implementation = "eager"
        model = T5Gemma2TextEncoder(c).eval()
        weights = {}
        for ordinal, (name, value) in enumerate(sorted(model.state_dict().items())):
            weights[name] = ((torch.arange(value.numel()) + ordinal * 3) % 17 - 8).float().reshape(value.shape) / 64
        model.load_state_dict(weights, strict=True)
        ids = [2, 3, 31, 4, 7, 8, 10]
        with torch.inference_mode():
            result = model(input_ids=torch.tensor([ids])).last_hidden_state[0].tolist()
        args.output.write_text(json.dumps({"upstream_revision":revision,"config":config,"ids":ids,
            "parameter_names":sorted(weights),"expected":result}, indent=2) + "\n")
        print(f"Saved {args.output}")
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --tiny is used")
    from transformers import AutoTokenizer
    from models.breeze import BreezeForConditionalGeneration

    model = BreezeForConditionalGeneration.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, fix_mistral_regex=False)
    text = "[S0]<ins_bos>A clear, calm English narrator.<ins_eos>The quick brown fox jumps over the lazy dog."
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        encoded = model.text_encoder(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
        projected = model.text_encoder_proj(encoded)
        hidden = model.backbone_model(inputs_embeds=projected, use_cache=False).last_hidden_state
        logits = model.lm_head(hidden[:, -1])
        first = int(logits[0, :2048].argmax())
        depth = model.depth_decoder(input_ids=torch.tensor([[0, first]]),
            backbone_last_hidden_state=hidden[:, -1], use_cache=False).logits
    result = {"upstream_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.upstream, text=True).strip(),
              "text": text, "ids": ids[0].tolist(),
              "text_features": encoded[0, :, :8].float().tolist(),
              "backbone_hidden": hidden[0, -1].float().tolist(),
              "backbone_logits": logits[0].float().tolist(),
              "first_codebook": first, "depth_logits": depth[0, -1].float().tolist()}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
