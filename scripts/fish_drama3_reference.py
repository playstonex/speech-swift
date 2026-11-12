#!/usr/bin/env python3
"""Create a tiny FP32 parity fixture from the pinned upstream PyTorch implementation.

Requires torch and loguru. No model checkpoints are downloaded.
"""
import argparse
import importlib.util
import json
import re
import sys
import tempfile
import types
import urllib.request
from pathlib import Path

UPSTREAM_REVISION = "541566932abbaebcb0a40e562a1f846738383f38"
PREFIX = "fish_speech.models.text2semantic.fish_omni"


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, help="Also record upstream tokenizer ids from a local bundle")
    args = parser.parse_args()
    torch.set_num_threads(1)
    for name in ["fish_speech", "fish_speech.models", "fish_speech.models.text2semantic", PREFIX]:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    config_module = types.ModuleType(PREFIX + ".config")
    for name in ("FishOmniTextConfig", "AudioDecoderConfig", "FishOmniConfig"):
        setattr(config_module, name, types.SimpleNamespace)
    sys.modules[config_module.__name__] = config_module
    fp8 = types.ModuleType(PREFIX + ".fp8_moe")
    fp8.FP8_BLOCK_SIZE = 128
    fp8.FP8_DTYPE = torch.float8_e4m3fn
    fp8.fp8_moe_forward = lambda *a, **k: (_ for _ in ()).throw(AssertionError("FP8 is unused by this FP32 fixture"))
    sys.modules[fp8.__name__] = fp8
    with tempfile.TemporaryDirectory(prefix="fish-drama3-reference-") as temp:
        modules = {}
        for name in ("layers", "moe", "backbone", "tower"):
            path = Path(temp) / (name + ".py")
            url = f"https://raw.githubusercontent.com/fishaudio/fish-speech/{UPSTREAM_REVISION}/fish_speech/models/text2semantic/fish_omni/{name}.py"
            source = urllib.request.urlopen(url, timeout=30).read()
            # Python 3.9 compatibility affects annotations and constructor syntax only.
            if sys.version_info < (3, 10):
                source = source.replace(b"@dataclass(kw_only=True)", b"@dataclass")
            path.write_bytes(b"from __future__ import annotations\n" + source)
            spec = importlib.util.spec_from_file_location(PREFIX + "." + name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            modules[name] = module
        t = dict(vocab_size=16, hidden_size=8, num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
                 num_attention_heads=2, num_key_value_heads=1, head_dim=8, rope_theta=10000.0, rotary_dim=4,
                 partial_rotary_factor=0.5, rms_norm_eps=1e-6, max_position_embeddings=256,
                 linear_conv_kernel_dim=3, linear_num_key_heads=1, linear_num_value_heads=2,
                 linear_key_head_dim=32, linear_value_head_dim=32, num_experts=3, num_experts_per_tok=2,
                 moe_intermediate_size=4, shared_expert_intermediate_size=4, norm_topk_prob=True)
        a = dict(dim=8, text_dim=8, intermediate_size=12, n_head=2, n_local_heads=1, head_dim=8,
                 rotary_dim=4, rotary_percent=0.5, norm_eps=1e-6, rope_base=10000.0, vocab_size=4,
                 num_codebooks=3, shared_enrichment_layers=1, n_layer=1, vq_backbone_codebooks=1,
                 decoder_type="codebook_mtp_shared", vq_feature_placement="backbone_partial")
        config = types.SimpleNamespace(text=types.SimpleNamespace(**t), audio_decoder=types.SimpleNamespace(**a),
                                       semantic_start_token_id=8, semantic_end_token_id=11, eos_token_id=1)
        backbone = modules["backbone"].FishOmniBackbone(config.text, dtype=torch.float32)
        tower = modules["tower"].FishOmniAudioDecoder(config, dtype=torch.float32)
        shifted = re.compile(r"(input_layernorm|post_attention_layernorm|q_norm|k_norm|enorm|hnorm|final_layernorm|attention_norm|ffn_norm)\.weight$")
        weights = {}
        with torch.no_grad():
            for prefix, model in (("backbone", backbone), ("tower", tower)):
                for index, (name, parameter) in enumerate(sorted(model.named_parameters())):
                    value = torch.sin(torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape) * 0.17 + index) * 0.08
                    is_shifted = bool(shifted.search(name)) or (prefix == "backbone" and name == "model.norm.weight")
                    if name.endswith(".norm.weight") and not is_shifted:
                        value = value + 1
                    parameter.copy_(value)
                    converted = value + 1 if is_shifted else value
                    key = prefix + "." + name
                    if key == "backbone.lm_head.weight":
                        key = "semantic_head.weight"
                        converted = converted[[8, 9, 10, 11, 1]]
                    if key.endswith(".linear_attn.conv1d.weight"):
                        converted = converted.transpose(1, 2)
                    if ".mlp.experts.w13_weight" in key:
                        prefix_key = key.removesuffix(".experts.w13_weight") + ".switch_mlp."
                        for part, tensor in zip(("gate", "up"), converted.chunk(2, dim=1)):
                            weights[prefix_key + part + "_proj.weight"] = {"shape": list(tensor.shape), "values": tensor.reshape(-1).tolist()}
                        continue
                    if ".mlp.experts.w2_weight" in key:
                        key = key.removesuffix(".experts.w2_weight") + ".switch_mlp.down_proj.weight"
                    weights[key] = {"shape": list(converted.shape), "values": converted.reshape(-1).tolist()}
            backbone.setup_caches(1, 64)
            tower.setup_caches(64)
            frames = torch.tensor([[2, 0, 0, 0], [8, 0, 1, 2], [9, 1, 2, 3], [3, 0, 0, 0]], dtype=torch.long)
            positions = torch.arange(4)
            embedding = tower.embed_backbone(frames, backbone.embed(frames[:, 0]))
            trace = {"embedding": embedding.reshape(-1).tolist()}
            handles = []
            for name in ("model.layers.0.input_layernorm", "model.layers.0.linear_attn.in_proj_qkv",
                         "model.layers.0.linear_attn.in_proj_z", "model.layers.0.linear_attn.norm",
                         "model.layers.0.linear_attn", "model.layers.0", "model.layers.1"):
                def record(module, inputs, output, key=name):
                    trace[key] = output.reshape(-1).tolist()
                handles.append(backbone.get_submodule(name).register_forward_hook(record))
            h = backbone(inputs_embeds=embedding[None], input_pos=positions)[0]
            for handle in handles:
                handle.remove()
            delta = backbone.model.layers[0].linear_attn
            projected = torch.tensor(trace["model.layers.0.linear_attn.in_proj_qkv"]).reshape(1, 4, delta.conv_dim).transpose(1, 2)
            padded = torch.nn.functional.pad(projected, (delta.conv_kernel_size - 1, 0))
            trace["convolution"] = torch.nn.functional.silu(torch.nn.functional.conv1d(
                padded, delta.conv1d.weight, groups=delta.conv_dim)).transpose(1, 2).reshape(-1).tolist()
            hidden = h.clone()
            semantic_logits = backbone.compute_logits(h[-1:])[:, [8, 9, 10, 11, 1]].clone()
            tower.prefill(h[:-1], positions[:-1], frames[:-1], frames[1:])
            c0 = 2
            h = tower.run_shared(h[-1:], torch.tensor([c0]), torch.tensor([True]), positions[-1:], frames[-1:])
            codes, code_logits = [c0], []
            for layer, head in zip(tower.residual_layers, tower.codebook_heads):
                h = layer(h, torch.tensor([codes[-1]]), torch.tensor([True]), positions[-1:])
                logits = head(h)
                code_logits.append(logits.reshape(-1).tolist())
                codes.append(int(logits.argmax()))
            feedback = torch.tensor([[8 + c0, *codes]])
            embedding = tower.embed_backbone(feedback, backbone.embed(feedback[:, 0]))
            next_hidden = backbone(inputs_embeds=embedding[None], input_pos=torch.tensor([4]))
        result = {"upstream_revision": UPSTREAM_REVISION,
                  "config": {"model_type": "fish_drama3", "speech_template_version": "reference-bracket-v1",
                             "text_config": {**t, "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5}},
                             "audio_decoder_config": a, "semantic_start_token_id": 8, "semantic_end_token_id": 11, "eos_token_id": 1},
                  "weights": weights, "frames": frames.tolist(), "trace": trace, "hidden": hidden.reshape(-1).tolist(),
                  "semantic_logits": semantic_logits.reshape(-1).tolist(), "code_logits": code_logits,
                  "codes": codes, "next_hidden": next_hidden.reshape(-1).tolist()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
        print(f"Wrote {args.output}: {len(weights)} tensors; no model weights downloaded")
        if args.tokenizer_json:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
            literals = ["<|im_start|>system\n", "convert the provided text to speech", "<|im_end|>\n",
                        "<|im_start|>user\n", "<|speaker:0|> Hello from Drama 3.",
                        "<|im_start|>assistant\n<|voice|>", "[sighs] Fine. [laughs] Maybe.",
                        "<|speaker:15|> नमस्ते दुनिया।", "café e\u0301 👩🏽‍🚀", "你好，世界！", "مرحبا بالعالم"]
            data = [{"text": text, "ids": tokenizer.encode(text, add_special_tokens=False).ids} for text in literals]
            path = args.output.with_name("drama3_tokenizer_reference.json")
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            print(f"Wrote {path}: {len(data)} tokenizer cases")


if __name__ == "__main__":
    main()
