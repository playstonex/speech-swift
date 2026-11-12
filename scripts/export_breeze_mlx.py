#!/usr/bin/env python3
"""Export a pinned Breeze TTS 2 checkpoint for the native BreezeTTS runtime.

Exports preserve the upstream model license and source provenance. The
original Qwen3 audio tokenizer stays unmodified.
Dependencies: huggingface_hub, mlx, numpy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

SOURCE_MODEL = "BreezeBlue/Breeze-TTS-2"
SOURCE_REVISION = "3e28c5151381a722f1d8661b4118c298caa77aa4"
DERIVATION_NOTICE = "Derived from Breeze TTS 2 by BreezeBlue and licensed for research and non-commercial use only."
COPYRIGHT_NOTICE = ("Breeze TTS 2 is licensed under the BreezeBlue Research and Non-Commercial\n"
                    "License Agreement. Copyright (c) 2026 RESONIA, INC. All Rights Reserved.")


def keep_tensor(name: str) -> bool:
    return not (name.startswith(("codec_model.", "embed_text_tokens."))
                or name.endswith(".initialized") or "rotary_emb.inv_freq" in name
                or name == "backbone_model.embed_tokens.embed_audio_tokens.weight")


def quantize_tensor(name: str, shape: tuple[int, ...]) -> bool:
    if len(shape) != 2 or not name.endswith(".weight"):
        return False
    module = name.rsplit(".", 1)[0].rsplit(".", 1)[-1]
    return module in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                      "lm_head", "text_encoder_proj", "inputs_embeds_projector",
                      "audio_embeds_projector", "backbone_hidden_state_projector"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(source: Path, output: Path, bits: int, source_revision: str = SOURCE_REVISION) -> dict:
    import mlx.core as mx

    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((source / "config.json").read_text())
    if original.get("model_type") != "breeze" or original.get("audio_num_codebooks") != 16:
        raise ValueError("Expected the Breeze TTS 2 16-codebook checkpoint")
    if not (source / "LICENSE").is_file():
        raise ValueError("The upstream model LICENSE is required")
    config = dict(original)
    if bits:
        config["quantization"] = {"bits": bits, "group_size": 64}
    else:
        config.pop("quantization", None)
    repo_id = f"aufklarer/Breeze-TTS-2-MLX-{'bf16' if not bits else str(bits) + 'bit'}"
    index = source / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    weights = {}
    quantized_names = []
    probe = None
    skipped = []
    for shard in shards:
        print(f"Reading {shard}", flush=True)
        for name, value in mx.load(str(source / shard)).items():
            if not keep_tensor(name):
                skipped.append(name)
                continue
            # Retain BF16 embeddings and depth codebook heads; quantize every Linear.
            if bits and quantize_tensor(name, value.shape):
                if value.shape[-1] % 64:
                    raise ValueError(f"Weight is not divisible by group size 64: {name}")
                packed, scales, biases = mx.quantize(value, group_size=64, bits=bits)
                mx.eval(packed, scales, biases)
                prefix = name.removesuffix(".weight")
                weights[name] = packed
                weights[prefix + ".scales"] = scales
                weights[prefix + ".biases"] = biases
                quantized_names.append(prefix)
                if probe is None:
                    restored = mx.dequantize(packed, scales, biases, group_size=64, bits=bits)
                    error = restored.astype(mx.float32) - value.astype(mx.float32)
                    probe = {"tensor": name, "rmse": float(mx.sqrt(mx.mean(error * error)).item()),
                             "max_absolute_error": float(mx.max(mx.abs(error)).item())}
            else:
                weights[name] = value
                mx.eval(value)
        mx.clear_cache()
    if "depth_decoder.model.embed_tokens.weight" not in weights:
        raise ValueError("Missing shared codebook embeddings")
    print(f"Saving {len(weights)} tensors ({len(quantized_names)} quantized Linear layers)", flush=True)
    mx.save_safetensors(str(output / "model.safetensors"), weights)
    del weights
    mx.clear_cache()
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "LICENSE"]:
        shutil.copy2(source / name, output / name)
    shutil.copytree(source / "audio_tokenizer", output / "audio_tokenizer")
    codec_license = Path(__file__).resolve().parents[1] / "ThirdPartyNotices/Qwen3TTSCodec.txt"
    shutil.copy2(codec_license, output / "audio_tokenizer/LICENSE")
    (output / "source_config.json").write_text(json.dumps(original, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "NOTICE").write_text(COPYRIGHT_NOTICE + "\n\n" + DERIVATION_NOTICE + "\n\n"
        + "Modifications: native MLX safetensors export; optional affine group-64 quantization\n"
        + "of Linear weights; unused legacy codec and text embeddings removed; tied audio\n"
        + "embeddings stored once. Text tokenizer and Qwen3 audio tokenizer are unchanged.\n")
    metadata = {"source_model": SOURCE_MODEL, "source_revision": source_revision,
                "format": "speech-swift-breeze-v1", "bits": bits or 16, "group_size": 64 if bits else None,
                "quantized_layers": quantized_names, "removed_tensors": sorted(skipped), "quantization_probe": probe,
                "files": {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                          for p in sorted(output.rglob("*")) if p.is_file()}}
    (output / "export_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    precision_description = f"Affine {bits}-bit Linear weights in groups of 64" if bits else "Unquantized BF16 Linear weights"
    (output / "README.md").write_text(f'''---
license: other
license_name: breezeblue-research-and-non-commercial-license
license_link: LICENSE
base_model: {SOURCE_MODEL}
library_name: mlx
pipeline_tag: text-to-speech
language: [en, zh]
tags: [mlx, speech-swift, text-to-speech, voice-cloning, voice-design]
---
# Breeze TTS 2 ({'BF16' if not bits else str(bits) + '-bit MLX'})

{DERIVATION_NOTICE}

Native MLX checkpoint for [speech-swift](https://github.com/soniqo/speech-swift).
English and Chinese synthesis, natural-language voice design, reference-guided voice
cloning and direction, mono 24 kHz PCM, and incremental audio chunks.

Source: [{SOURCE_MODEL}](https://huggingface.co/{SOURCE_MODEL}/tree/{source_revision}).
The model and self-hosted outputs are restricted to research and non-commercial use.
See [LICENSE](LICENSE) and [NOTICE](NOTICE). This is an independently converted checkpoint.

## Conversion

{precision_description}; embeddings, normalization weights,
and depth codebook heads retain upstream precision. The Qwen3 audio tokenizer is copied
unchanged. Unused legacy codec and text embeddings are removed; tied audio embeddings
are stored once. Immutable provenance, file hashes, and conversion details are in
`export_config.json`. No inference quality or performance claim follows from export alone.

## Swift

```swift
import BreezeTTS

let model = try await BreezeTTSModel.fromPretrained(modelId: "{repo_id}")
let samples = try await model.generate(
    text: "The quick brown fox jumps over the lazy dog.",
    instruction: "A clear, calm English narrator.")
```

## CLI

```bash
speech speak "The quick brown fox jumps over the lazy dog." --engine breeze \\
  --breeze-model-id {repo_id} --instruct "A clear English narrator" --stream -o speech.wav
```

Voice cloning requires `--voice-sample reference.wav --breeze-reference-text "Exact transcript"`.
Add `--instruct` to direct the reference voice. English and Chinese are inferred from text.
Use one generation at a time per model. Reaching the frame cap before EOS raises an error.
''')
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bits", type=int, choices=[0, 4, 8], default=8, help="0 retains upstream BF16")
    parser.add_argument("--upload", help="Upload the completed export to this Hugging Face model repository")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.source_revision):
        parser.error("Use an immutable 40-character source revision")
    source = args.source_dir
    if source is None:
        from huggingface_hub import snapshot_download
        source = Path(snapshot_download(SOURCE_MODEL, revision=args.source_revision,
            allow_patterns=["*.json", "*.safetensors", "LICENSE", "audio_tokenizer/*"]))
    result = export(source, args.output, args.bits, args.source_revision)
    print(json.dumps({"output": str(args.output), "bits": result["bits"],
                      "quantized_layers": len(result["quantized_layers"]), "probe": result["quantization_probe"]}))
    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.upload, exist_ok=True)
        info = api.upload_folder(repo_id=args.upload, folder_path=args.output,
                                 commit_message="Add native MLX conversion and source provenance")
        print(info)


if __name__ == "__main__":
    main()
