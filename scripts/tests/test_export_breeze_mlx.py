import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("export_breeze", Path(__file__).parents[1] / "export_breeze_mlx.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class BreezeExportTests(unittest.TestCase):
    def test_removes_only_unused_and_duplicate_tensors(self):
        for name in ["codec_model.encoder.weight", "embed_text_tokens.weight",
                     "backbone_model.embed_tokens.embed_audio_tokens.weight", "codec.initialized"]:
            self.assertFalse(exporter.keep_tensor(name))
        for name in ["depth_decoder.model.embed_tokens.weight", "text_encoder.embed_tokens.eoi_embedding",
                     "lm_head.weight", "depth_decoder.codebooks_head.weight"]:
            self.assertTrue(exporter.keep_tensor(name))

    def test_quantizes_linear_weights_and_keeps_embeddings_and_heads(self):
        for name in ["backbone_model.layers.0.self_attn.q_proj.weight", "text_encoder_proj.weight",
                     "depth_decoder.model.inputs_embeds_projector.weight", "lm_head.weight"]:
            self.assertTrue(exporter.quantize_tensor(name, (2048, 2048)))
        for name, shape in [("depth_decoder.model.embed_tokens.weight", (32816, 2048)),
                            ("text_encoder.embed_tokens.weight", (262158, 1152)),
                            ("depth_decoder.codebooks_head.weight", (15, 1024, 2051)),
                            ("text_encoder.norm.weight", (1152,))]:
            self.assertFalse(exporter.quantize_tensor(name, shape))

    def test_source_is_pinned_and_license_notice_is_preserved(self):
        self.assertRegex(exporter.SOURCE_REVISION, r"^[a-f0-9]{40}$")
        self.assertIn("research and non-commercial use only", exporter.DERIVATION_NOTICE)
        self.assertIn("2026 RESONIA, INC.", exporter.COPYRIGHT_NOTICE)


if __name__ == "__main__":
    unittest.main()
