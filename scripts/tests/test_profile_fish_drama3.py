import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "profiling"))
import profile_fish_drama3 as profile


class FishDrama3ProfileTests(unittest.TestCase):
    def fixture(self):
        rows = [{"index": i, "phase": "first" if i == 0 else "warm", "audio_seconds": 4,
                 "generation_seconds": seconds, "rtf": 999} for i, seconds in enumerate([8, 2, 4])]
        data = {"mode": "text-only", "load_seconds": 3, "warm_repeats": 2, "rows": rows,
                "reference_encoding_in_generation": False,
                "baseline_memory": {"physical_bytes": 20}, "loaded_memory": {"physical_bytes": 300},
                "final_memory": {"peak_rss_bytes": 200, "peak_physical_bytes": 400}}
        verification = [{"id": f"text-only-{i}", "passed": True} for i in range(3)]
        return data, verification

    def test_reports_warm_median_and_wall_time_over_audio_rtf(self):
        data, verification = self.fixture()
        result = profile.summarize(data, verification)
        self.assertEqual(result["warm_synthesis_median_seconds"], 3)
        self.assertEqual(result["warm_rtf_median"], 0.75)
        self.assertEqual(result["first_synthesis_seconds"], 8)
        self.assertEqual(result["peak_rss_bytes"], 200)
        self.assertEqual(result["peak_physical_bytes"], 400)

    def test_rejects_missing_or_failed_asr_checks(self):
        data, verification = self.fixture()
        with self.assertRaises(ValueError):
            profile.summarize(data, verification[:-1])
        verification[1]["passed"] = False
        with self.assertRaises(ValueError):
            profile.summarize(data, verification)

    def test_streaming_reports_first_pcm_and_rejects_invalid_latencies(self):
        data, verification = self.fixture()
        data["delivery"] = "streaming"
        for row, latency in zip(data["rows"], [1.5, 0.2, 0.4]):
            row.update(first_audio_seconds=latency, chunk_count=5)
        result = profile.summarize(data, verification)
        self.assertAlmostEqual(result["warm_first_audio_median_seconds"], 0.3)
        self.assertEqual(result["first_audio_seconds"], 1.5)
        for latency in [0, float("nan"), float("inf"), 100]:
            data["rows"][1]["first_audio_seconds"] = latency
            with self.assertRaises(ValueError):
                profile.summarize(data, verification)

    def test_rejects_missing_repeats_and_nonfinite_timings(self):
        data, verification = self.fixture()
        data["warm_repeats"] = 3
        with self.assertRaises(ValueError):
            profile.summarize(data, verification)
        data["warm_repeats"] = 2
        for value in [0, -1, float("nan"), float("inf")]:
            damaged = copy.deepcopy(data)
            damaged["rows"][1]["generation_seconds"] = value
            with self.assertRaises(ValueError):
                profile.summarize(damaged, verification)

    def test_validates_export_hashes_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"checkpoint bytes"
            payload = root / "model.safetensors"
            payload.write_bytes(content)
            manifest = {"files": {payload.name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}}}
            (root / "soniqo_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(profile.validate_bundle(root), manifest)
            payload.write_bytes(b"damaged payload!")
            with self.assertRaises(ValueError):
                profile.validate_bundle(root)

    def test_ignores_owned_process_tree_but_detects_other_models(self):
        output = "10 1 /bin/fish-drama3-profile\n11 10 /bin/metal\n12 11 /bin/metallib\n20 1 /bin/xctest\n"
        with patch.object(profile.subprocess, "check_output", return_value=output):
            self.assertEqual(profile.active_jobs(ignore_pid=10), [(20, "xctest")])


if __name__ == "__main__":
    unittest.main()
