"""Exercise profiler preflight and manifest recovery without Swift or models."""
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


class BreezeProfileBuildTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        scripts = root / "scripts/profiling"
        scripts.mkdir(parents=True)
        builder = scripts / "build_breeze_profile.py"
        shutil.copy2(Path(__file__).with_name(builder.name), builder)
        builder.with_name("BreezeTTSBenchmark.swift").write_text("// fixture\n")
        support = root / "Sources/AsrBenchmark"
        support.mkdir(parents=True)
        for name in ("Normalizer.swift", "WER.swift", "Memory.swift"):
            (support / name).write_text("// fixture\n")
        manifest = root / "Package.swift"
        manifest.write_bytes(b"// original manifest\n")
        return root, builder, manifest

    def testActiveJobsBlockBeforeWritingOrBuilding(self):
        jobs = ["/usr/bin/swift-test", "/usr/bin/xcodebuild",
                "/tmp/worktree/.build/release/native-candidate-probe"]
        for job in jobs:
            with self.subTest(job=job), tempfile.TemporaryDirectory() as directory:
                root, builder, manifest = self.fixture(directory)
                with patch("subprocess.check_output", return_value=job + "\n"), \
                     patch("subprocess.run") as build:
                    with self.assertRaisesRegex(RuntimeError, "Finish other"):
                        runpy.run_path(str(builder))
                    build.assert_not_called()
                self.assertEqual(manifest.read_bytes(), b"// original manifest\n")
                self.assertFalse((root / ".build").exists())

    def testBuildFailureRestoresOriginalManifest(self):
        with tempfile.TemporaryDirectory() as directory:
            _, builder, manifest = self.fixture(directory)
            failure = subprocess.CalledProcessError(1, ["swift", "build"])
            with patch("subprocess.check_output", return_value="/usr/bin/python3\n"), \
                 patch("subprocess.run", side_effect=failure) as build:
                with self.assertRaises(subprocess.CalledProcessError):
                    runpy.run_path(str(builder))
                build.assert_called_once()
            self.assertEqual(manifest.read_bytes(), b"// original manifest\n")


if __name__ == "__main__":
    unittest.main()
