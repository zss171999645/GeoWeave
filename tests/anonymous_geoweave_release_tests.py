import importlib
import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "release" / "anonymous_geoweave_model"

FORBIDDEN_PATTERNS = [
    r"feng01",
    r"horizon",
    r"saturn",
    r"easyvolcap",
    r"aidi",
    r"meshx",
    r"bucket",
    r"tao02",
    r"yingfeng",
    r"runsong",
    r"zinan",
    r"qingfeng",
    r"/Users/",
    r"/home/users/",
    r"/horizon-",
    r"dev-5090",
    r"dev-4090",
    r"checkpoint",
    r"trained_model",
    r"record/",
    r"result/",
    r"cluster",
    r"queue",
]


def iter_release_text_files():
    assert RELEASE_DIR.exists()
    for path in RELEASE_DIR.rglob("*"):
        if path.is_file():
            yield path


class AnonymousGeoWeaveReleaseTests(unittest.TestCase):
    def test_release_contains_two_anonymous_model_entries(self):
        self.assertTrue((RELEASE_DIR / "geoweave_model" / "models" / "geoweave_pi3.py").exists())
        self.assertTrue((RELEASE_DIR / "geoweave_model" / "models" / "geoweave_vggt.py").exists())

    def test_release_imports_and_runs_tiny_forward(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed in this local Python environment")

        sys.path.insert(0, str(RELEASE_DIR))
        try:
            pkg = importlib.import_module("geoweave_model")
            for factory_name in ("build_geoweave_pi3", "build_geoweave_vggt"):
                model = getattr(pkg, factory_name)(tiny=True)
                model.eval()
                images = torch.rand(1, 2, 3, 32, 32)
                with torch.no_grad():
                    output = model(images)
                self.assertEqual(tuple(output["world_points"].shape), (1, 2, 32, 32, 3))
                self.assertEqual(tuple(output["confidence"].shape), (1, 2, 32, 32, 1))
                self.assertEqual(tuple(output["camera_poses"].shape), (1, 2, 4, 4))
        finally:
            sys.path.remove(str(RELEASE_DIR))

    def test_release_has_no_internal_identity_leaks(self):
        compiled = [re.compile(pattern, re.IGNORECASE) for pattern in FORBIDDEN_PATTERNS]
        leaks = []
        for path in iter_release_text_files():
            rel = path.relative_to(RELEASE_DIR).as_posix()
            for regex in compiled:
                if regex.search(rel):
                    leaks.append(f"{rel}:<filename>:{regex.pattern}")
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                for regex in compiled:
                    if regex.search(line):
                        leaks.append(f"{rel}:{line_no}:{regex.pattern}")
        self.assertFalse(leaks, "\n".join(leaks))


if __name__ == "__main__":
    unittest.main()
