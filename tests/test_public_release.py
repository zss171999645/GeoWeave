import tempfile
import unittest
from pathlib import Path

from geoweave.inference import collect_images, normalize_state_dict


class PublicReleaseTests(unittest.TestCase):
    def test_natural_order_and_case_insensitive_extensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('frame10.JPG', 'frame2.png', 'frame1.jpeg', 'notes.txt'):
                (root / name).touch()
            self.assertEqual([p.name for p in collect_images(root)],
                             ['frame1.jpeg', 'frame2.png', 'frame10.JPG'])

    def test_empty_input_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                collect_images(Path(directory))

    def test_distributed_training_keys_are_normalized(self):
        self.assertEqual(normalize_state_dict({'module.vggt.layer.weight': 1}, 'vggt'),
                         {'layer.weight': 1})

    def test_mixed_prefixes_are_not_silently_discarded(self):
        with self.assertRaises(ValueError):
            normalize_state_dict({'vggt.layer.weight': 1, 'other.weight': 2}, 'vggt')

    def test_special_token_compatibility(self):
        self.assertEqual(normalize_state_dict({'aggregator.camera_token': 1}, 'vggt'),
                         {'aggregator.special_tokens.camera_token': 1})

    def test_key_collision_is_an_error(self):
        with self.assertRaises(ValueError):
            normalize_state_dict({'aggregator.camera_token': 1,
                                  'aggregator.special_tokens.camera_token': 2}, 'vggt')


if __name__ == '__main__':
    unittest.main()
