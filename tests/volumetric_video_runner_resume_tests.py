import unittest

from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner


class VolumetricVideoRunnerResumeTests(unittest.TestCase):
    def make_runner(self, pretrained_model='ckpt.pt', pretrained_load_training_state=False):
        runner = VolumetricVideoRunner.__new__(VolumetricVideoRunner)
        runner.pretrained_model = pretrained_model
        runner.pretrained_load_training_state = pretrained_load_training_state
        return runner

    def test_weights_only_pretrained_does_not_advance_epoch_when_no_true_resume(self):
        runner = self.make_runner(pretrained_load_training_state=False)
        self.assertEqual(runner._resolve_begin_epoch(pretrained_epoch=31, resume_epoch=0), 0)

    def test_true_resume_keeps_resume_epoch_even_with_weights_only_pretrained(self):
        runner = self.make_runner(pretrained_load_training_state=False)
        self.assertEqual(runner._resolve_begin_epoch(pretrained_epoch=31, resume_epoch=44), 44)

    def test_training_state_pretrained_keeps_checkpoint_epoch_when_no_true_resume(self):
        runner = self.make_runner(pretrained_load_training_state=True)
        self.assertEqual(runner._resolve_begin_epoch(pretrained_epoch=31, resume_epoch=0), 31)


if __name__ == '__main__':
    unittest.main()
