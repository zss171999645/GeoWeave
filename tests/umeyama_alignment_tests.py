import numpy as np

from easyvolcap.utils.metric_utils import align_xyz_umeyama
from easyvolcap.utils.test_utils import my_tests


def _rot_z(theta_deg: float) -> np.ndarray:
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def test_align_xyz_umeyama_recovers_similarity():
    rng = np.random.default_rng(20260320)
    src = rng.normal(size=(64, 3))
    scale = 1.7
    R = _rot_z(35.0)
    t = np.array([0.5, -1.2, 2.3], dtype=np.float64)
    dst = scale * (src @ R.T) + t
    aligned = align_xyz_umeyama(src, dst)
    err = np.max(np.abs(aligned - dst))
    assert err < 1e-5, err


if __name__ == '__main__':
    my_tests(globals())
