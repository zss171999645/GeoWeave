from dataclasses import dataclass


@dataclass
class GameConfig:
    type: str
    near: float
    far: float
    inverse: bool = False
    fov_type: str = 'vfov'
    delta: int = 0
    scale: float = 1.0
    max_allowd_diff_pose_timestamps: int = 1


game_cfgs = {
    '2077': GameConfig(
        type='2077',
        near=0.02,
        far=10000, # 1e4
        delta=-2,
        max_allowd_diff_pose_timestamps=2,
    ),
    'wukong': GameConfig(
        type='wukong',
        near=0.05,
        far=1e5,
        delta=-1,
        scale=0.005,
        fov_type='hfov',
        max_allowd_diff_pose_timestamps=2,
    ),
    'rdr2': GameConfig(
        type='rdr2',
        near=0.05,
        far=10000, # 1e4
        delta=-2,
        max_allowd_diff_pose_timestamps=8,
    ),
    'default': GameConfig(
        type='default',
        near=0.02,
        far=1e9,
        delta=0,
        max_allowd_diff_pose_timestamps=1,
    ),
}
