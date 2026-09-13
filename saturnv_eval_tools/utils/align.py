
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from sklearn.covariance import MinCovDet


def compute_robust_weights(residuals):
    """基于马氏距离的自动权重计算"""
    # 使用最小协方差行列式估计器
    robust_cov = MinCovDet().fit(residuals)
    mahalanobis = robust_cov.mahalanobis(residuals)
    
    # 计算权重 (Tukey biweight函数)
    c = 4.685  # 95%效率的调谐常数
    weights = (1 - np.minimum(mahalanobis / c**2, 1))**2
    return weights

def robust_pose_alignment(src_poses, tgt_poses, max_iter=20, huber_threshold=0.5):
    """
    全位姿鲁棒对齐方法，保证缩放因子始终为正
    改进点：
    1. 参数重映射确保scale>0
    2. 添加优化变量边界约束
    3. 更合理的初始缩放估计
    """
    # 检查数据有效性
    assert len(src_poses) >= 2 and len(tgt_poses) >= 2, "至少需要2帧数据"
    
    # 提取首尾帧 ----------------------------------------------------------------
    src_first = src_poses[0][1:4]  # [tx, ty, tz]
    src_last  = src_poses[-1][1:4]
    tgt_first = tgt_poses[0][1:4]
    tgt_last  = tgt_poses[-1][1:4]

    # 计算初始缩放因子（基于首尾帧欧氏距离）---------------------------------------
    def safe_scale(numerator, denominator):
        eps = 1e-6
        ratio = numerator / (denominator + eps)
        return np.clip(ratio, 0.1, 10.0)  # 限制在合理范围

    # 计算源和目标轨迹长度
    src_dist = np.linalg.norm(src_last - src_first)
    tgt_dist = np.linalg.norm(tgt_last - tgt_first)
    initial_scale = safe_scale(tgt_dist, src_dist)

    # 改进旋转初始化（利用首尾帧方向）-------------------------------------------
    # 计算源轨迹方向向量
    src_dir = (src_last - src_first) / (src_dist + 1e-6)
    tgt_dir = (tgt_last - tgt_first) / (tgt_dist + 1e-6)
    
    # 构建方向对齐旋转矩阵
    cross = np.cross(src_dir, tgt_dir)
    dot = np.dot(src_dir, tgt_dir)
    R_dir = R.from_rotvec(cross * np.arccos(dot)).as_matrix()

    # 结合首帧旋转（加权平均）-------------------------------------------------
    # 提取首帧旋转
    q_src = np.array([src_poses[0][4], src_poses[0][5], src_poses[0][6], src_poses[0][7]])  # (qx, qy, qz, qw)
    q_tgt = np.array([tgt_poses[0][4], tgt_poses[0][5], tgt_poses[0][6], tgt_poses[0][7]])
    R_src = R.from_quat(q_src).as_matrix()
    R_tgt = R.from_quat(q_tgt).as_matrix()
    R_rel_frame0 = R_tgt @ R_src.T

    # 融合旋转（方向对齐占70%权重，首帧旋转占30%）
    R_rel = 0.0 * R_dir + 1.0 * R_rel_frame0
    U, _, Vt = np.linalg.svd(R_rel)  # 正交化
    R_rel = U @ Vt

    # 计算初始平移 ------------------------------------------------------------
    t_rel = tgt_first - initial_scale * R_rel @ src_first

    # 参数初始化 -------------------------------------------------------------
    rotvec = R.from_matrix(R_rel).as_rotvec()
    params_init = np.concatenate([[np.log(initial_scale)], rotvec, t_rel])



    # 设置优化边界（防止旋转向量过大）
    bounds = (
        [-np.inf] + [-np.pi]*3 + [-np.inf]*3,  # 下限：log_scale无限制，旋转向量限制在±180°
        [np.inf] + [np.pi]*3 + [np.inf]*3
    )

    # 多阶段优化
    current_weights = np.ones(len(src_poses))
    for stage in range(max_iter):
        res = least_squares(
            lambda p: modified_weighted_cost(p, src_poses, tgt_poses, current_weights),
            params_init,
            bounds=bounds,  # 新增边界约束
            loss='huber',
            f_scale=huber_threshold,
            method='trf'
        )
        
        # 更新权重和收敛检查
        residuals = compute_residuals_exp(res.x, src_poses, tgt_poses)
        current_weights = compute_robust_weights(residuals)
        if np.linalg.norm(res.x - params_init) < 1e-6:
            break
        params_init = res.x

    # 解析参数（应用指数函数确保scale>0）
    log_scale = res.x[0]
    scale = np.exp(log_scale)  # 保证scale正数
    rot_vec = res.x[1:4]
    t = res.x[4:7]
    final_R = R.from_rotvec(rot_vec).as_matrix()
    
    return final_R, t, scale, residuals

def modified_weighted_cost(params, src_poses, tgt_poses, weights):
    """修改后的代价函数，处理对数缩放因子"""
    log_scale = params[0]
    scale = np.exp(log_scale)  # 映射到正数域
    R_mat = R.from_rotvec(params[1:4]).as_matrix()
    t_vec = params[4:7]
    
    errors = []
    for w, src, tgt in zip(weights, src_poses, tgt_poses):
        # 平移项（应用指数缩放）
        t_pred = scale * R_mat @ src[1:4] + t_vec
        t_error = (t_pred - tgt[1:4]) * w**0.5
        
        # 旋转项（添加缩放因子正则项）
        R_src = R.from_quat(src[4:]).as_matrix()
        R_tgt = R.from_quat(tgt[4:]).as_matrix()
        R_error = (R_mat @ R_src @ R_tgt.T - np.eye(3)).flatten() * w**0.5
        
        # 添加缩放因子正则项（防止指数爆炸）
        scale_reg = 0.01 * log_scale  # 限制缩放因子在合理范围
        
        errors.extend(t_error.tolist())
        errors.extend(R_error.tolist())
        errors.append(scale_reg)
        
    return np.array(errors)

def compute_residuals_exp(params, src_poses, tgt_poses):
    """处理对数缩放因子的残差计算"""
    scale = np.exp(params[0])
    R_mat = R.from_rotvec(params[1:4]).as_matrix()
    t_vec = params[4:7]
    
    residuals = []
    for src, tgt in zip(src_poses, tgt_poses):
        # 平移残差
        t_pred = scale * R_mat @ src[1:4] + t_vec
        t_error = np.linalg.norm(t_pred - tgt[1:4])
        
        # 旋转残差（考虑缩放因子影响）
        R_src = R.from_quat(src[4:]).as_matrix()
        R_tgt = R.from_quat(tgt[4:]).as_matrix()
        angle_error = R.from_matrix(R_mat @ R_src @ R_tgt.T).magnitude()
        if angle_error > np.pi:
            angle_error = 2 * np.pi - angle_error
        
        residuals.append(np.array([t_error, angle_error]))
    return np.array(residuals)