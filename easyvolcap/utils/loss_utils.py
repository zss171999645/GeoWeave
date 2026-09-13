import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vgg
from collections import namedtuple
from typing import Callable, Tuple, List, Literal

from math import exp, ceil, floor
from torch.autograd import Variable
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures

from easyvolcap.utils.prop_utils import searchsorted, matchup_channels
from easyvolcap.utils.console_utils import *

from enum import Enum, auto


class ElasticLossReduceType(Enum):
    WEIGHT = auto()
    MEDIAN = auto()


class ImgLossType(Enum):
    PERC = auto()  # lpips
    CHARB = auto()
    HUBER = auto()
    L1 = auto()
    L2 = auto()
    SSIM = auto()
    MSSSIM = auto()
    WL1 = auto()


class DptLossType(Enum):
    SMOOTHL1 = auto()
    WEIGHTL1 = auto()
    WEIGHTL2 = auto()
    L1 = auto()
    L2 = auto()
    REG = auto()
    WEIGHTREG = auto()
    SSIMSE = auto()
    SSIMAE = auto()
    SILOG = auto()
    CONTINUITY = auto()
    RANKING = auto()


class XyzLossType(Enum):
    SMOOTHL1 = auto()
    WEIGHTL1 = auto()
    WEIGHTL2 = auto()
    L1 = auto()
    L2 = auto()
    REG = auto()
    WEIGHTREG = auto()
    SSIMSE = auto()
    SSIMAE = auto()
    ROEMAE = auto()
    LOCALROEMAE = auto()


class PoseLossType(Enum):
    HUBER = auto()
    L1 = auto()
    L2 = auto()
    L21 = auto()
    L1_CONF = auto()
    L21_CONF = auto()

    def has_confidence(self):
        return self in [PoseLossType.L1_CONF, PoseLossType.L21_CONF]


def compute_val_pair_around_range(pts: torch.Tensor, decoder: Callable[[torch.Tensor], torch.Tensor], diff_range: float):
    # sample around input point and compute values
    # pts and its random neighbor are concatenated in second dimension
    # if needed, decoder should return multiple values together to save computation
    neighbor = pts + (torch.rand_like(pts) - 0.5) * diff_range
    full_pts = torch.cat([pts, neighbor], dim=-2)  # cat in n_masked dim
    raw: torch.Tensor = decoder(full_pts)  # (n_batch, n_masked, 3)
    return raw

# from mipnerf360


def inner_outer(t0, t1, y1):
    """Construct inner and outer measures on (t1, y1) for t0."""
    cy1 = torch.cat([torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1)  # 129
    idx_lo, idx_hi = searchsorted(t1, t0)

    cy1_lo = torch.take_along_dim(cy1, idx_lo, dim=-1)  # 128
    cy1_hi = torch.take_along_dim(cy1, idx_hi, dim=-1)

    y0_outer = cy1_hi[..., 1:] - cy1_lo[..., :-1]  # 127
    y0_inner = torch.where(idx_hi[..., :-1] <= idx_lo[..., 1:], cy1_lo[..., 1:] - cy1_hi[..., :-1], 0)
    return y0_inner, y0_outer

# from mipnerf360


def lossfun_outer(t: torch.Tensor, w: torch.Tensor, t_env: torch.Tensor, w_env: torch.Tensor, eps=torch.finfo(torch.float32).eps):
    # accepts t.shape[-1] = w.shape[-1] + 1
    t, w = matchup_channels(t, w)
    t_env, w_env = matchup_channels(t_env, w_env)
    """The proposal weight should be an upper envelope on the nerf weight."""
    _, w_outer = inner_outer(t, t_env, w_env)
    # We assume w_inner <= w <= w_outer. We don't penalize w_inner because it's
    # more effective to pull w_outer up than it is to push w_inner down.
    # Scaled half-quadratic loss that gives a constant gradient at w_outer = 0.
    return (w - w_outer).clip(0.).pow(2) / (w + eps)


def blur_stepfun(x, y, r):
    xr, xr_idx = torch.sort(torch.cat([x - r, x + r], dim=-1))
    y1 = (torch.cat([y, torch.zeros_like(y[..., :1])], dim=-1) -
          torch.cat([torch.zeros_like(y[..., :1]), y], dim=-1)) / (2 * r)
    y2 = torch.cat([y1, -y1], dim=-1).take_along_dim(xr_idx[..., :-1], dim=-1)
    yr = torch.cumsum((xr[..., 1:] - xr[..., :-1]) *
                      torch.cumsum(y2, dim=-1), dim=-1).clamp_min(0)
    yr = torch.cat([torch.zeros_like(yr[..., :1]), yr], dim=-1)
    return xr, yr


def sorted_interp_quad(x, xp, fpdf, fcdf):
    """interp in quadratic"""

    # Identify the location in `xp` that corresponds to each `x`.
    # The final `True` index in `mask` is the start of the matching interval.
    mask = x[..., None, :] >= xp[..., :, None]

    def find_interval(x, return_idx=False):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0, x0_idx = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
        x1, x1_idx = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)
        if return_idx:
            return x0, x1, x0_idx, x1_idx
        return x0, x1

    fcdf0, fcdf1, fcdf0_idx, fcdf1_idx = find_interval(fcdf, return_idx=True)
    fpdf0 = fpdf.take_along_dim(fcdf0_idx, dim=-1)
    fpdf1 = fpdf.take_along_dim(fcdf1_idx, dim=-1)
    xp0, xp1 = find_interval(xp)

    offset = torch.clip(torch.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
    ret = fcdf0 + (x - xp0) * (fpdf0 + fpdf1 * offset + fpdf0 * (1 - offset)) / 2
    return ret


def lossfun_zip_outer(t, w, t_env, w_env, pulse_width, eps=1e-6):
    t, w = matchup_channels(t, w)
    t_env, w_env = matchup_channels(t_env, w_env)

    w_normalize = w / torch.clamp_min(t[..., 1:] - t[..., :-1], eps)

    t_, w_ = blur_stepfun(t, w_normalize, pulse_width)
    w_ = torch.clip(w_, min=0.)
    assert (w_ >= 0.0).all()

    # piecewise linear pdf to piecewise quadratic cdf
    area = 0.5 * (w_[..., 1:] + w_[..., :-1]) * (t_[..., 1:] - t_[..., :-1])

    cdf = torch.cat([torch.zeros_like(area[..., :1]), torch.cumsum(area, dim=-1)], dim=-1)

    # query piecewise quadratic interpolation
    cdf_interp = sorted_interp_quad(t_env, t_, w_, cdf)
    # difference between adjacent interpolated values
    w_s = torch.diff(cdf_interp, dim=-1)

    return ((w_s - w_env).clip(0.).pow(2) / (w_env + eps)).mean()


def lossfun_distortion(t: torch.Tensor, w: torch.Tensor):
    # accepts t.shape[-1] = w.shape[-1] + 1
    t, w = matchup_channels(t, w)
    """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
    # The loss incurred between all pairs of intervals.
    ut = (t[..., 1:] + t[..., :-1]) / 2  # 64
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])  # 64
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    # The loss incurred within each individual interval with itself.
    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return loss_inter + loss_intra


def interval_distortion(t0_lo, t0_hi, t1_lo, t1_hi):
    """Compute mean(abs(x-y); x in [t0_lo, t0_hi], y in [t1_lo, t1_hi])."""
    # Distortion when the intervals do not overlap.
    d_disjoint = torch.abs((t1_lo + t1_hi) / 2 - (t0_lo + t0_hi) / 2)

    # Distortion when the intervals overlap.
    d_overlap = (2 *
                 (torch.minimum(t0_hi, t1_hi)**3 - torch.maximum(t0_lo, t1_lo)**3) +
                 3 * (t1_hi * t0_hi * torch.abs(t1_hi - t0_hi) +
                      t1_lo * t0_lo * torch.abs(t1_lo - t0_lo) + t1_hi * t0_lo *
                      (t0_lo - t1_hi) + t1_lo * t0_hi *
                      (t1_lo - t0_hi))) / (6 * (t0_hi - t0_lo) * (t1_hi - t1_lo))

    # Are the two intervals not overlapping?
    are_disjoint = (t0_lo > t1_hi) | (t1_lo > t0_hi)

    return torch.where(are_disjoint, d_disjoint, d_overlap)


def anneal_loss_weight(weight: float, gamma: float, iter: int, mile: int):
    # exponentially anneal the loss weight
    return weight * gamma ** min(iter / mile, 1)


def gaussian_entropy_relighting4d(albedo_pred):
    albedo_entropy = 0
    for i in range(3):
        channel = albedo_pred[..., i]
        hist = GaussianHistogram(15, 0., 1., sigma=torch.var(channel))
        h = hist(channel)
        if h.sum() > 1e-6:
            h = h.div(h.sum()) + 1e-6
        else:
            h = torch.ones_like(h)
        albedo_entropy += torch.sum(-h * torch.log(h))
    return albedo_entropy


class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins, device=sigma.device).float() + 0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5 * (x / self.sigma)**2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta
        x = x.sum(dim=1)
        return x


def gaussian_entropy(x: torch.Tensor, *args, **kwargs):
    eps = 1e-6
    hps = 1e-9
    h = gaussian_histogram(x, *args, **kwargs)
    # h = (h / (h.sum(dim=0) + hps)).clip(eps)  # 3,
    # entropy = (-h * h.log()).sum(dim=0).sum(dim=0)  # per channel entropy summed
    entropy = 0
    for i in range(3):
        hi = h[..., i]
        if hi.sum() > eps:
            hi = hi / hi.sum() + eps
        else:
            hi = torch.ones_like(hi)
        entropy += torch.sum(-hi * torch.log(hi))
    return entropy


def gaussian_histogram(x: torch.Tensor, bins: int = 15, min: float = 0.0, max: float = 1.0):
    x = x.view(-1, x.shape[-1])  # N, 3
    sigma = x.var(dim=0)  # 3,
    delta = (max - min) / bins
    centers = min + delta * (torch.arange(bins, device=x.device, dtype=x.dtype) + 0.5)  # BIN
    x = x[None] - centers[:, None, None]  # BIN, N, 3
    x = (-0.5 * (x / sigma).pow(2)).exp() / (sigma * np.sqrt(np.pi * 2)) * delta  # BIN, N, 3
    x = x.sum(dim=1)
    return x  # BIN, 3


def reg_diff_crit(x: torch.Tensor, iter_step: int, max_weight: float = 1e-4, ann_iter: int = 100 * 500):
    weight = min(iter_step, ann_iter) * max_weight / ann_iter
    return reg(x), weight


def reg_raw_crit(x: torch.Tensor, iter_step: int, max_weight: float = 1e-4, ann_iter: int = 100 * 500):
    weight = min(iter_step, ann_iter) * max_weight / ann_iter
    n_batch, n_pts_x2, D = x.shape
    n_pts = n_pts_x2 // 2
    length = x.norm(dim=-1, keepdim=True)  # length
    vector = x / (length + 1e-8)  # vector direction (normalized to unit sphere)
    # loss_length = mse(length[:, n_pts:, :], length[:, :n_pts, :])
    loss_vector = reg((vector[:, n_pts:, :] - vector[:, :n_pts, :]))
    # loss = loss_length + loss_vector
    loss = loss_vector
    return loss, weight


def lpips(x: torch.Tensor, y: torch.Tensor, net='alex'):  # for computing loss, use alex, faster
    # B, 3, H, W
    # B, 3, H, W
    if not hasattr(lpips, 'net_map'):
        lpips.net_map = dotdict()
    if net not in lpips.net_map:
        import lpips as lpips_module
        log(f'Initializing LPIPS network: {green(net)}')
        lpips.net_map[net] = lpips_module.LPIPS(net=net, verbose=False).cuda()

    return lpips.net_map[net](x.cuda() * 2 - 1, y.cuda() * 2 - 1).mean()


def eikonal(x: torch.Tensor, th=1.0) -> torch.Tensor:
    return ((x.norm(dim=-1) - th)**2).mean()


def sdf_mask_crit(ret, batch):
    msk_sdf = ret['msk_sdf']
    msk_label = ret['msk_label']

    alpha = 50
    alpha_factor = 2
    alpha_milestones = [10000, 20000, 30000, 40000, 50000]
    for milestone in alpha_milestones:
        if batch['iter_step'] > milestone:
            alpha = alpha * alpha_factor

    msk_sdf = -alpha * msk_sdf
    mask_loss = F.binary_cross_entropy_with_logits(msk_sdf, msk_label) / alpha

    return mask_loss


def cross_entropy(x: torch.Tensor, y: torch.Tensor):
    # x: unormalized input logits
    # channel last cross entropy loss
    x = x.view(-1, x.shape[-1])  # N, C
    y = y.view(-1, y.shape[-1])  # N, C
    return F.cross_entropy(x, y)


def huber(x: torch.Tensor, y: torch.Tensor, delta: float = 1.0, reduction: str = 'mean'):
    return F.huber_loss(x, y, reduction=reduction, delta=delta)


def smoothl1(x: torch.Tensor, y: torch.Tensor):
    return F.smooth_l1_loss(x, y)


def weightl1(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
    return ((x - y).abs() * w).mean()


def weightl2(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
    return ((x - y) ** 2 * w).mean()


def mse(x: torch.Tensor, y: torch.Tensor):
    return ((x.float() - y.float())**2).mean()


def dot(x: torch.Tensor, y: torch.Tensor):
    return (x * y).sum(dim=-1)


def l1(x: torch.Tensor, y: torch.Tensor):
    return l1_reg(x - y)


def wl1(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor):
    return l1_reg(w * (x - y))


def l2(x: torch.Tensor, y: torch.Tensor):
    return l2_reg(x - y)


def l21(x: torch.Tensor, y: torch.Tensor):
    return torch.norm(x - y, dim=-1).mean()


def l1_reg(x: torch.Tensor):
    # return x.abs().sum(dim=-1).mean()
    return x.abs().mean()


def l2_reg(x: torch.Tensor) -> torch.Tensor:
    # return (x**2).sum(dim=-1).mean()
    return (x**2).mean()


def l1_with_confidence(x: torch.Tensor, y: torch.Tensor, confidence: torch.Tensor, alpha=0.2):
    assert x.shape == y.shape, f"x.shape: {x.shape}, y.shape: {y.shape}"
    assert x.shape[:-1] == confidence.shape, f"x.shape: {x.shape}, confidence.shape: {confidence.shape}"

    confidence = confidence.unsqueeze(-1)    
    reg_loss = (x - y).abs()
    # reg_loss = check_and_fix_inf_nan(reg_loss, 'reg_loss')
    conf_loss = reg_loss * confidence - alpha * torch.log(confidence)  # (N, 1)

    # print(f"conf_loss: {conf_loss.mean()} mean confidence: {confidence.mean()}, min confidence: {confidence.min()}, max confidence: {confidence.max()}")

    return conf_loss.mean()

def l21_with_confidence(x: torch.Tensor, y: torch.Tensor, confidence: torch.Tensor, alpha=0.2):
    assert x.shape == y.shape, f"x.shape: {x.shape}, y.shape: {y.shape}"
    assert x.shape[:-1] == confidence.shape, f"x.shape: {x.shape}, confidence.shape: {confidence.shape}"

    reg_loss = torch.norm(x - y, dim=-1)
    conf_loss = reg_loss * confidence - alpha * torch.log(confidence)  # (N, 1)

    return conf_loss.mean()


def bce_loss(x: torch.Tensor, y: torch.Tensor, w_logits: bool = False):
    if not w_logits:
        return F.binary_cross_entropy(x, y)
    else:
        return F.binary_cross_entropy_with_logits(x, y)


def cos(x: torch.Tensor, y: torch.Tensor):
    return (1 - F.cosine_similarity(x, y, dim=-1)).mean()


def mIoU_loss(x: torch.Tensor, y: torch.Tensor):
    """
    Compute the mean intersection of union loss over masked regions
    x, y: B, N, 1
    """
    I = (x * y).sum(-1).sum(-1)
    U = (x + y).sum(-1).sum(-1) - I
    mIoU = (I / (U.detach() + 1e-8)).mean()  # avoid nans
    return 1 - mIoU


def wreg(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-8):
    return (w * x.norm(dim=-1) + eps).mean()


def reg(x: torch.Tensor) -> torch.Tensor:
    return x.norm(dim=-1).mean()


# def conf_reg_loss(x: torch.Tensor, c: torch.Tensor, alpha: float = 1.0):
#     if c.numel() == 0: return 0
#     # FIXME: sure not to make torch.log(c) more safe?
#     m = x.norm(dim=-1, keepdim=True) * c - alpha * torch.log(c)
#     return m.mean() if m.numel() > 0 else 0


def thresh(x: torch.Tensor, a: torch.Tensor, eps: float = 1e-8):
    return 1 / (l2(x, a) + eps)


def elastic_crit(jac: torch.Tensor) -> torch.Tensor:
    """Compute the raw 'log_svals' type elastic energy, and
    remap it using the Geman-McClure type of robust loss.
    Args:
        jac (torch.Tensor): (B, N, 3, 3), the gradient of warpped xyz with respect to the original xyz
    Return:
        elastic_loss (torch.Tensor): (B, N), 
    """
    # !: CUDA IMPLEMENTATION OF SVD IS EXTREMELY SLOW
    # old_device = jac.device
    # jac = jac.cpu()
    # svd_backward: Setting compute_uv to false in torch.svd doesn't compute singular matrices, and hence we cannot compute backward. Please use torch.svd(compute_uv=True)
    _, S, _ = torch.svd(jac, compute_uv=True)           # (B, N, 3)
    # S = S.to(old_device)
    log_svals = torch.log(torch.clamp(S, min=1e-6))     # (B, N, 3)
    sq_residual = torch.sum(log_svals**2, dim=-1)       # (B, N)
    # TODO: determine whether it is a good choice to compute the robust loss here
    elastic_loss = general_loss_with_squared_residual(sq_residual, alpha=-2.0, scale=0.03)
    return elastic_loss


def general_loss_with_squared_residual(squared_x, alpha, scale):
    r"""The general loss that takes a squared residual.
    This fuses the sqrt operation done to compute many residuals while preserving
    the square in the loss formulation.
    This implements the rho(x, \alpha, c) function described in "A General and
    Adaptive Robust Loss Function", Jonathan T. Barron,
    https://arxiv.org/abs/1701.03077.
    Args:
        squared_x: The residual for which the loss is being computed. x can have
        any shape, and alpha and scale will be broadcasted to match x's shape if
        necessary.
        alpha: The shape parameter of the loss (\alpha in the paper), where more
        negative values produce a loss with more robust behavior (outliers "cost"
        less), and more positive values produce a loss with less robust behavior
        (outliers are penalized more heavily). Alpha can be any value in
        [-infinity, infinity], but the gradient of the loss with respect to alpha
        is 0 at -infinity, infinity, 0, and 2. Varying alpha allows for smooth
        interpolation between several discrete robust losses:
            alpha=-Infinity: Welsch/Leclerc Loss.
            alpha=-2: Geman-McClure loss.
            alpha=0: Cauchy/Lortentzian loss.
            alpha=1: Charbonnier/pseudo-Huber loss.
            alpha=2: L2 loss.
        scale: The scale parameter of the loss. When |x| < scale, the loss is an
        L2-like quadratic bowl, and when |x| > scale the loss function takes on a
        different shape according to alpha.
    Returns:
        The losses for each element of x, in the same shape as x.
    """
    # https://pytorch.org/docs/stable/type_info.html
    eps = torch.tensor(torch.finfo(torch.float32).eps)

    # convert the float to torch.tensor
    alpha = torch.tensor(alpha).to(squared_x.device)
    scale = torch.tensor(scale).to(squared_x.device)

    # This will be used repeatedly.
    squared_scaled_x = squared_x / (scale ** 2)

    # The loss when alpha == 2.
    loss_two = 0.5 * squared_scaled_x
    # The loss when alpha == 0.
    loss_zero = log1p_safe(0.5 * squared_scaled_x)
    # The loss when alpha == -infinity.
    loss_neginf = -torch.expm1(-0.5 * squared_scaled_x)
    # The loss when alpha == +infinity.
    loss_posinf = expm1_safe(0.5 * squared_scaled_x)

    # The loss when not in one of the above special cases.
    # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
    beta_safe = torch.maximum(eps, torch.abs(alpha - 2.))
    # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
    alpha_safe = torch.where(
        torch.greater_equal(alpha, torch.tensor(0.)), torch.ones_like(alpha),
        -torch.ones_like(alpha)) * torch.maximum(eps, torch.abs(alpha))
    loss_otherwise = (beta_safe / alpha_safe) * (
        torch.pow(squared_scaled_x / beta_safe + 1., 0.5 * alpha) - 1.)

    # Select which of the cases of the loss to return.
    loss = torch.where(
        alpha == -torch.inf, loss_neginf,
        torch.where(
            alpha == 0, loss_zero,
            torch.where(
                alpha == 2, loss_two,
                torch.where(alpha == torch.inf, loss_posinf, loss_otherwise))))

    return scale * loss


def log1p_safe(x):
    """The same as torch.log1p(x), but clamps the input to prevent NaNs."""
    return torch.log1p(torch.minimum(x, torch.tensor(3e37)))


def expm1_safe(x):
    """The same as torch.expm1(x), but clamps the input to prevent NaNs."""
    return torch.expm1(torch.minimum(x, torch.tensor(87.5)))


def compute_plane_tv(t):
    batch_size, c, h, w = t.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    h_tv = torch.square(t[..., 1:, :] - t[..., :h - 1, :]).sum()
    w_tv = torch.square(t[..., :, 1:] - t[..., :, :w - 1]).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)  # This is summing over batch and c instead of avg


def compute_planes_tv(embedding):
    tv_loss = 0
    for emb in embedding:
        tv_loss += compute_plane_tv(emb)
    return tv_loss


def compute_plane_smoothness(t):
    batch_size, c, h, w = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:] - t[..., :w - 1]  # [batch, c, h-1, w]
    second_difference = first_difference[..., 1:] - first_difference[..., :w - 2]  # [batch, c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()


def compute_time_planes_smooth(embedding):
    loss = 0.
    for emb in embedding:
        loss += compute_plane_smoothness(emb)
    return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def gsssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(x: torch.Tensor, y: torch.Tensor, data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03)):
    from easyvolcap.utils.ssim_utils import ssim as compute_ssim
    return compute_ssim(x, y, data_range=data_range, win_size=win_size, win_sigma=win_sigma, K=K)


def msssim(x: torch.Tensor, y: torch.Tensor, data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03)):
    from easyvolcap.utils.ssim_utils import ms_ssim as compute_msssim
    return compute_msssim(x, y, data_range=data_range, win_size=win_size, win_sigma=win_sigma, K=K)


# from MonoSDF
def compute_scale_and_shift(prediction, target, mask):
    # System matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # Right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # Solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # Average of all valid pixels of the batch
    # Avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0: return torch.tensor(0.0)
    else: return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # Mean of average of valid pixels of an image
    # Avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()
    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):
    # Number of valid pixels
    M = torch.sum(mask, (1, 2))  # (B,)

    # L2 loss
    res = prediction - target  # (B, H, W)
    image_loss = torch.sum(mask * res * res, (1, 2))  # (B,)

    return reduction(image_loss, 2 * M)


def gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    grad_max: float = 100.0,
    confidence: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    alpha: float = 0.2,
    reduction: Callable = reduction_batch_based,
):
    """ Compute the gradient loss between the prediction and the target.

    Args:
        prediction (torch.Tensor), (B, H, W) or (B, H, W, C): The prediction tensor.
        target (torch.Tensor), (B, H, W) or (B, H, W, C): The target tensor.
        mask (torch.Tensor), (B, H, W) or (B, H, W, 1): The mask tensor.
        confidence (torch.Tensor), (B, H, W) or (B, H, W, 1): The confidence tensor.
        gamma (float): The gamma parameter.
        alpha (float): The alpha parameter.
        reduction (Callable): The reduction method.
    
    Returns:
        grad_x (torch.Tensor), (B, H, W): The gradient tensor.
        grad_y (torch.Tensor), (B, H, W): The gradient tensor.
        M (torch.Tensor), (B,): The mask tensor.
    """
    # Expand the mask to the same shape as the prediction
    if mask.ndim < prediction.ndim:
        mask = mask[..., None].expand(
            (-1,) * (prediction.ndim - 1) + (prediction.shape[-1],)
        )  # (B, H, W, C)

    # Number of valid pixels in each batch
    M = torch.sum(mask, (1, 2, 3))  # (B,)

    # Compute the difference between the prediction and the target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute the gradient in the x direction
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x).clamp(max=grad_max)

    # Compute the gradient in the y direction
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y).clamp(max=grad_max)

    # Compute the confidence-weighted loss
    if confidence is not None:
        # Expand the confidence to the same shape as the gradient
        if confidence.ndim < grad_x.ndim:
            confidence = confidence[..., None].expand(-1, -1, -1, grad_x.shape[-1])

        # Weight the gradient in the x direction
        conf_x = confidence[:, :, 1:]
        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)

        # Weight the gradient in the y direction
        conf_y = confidence[:, 1:, :]
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Compute the total gradient loss
    image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    return reduction(image_loss, M)


def gradient_loss_multiscale(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    grad_max: float = 100.0,
    confidence: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    alpha: float = 0.2,
    reduction: str = 'batch-based',
    scales: int = 4,
):
    # Determine the reduction method
    if reduction == 'batch-based': reduction = reduction_batch_based
    elif reduction == 'image-based': reduction = reduction_image_based
    else: reduction = lambda x, y: x

    total = 0

    for scale in range(scales):
        step = pow(2, scale)

        total += gradient_loss(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            grad_max=grad_max,
            confidence=confidence[:, ::step, ::step] if confidence is not None else None,
            gamma=gamma,
            alpha=alpha,
            reduction=reduction
        )

    total = total / scales
    return total


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        elif reduction == 'image-based':
            self.__reduction = reduction_image_based
        else:
            self.__reduction = lambda x, y: x

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self,
        scales: int = 4,
        grad_max: float = 100.0,
        gamma: float = 1.0,
        alpha: float = 0.2,
        reduction: str = 'batch-based',
    ):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        elif reduction == 'image-based':
            self.__reduction = reduction_image_based
        else:
            self.__reduction = lambda x, y: x

        self.__scales = scales
        self.__grad_max = grad_max
        self.__gamma = gamma
        self.__alpha = alpha

    def forward(self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                grad_max=self.__grad_max,
                confidence=confidence[:, ::step, ::step] if confidence is not None else None,
                gamma=self.__gamma,
                alpha=self.__alpha,
                reduction=self.__reduction
            )

        total = total / self.__scales
        return total


class ScaleAndShiftInvariantMSELoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        # Deal with the channel dimension, the input dimension may have (B, C, H, W) or (B, H, W)
        if prediction.ndim == 4: prediction = prediction[:, 0]  # (B, H, W)
        if target.ndim == 4: target = target[:, 0]  # (B, H, W)
        if mask.ndim == 4: mask = mask[:, 0]  # (B, H, W)

        # Compute scale and shift
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
        total = self.__data_loss(self.__prediction_ssi, target, mask)

        # Add regularization if needed
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
# from MonoSDF


def median_normalize(x, mask):
    """ Median normalize a tensor for all valid pixels.
        This operation is performed without batch dimension.
    Args:
        x (torch.Tensor): (H, W) or (H, W, C), original tensor
        mask (torch.Tensor): (H, W) or (H, W, 1), mask tensor
    Return:
        y (torch.Tensor): (H, W) or (H, W, C), median normalized tensor
    """
    if mask.shape[-1] == 1:
        mask = mask.squeeze(-1)

    M = torch.sum(mask)

    # Return original tensor if there is no valid pixel
    if M == 0:
        return x

    # Compute median and scale
    t = torch.quantile(x[mask == 1], q=0.5, dim=0)  # scalar or (C,)
    # FIXME: Figure out why we need a abs to avoid inconvergence
    s = torch.sum(torch.abs(x[mask == 1] - t)) / M  # scalar

    # Return median normalized tensor
    return (x - t) / s


def mae_loss(prediction, target, mask, reduction=reduction_batch_based, weight=None, beta=0.0):
    # Number of valid pixels
    M = torch.sum(mask, dim=[-i for i in range(1, mask.ndim)])  # (B,)

    # L1 loss
    res = (prediction - target).abs()  # (B, H, W) or (B, H, W, C)

    # Apply weight if needed
    if weight is not None:
        res = weight * res  # (B, H, W) or (B, H, W, C)

    # Apply beta smooth if needed
    if beta > 0:
        res = smooth(res, beta)

    image_loss = torch.sum(mask * res, dim=[-i for i in range(1, res.ndim)])  # (B,)

    return reduction(image_loss, 2 * M)


class MAELoss(nn.Module):
    def __init__(self, reduction='batch-based', beta=0.0):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        elif reduction == 'image-based':
            self.__reduction = reduction_image_based
        else:
            self.__reduction = lambda x, y: x

        self.__beta = beta

    def forward(self, prediction, target, mask, weight=None):
        return mae_loss(prediction, target, mask, reduction=self.__reduction, weight=weight, beta=self.__beta)


class ScaleAndShiftInvariantMAELoss(nn.Module):
    def __init__(self, alpha=0.0, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MAELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

    def forward(self, prediction, target, mask, weight=None):
        # The input dimension must have (B, H, W) or (B, H, W, C)
        # TODO: Maybe there is a better way to do the batching
        # But `torch.quantile` does not support multiple `dim` argument for now
        for i in range(prediction.shape[0]):
            prediction[i] = median_normalize(prediction[i], mask[i])  # (H, W) or (H, W, C)
            target[i] = median_normalize(target[i], mask[i])  # (H, W) or (H, W, C)

        # Compute the scale-and-shift invariant MAE loss
        total = self.__data_loss(prediction, target, mask, weight)

       # Add regularization if needed
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(prediction, target, mask)

        return total


def check_and_fix_inf_nan(x: torch.Tensor, name: str, max_value: float = 100):
    """ Checks if 'x' contains inf or nan. If it does, replace those
        values with zero and print the name of the loss tensor.

    Args:
        x (torch.Tensor): The loss tensor to check.
        name (str): Name of the loss (for diagnostic prints).

    Returns:
        torch.Tensor: The checked and fixed loss tensor, with inf/nan replaced by 0.
    """
    if torch.isnan(x).any() or torch.isinf(x).any():
        for _ in range(10):
            nan_num = torch.isnan(x).sum()
            inf_num = torch.isinf(x).sum()
            nan_ratio = nan_num / x.numel()
            inf_ratio = inf_num / x.numel()
            print(f"{name} has {inf_num} inf (ratio {inf_ratio}) and {nan_num} nan (ratio {nan_ratio}). Setting those values to 0.")
            if x.numel() > 100 and (nan_ratio + inf_ratio) > 0.1:
                raise ValueError(f"{name} has {inf_num} inf (ratio {inf_ratio}) and {nan_num} nan (ratio {nan_ratio}).")
        x = torch.where(
            torch.isnan(x) | torch.isinf(x),
            torch.tensor(0.0, device=x.device),
            x
        )

    x = torch.clamp(x, min=-max_value, max=max_value)
    return x


def filter_by_quantile(
    x: torch.Tensor,
    range: float,
    min_numel: int = 1000,
    max_numel: int = 100000000,
    rad_numel: int = 1000000,
    max_value: float = 100,
):
    """ Filter a loss tensor by keeping only values below a certain quantile threshold.
        Also clamps individual values to hard_max.

    Args:
        x (torch.Tensor): Tensor containing loss values
        valid_range (float): Float between 0 and 1 indicating the quantile threshold
        min_elements (int): Minimum number of elements required to apply filtering
        hard_max (float): Maximum allowed value for any individual loss
        valid_range (float): Float between 0 and 1 indicating the quantile threshold
        min_elements (int): Minimum number of elements required to apply filtering
        hard_max (float): Maximum allowed value for any individual loss

    Returns:
        Filtered and clamped loss tensor
    """
    # Directly return if the tensor is too small
    if x.numel() <= min_numel:
        return x

    # Randomly sample some elements if the tensor is too large
    if x.numel() > max_numel:
        inds = torch.randperm(x.numel(), device=x.device)[:rad_numel]
        x = x.view(-1)[inds]

    # Clamp individual values to avoid outliers
    x = x.clamp(max=max_value)

    # Compute the quantile threshold
    thresh = torch_quantile(x.detach(), range)
    thresh = min(thresh, max_value)

    # Apply quantile filtering if enough elements remain
    m = x < thresh
    if m.sum() > min_numel:
        return x[m]
    # Return the original tensor if no elements are kept
    return x


def torch_quantile(
    input: torch.Tensor,
    q: float | torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Sanitization: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Sanitization: inteporlation
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Sanitization: out
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Logic
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Rectification: keepdim
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


def conf_grad_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    confidence: torch.Tensor,

    conf_disable: bool = False,
    grad_conf_disable: bool = True,
    range: float = -1.0,
    grad_max: float = 100.0,
    gamma: float = 1.0,
    alpha: float = 0.2,
    reduction: str = 'batch-based',
    scales: int = 4,
) -> torch.Tensor:
    # Compute the regression loss
    reg_loss = torch.norm(prediction[mask] - target[mask], dim=-1, keepdim=True)  # (N, 1)
    reg_loss = check_and_fix_inf_nan(reg_loss, 'reg_loss')

    # If confidence is disabled, return the gradient loss
    if conf_disable:
        conf_loss = gamma * reg_loss  # (N, 1)
    else:
        conf_loss = gamma * reg_loss * confidence[mask] - alpha * torch.log(confidence[mask])  # (N, 1)

    if conf_loss.numel() > 0:
        # Filter the loss by quantile if range is set
        if range > 0:
            conf_loss = filter_by_quantile(conf_loss, range)
        conf_loss = check_and_fix_inf_nan(conf_loss, 'conf_loss')
    else:
        conf_loss = prediction * 0.0
        print(f"No valid confidence values. Setting conf_loss to 0.")

    # Compute the gradient loss
    grad_loss = gradient_loss_multiscale(
        prediction.reshape(-1, *prediction.shape[2:]),  # (B * S, H, W, C)
        target.reshape(-1, *target.shape[2:]),  # (B * S, H, W, C)
        mask.reshape(-1, *mask.shape[2:]),  # (B * S, H, W)
        confidence=None if grad_conf_disable else confidence.reshape(-1, *confidence.shape[2:]),  # (B * S, H, W, 1)
        grad_max=grad_max,
        gamma=gamma,
        alpha=alpha,
        reduction=reduction,
        scales=scales,
    )
    grad_loss = check_and_fix_inf_nan(grad_loss, 'grad_loss')

    return conf_loss, grad_loss


# Modified version of Adabins repository
# https://github.com/shariqfarooq123/AdaBins/blob/0952d91e9e762be310bb4cd055cbfe2448c0ce20/loss.py#L7
class ScaleInvariantLogLoss(nn.Module):
    def __init__(self, alpha=10.0, beta=0.15, eps=0.0):
        super(ScaleInvariantLogLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        # The eps is added to avoid log(0) and division by zero
        # But it should be gauranteed that the network output is always non-negative
        self.eps = eps

    def forward(self, prediction, target, mask):
        # Deal with the channel dimension, the input dimension may have (B, C, H, W) or (B, H, W)
        if prediction.ndim == 4: prediction = prediction[:, 0]  # (B, H, W)
        if target.ndim == 4: target = target[:, 0]  # (B, H, W)
        if mask.ndim == 4: mask = mask[:, 0]  # (B, H, W)

        total = 0
        # Maybe there is a better way to do the batching
        for i in range(prediction.shape[0]):
            g = torch.log(prediction[i][mask[i]] + self.eps) - torch.log(target[i][mask[i]] + self.eps)  # (N,)
            Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)  # scalar
            total += self.alpha * torch.sqrt(Dg)

        return total


# Adapted from MoGe
# https://github.com/microsoft/MoGe
def roe_alignment_subproblem_wo_trunc(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    trunc: float = None,  # for compatibility
    eps: float = 1e-7
) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
    """ Implementation of the ROE alignment subproblem without truncation.
    See the supplymentary Algorithm 2 of MoGe for more details: http://arxiv.org/abs/2410.19115.
    Solve `min sum_i w_i * |a * x_i - y_i|`, note that w_i must be >= 0.
    The input tensors should not have the batch dimension.

    Args:
        x (torch.Tensor), (..., C): predicted values
        y (torch.Tensor), (..., C): target values
        w (torch.Tensor), (..., C): weights

    Returns:
        scale (torch.Tensor), (...,): differentiable
        index (torch.Tensor), (...,)：index of where s = y[idx] / x[idx]
        loss (torch.Tensor), (...,): value of loss function at `s`, detached
    """
    # Broadcast to the same shape
    x, y, w = torch.broadcast_tensors(x, y, w)  # (..., C), (..., C), (..., C)

    # Sort the division of y / x
    s = torch.sign(x)  # (..., C)
    x, y = s * x, s * y  # FIXME: why do we need to multiply the sign?
    division = y / x.clamp_min(eps)  # (..., C)
    division, indice = division.sort(dim=-1)  # (..., C), (..., C)

    wx = torch.gather(w * x, dim=-1, index=indice)  # (..., C)
    grad = 2 * wx.cumsum(dim=-1) - wx.sum(dim=-1, keepdim=True)  # (..., C)
    search = torch.searchsorted(
        grad,
        torch.zeros_like(grad[..., :1]),
        side='left'
    ).clamp_max(grad.shape[-1] - 1)

    scale = division.gather(dim=-1, index=search).squeeze(-1)  # (...,)
    index = indice.gather(dim=-1, index=search).squeeze(-1)  # (...,)
    loss = (w * (scale[..., None] * x - y).abs()).sum(dim=-1)  # (...,)

    return scale, index, loss


def extend_inf(x: torch.Tensor):
    return torch.cat([
        torch.full_like(x[..., :1], -torch.inf),
        x,
        torch.full_like(x[..., :1], torch.inf)
    ], dim=-1)


def extend_cumsum(cumsum: torch.Tensor):
    return torch.cat([
        torch.zeros_like(cumsum[..., :1]),
        cumsum,
        cumsum[..., -1:]
    ], dim=-1)


@torch.jit.script
def truncated_residule(
    a: torch.Tensor,
    xyw: torch.Tensor,
    trunc: float
):
    return a.mul(xyw[..., 0]).sub(xyw[..., 1]).abs().mul(xyw[..., 2]).clamp_max(trunc).sum(dim=-1)


def scatter_minimum(
    size: int,
    dim: int,
    index: torch.LongTensor,
    src: torch.Tensor
) -> torch.return_types.min:
    # Scatter the minimum value along the given dimension of
    # `input` into `src` at the indices specified in `index`
    sh = src.shape[:dim] + (size,) + src.shape[dim + 1:]

    # Create a tensor with the same shape as `src` as output
    out = torch.full(sh, float("inf"), dtype=src.dtype, device=src.device)
    out = out.scatter_reduce(dim=dim, index=index, src=src, reduce="amin", include_self=False)

    # Find the indices of the minimum value
    cor = torch.where(src == torch.gather(out, dim=dim, index=index))
    idx = torch.full(sh, -1, dtype=torch.long, device=src.device)
    # Fill the indices of the minimum value
    idx[(*cor[:dim], index[cor], *cor[dim + 1:])] = cor[dim]

    return torch.return_types.min((out, idx))


def roe_alignment_subproblem_wi_trunc(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    trunc: float = 1.0,
    eps: float = 1e-7,
    max_elements: int = 4096 ** 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
    """ Implementation of the ROE alignment subproblem without truncation.
    See the supplymentary Algorithm 3 of MoGe for more details: http://arxiv.org/abs/2410.19115.
    Solve `min sum_i min(trunc, w_i * |a * x_i - y_i|)`, note that w_i must be >= 0.
    The input tensors should not have the batch dimension.

    Args:
        x (torch.Tensor), (..., C): predicted values
        y (torch.Tensor), (..., C): target values
        w (torch.Tensor), (..., C): weights
        trunc (float): truncation value

    Returns:
        scale (torch.Tensor), (...,): differentiable
        index (torch.Tensor), (...,)：index of where s = y[idx] / x[idx]
        loss (torch.Tensor), (...,): value of loss function at `s`, detached
    """
    # Broadcast to the same shape
    x, y, w = torch.broadcast_tensors(x, y, w)  # (..., C), (..., C), (..., C)
    # Reshape the pre-pending dimensions to a single dimension
    sh = x.shape[:-1]
    x, y, w = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]), w.reshape(-1, w.shape[-1])  # (P, C)

    # Prepare variables
    s = torch.sign(x)  # (..., C)
    x, y =  s* x, s * y  # FIXME: why do we need to multiply the sign?
    wx, wy = w * x, w * y  # (P, C), (P, C)
    xyw = torch.stack([x, y, w], dim=-1)  # (P, C, 3)

    # Three conditions
    A = y / x.clamp_min(eps)  # (P, C)
    B = (wy - trunc) / wx.clamp_min(eps)  # (P, C)
    C = (wy + trunc) / wx.clamp_min(eps)  # (P, C)

    with torch.no_grad():
        # Calculate predix sum by orders of A, B, C
        A, A_indice = A.sort(dim=-1)  # FIXME: why put A in the first place outside the `torch.no_grad()`
        A = A.contiguous()  # FIXME: why do we need to do this?
        A_cumsum = torch.cumsum(torch.gather(wx, dim=-1, index=A_indice), dim=-1)  # (P, C)

        # Calculate B and its prefix sum
        B, B_indice = B.sort(dim=-1)  # (P, C)
        B_cumsum = torch.cumsum(torch.gather(wx, dim=-1, index=B_indice), dim=-1)  # (P, C)
        B, B_cumsum = extend_inf(B), extend_cumsum(B_cumsum)  # (P, C), (P, C)

        # Calculate C and its prefix sum
        C, C_indice = C.sort(dim=-1)  # (P, C)
        C_cumsum = torch.cumsum(torch.gather(wx, dim=-1, index=C_indice), dim=-1)  # (P, C)
        C, C_cumsum = extend_inf(C), extend_cumsum(C_cumsum)  # (P, C), (P, C)

        # Calculate the left gradient of A
        B_search = torch.searchsorted(B, A, side='left').sub_(1)  # (P, C)
        C_search = torch.searchsorted(C, A, side='left').sub_(1)  # (P, C)
        l_grad = (
            2 * torch.cat([torch.zeros_like(A_cumsum[..., :1]), A_cumsum[..., :-1]], dim=-1)  # (P, C)
            - torch.gather(B_cumsum, dim=-1, index=B_search)  # (P, C)
            - torch.gather(C_cumsum, dim=-1, index=C_search)  # (P, C)
        )  # (P, C)

        # Calculate the right gradient of A
        B_search = torch.searchsorted(B, A, side='right').sub_(1)  # (P, C)
        C_search = torch.searchsorted(C, A, side='right').sub_(1)  # (P, C)
        r_grad = (
            2 * A_cumsum  # (P, C)
            - torch.gather(B_cumsum, dim=-1, index=B_search)  # (P, C)
            - torch.gather(C_cumsum, dim=-1, index=C_search)  # (P, C)
        )  # (P, C)

        # Find extrema and calculate their values
        is_extrema = (l_grad < 0) & (r_grad >= 0)  # (P, C)
        # In case all derivatives are zero, take the first one as extrema.
        is_extrema[..., 0] |= ~is_extrema.any(dim=-1)  # (P, C)
        extrema_i, extrema_j = torch.where(is_extrema)  # i in the P, j in the C
        extrema_a = A[extrema_i, extrema_j]  # (N,)

        # Split into small batches to avoid OOM (~1G for 4096^2)
        chunk_size = max_elements // x.shape[-1]
        extrema_v = torch.cat([
            # Inplace operations to save memory
            truncated_residule(
                extrema_a_split[:, None], xyw[extrema_i_split, :, :], trunc
            )
            for extrema_a_split, extrema_i_split in zip(
                extrema_a.split(chunk_size), extrema_i.split(chunk_size)
            )
        ])  # (N,)

        # Find minima among corresponding extrema
        minima, indices = scatter_minimum(
            size=math.prod(sh), dim=0, index=extrema_i, src=extrema_v
        )  # (P,)
        search = extrema_j[indices]

    scale = torch.gather(A, dim=-1, index=search[..., None]).squeeze(-1)
    scale = scale.reshape(sh)
    index = torch.gather(A_indice, dim=-1, index=search[..., None]).squeeze(-1)
    loss = minima.reshape(sh)

    return scale, index, loss


def align_xyz_scale_z_shift(
    src_xyz: torch.Tensor,
    tar_xyz: torch.Tensor,
    weight: Optional[torch.Tensor],
    trunc: Optional[Union[float, torch.Tensor]] = None,
    max_elements: int = 2 ** 20
):
    """ Align `src_xyz` to `tar_xyz` with respect to a shared xyz scale and z shift.
    It is similar to `align_affine`, but scale and shift are applied to different dimensions.
    The input tensors should not have the batch dimension.

    Args:
        src_xyz (torch.Tensor), (..., P, 3): source xyz coordinates
        tar_xyz (torch.Tensor), (..., P, 3): target xyz coordinates
        weight (torch.Tensor), (..., P): weights for each point
        trunc (float): truncation value
        max_elements (int): maximum number of elements to process in parallel

    Returns:
        scale (torch.Tensor), (...): optimal scale
        shift (torch.Tensor), (..., 3): optimal shift, x and y shifts are zeros
    """
    # Lazy import to avoid circular import
    from easyvolcap.utils.chunk_utils import chunkify

    # Shape things
    sh = src_xyz.shape[:-2]
    B, P = math.prod(sh), src_xyz.shape[-2]
    src_xyz, tar_xyz, weight = src_xyz.reshape(B, P, 3), tar_xyz.reshape(B, P, 3), weight.reshape(B, P)

    # Decroration for the parallel computation
    fn = chunkify(max_elements // P, dim=0, print_progress=False, move_to_cpu=False)(
        roe_alignment_subproblem_wo_trunc
        if trunc is None
        else roe_alignment_subproblem_wi_trunc
    )

    # Find indices of valid points
    bidx, pidx = torch.where(weight > 0)  # (N,), (N,)

    # Compute the optimal scale and shift
    with torch.no_grad():
        # Prepare the optimal alignment inputs, only consider the z values
        xyfill = torch.zeros(bidx.shape[0]).to(src_xyz)  # (N,)
        source = torch.stack([xyfill, xyfill, src_xyz[bidx, pidx, 2]], dim=-1)  # (N, 3)
        target = torch.stack([xyfill, xyfill, tar_xyz[bidx, pidx, 2]], dim=-1)  # (N, 3)

        # Compute the residual, FIXME: why do we need to subtract the input?
        source = src_xyz[bidx, :, :] - source[..., None, :]  # (N, P, 3)
        target = tar_xyz[bidx, :, :] - target[..., None, :]  # (N, P, 3)

        # Solve optimal scale and shift for each anchor
        scale, index, loss = fn(
            source.flatten(-2),  # (N, P * 3)
            target.flatten(-2),  # (N, P * 3)
            weight[bidx, :, None].expand(-1, -1, 3).flatten(-2),  # (N, P * 3)
            trunc,  # (N,)
        )  # (N,), (N,), (N,)

        # Scatter the optimal scale and shift to the batch
        loss, aidx = scatter_minimum(
            size=B, dim=0, index=bidx, src=loss
        )  # (N,)

    # Reproduce by indexing for shorter compute graph
    idx2 = index[aidx]  # (B,), [0, 3n)
    idx1 = pidx[aidx] * 3 + idx2 % 3  # (B,), [0, 3n)

    # Create the source and target with zero x and y
    xyfill = torch.zeros((B, P)).to(src_xyz)  # (B, P)
    src_xyz_00z = torch.stack([xyfill, xyfill, src_xyz[..., 2]], dim=-1)  # (B, P, 3)
    tar_xyz_00z = torch.stack([xyfill, xyfill, tar_xyz[..., 2]], dim=-1)  # (B, P, 3)

    # Gather the necessary values
    src1 = torch.gather(src_xyz_00z.flatten(-2), dim=1, index=idx1[..., None]).squeeze(-1)  # (B, 3)
    tar1 = torch.gather(tar_xyz_00z.flatten(-2), dim=1, index=idx1[..., None]).squeeze(-1)  # (B, 3)
    src2 = torch.gather(src_xyz.flatten(-2), dim=1, index=idx2[..., None]).squeeze(-1)  # (B, 3)
    tar2 = torch.gather(tar_xyz.flatten(-2), dim=1, index=idx2[..., None]).squeeze(-1)  # (B, 3)

    # Compute the optimal scale and shift
    scale = (tar2 - tar1) / torch.where(src2 != src1, src2 - src1, 1.0)  # (B, 3)
    shift = torch.gather(
        tar_xyz_00z, dim=1, index=(idx1 // 3)[..., None, None].expand(-1, -1, 3)  # (B, P, 3)
    ).squeeze(-2) - scale[..., None] * torch.gather(
        src_xyz_00z, dim=1, index=(idx1 // 3)[..., None, None].expand(-1, -1, 3)  # (B, P, 3)
    ).squeeze(-2)
    scale, shift = scale.reshape(sh), shift.reshape(*sh, 3)  # (...,), (..., 3)

    return scale, shift


def align_xyz_scale_xyz_shift(
    src_xyz: torch.Tensor,
    tar_xyz: torch.Tensor,
    weight: Optional[torch.Tensor],
    trunc: Optional[Union[float, torch.Tensor]] = None,
    max_elements: int = 2 ** 20
):
    """ Align `src_xyz` to `tar_xyz` with respect to a shared xyz scale and z shift.
    It is similar to `align_affine`, but scale and shift are applied to different dimensions.
    The input tensors should not have the batch dimension.

    Args:
        src_xyz (torch.Tensor), (..., P, 3): source xyz coordinates
        tar_xyz (torch.Tensor), (..., P, 3): target xyz coordinates
        weights (torch.Tensor), (..., P): weights for each point

    Returns:
        scale (torch.Tensor), (...): optimal scale
        shift (torch.Tensor), (..., 3): optimal shift
    """
    # Lazy import to avoid circular import
    from easyvolcap.utils.chunk_utils import chunkify

    # Shape things
    sh = src_xyz.shape[:-2]
    B, P = math.prod(sh), src_xyz.shape[-2]
    src_xyz, tar_xyz, weight = src_xyz.reshape(B, P, 3), tar_xyz.reshape(B, P, 3), weight.reshape(B, P)

    # Decroration for the parallel computation
    fn = chunkify(max_elements // P, dim=0, print_progress=False, move_to_cpu=False)(
        roe_alignment_subproblem_wo_trunc
        if trunc is None
        else roe_alignment_subproblem_wi_trunc
    )

    # Find indices of valid points
    bidx, pidx = torch.where(weight > 0)  # (N,), (N,)

    # Compute the optimal scale and shift
    with torch.no_grad():
        # Prepare the optimal alignment inputs
        source = src_xyz[bidx, :, :] - src_xyz[bidx, pidx][..., None, :]  # (N, P, 3)
        target = tar_xyz[bidx, :, :] - tar_xyz[bidx, pidx][..., None, :]  # (N, P, 3)

        # Solve optimal scale and shift for each anchor
        scale, index, loss = fn(
            source.flatten(-2),  # (N, P * 3)
            target.flatten(-2),  # (N, P * 3)
            weight[bidx, :, None].expand(-1, -1, 3).flatten(-2),  # (N, P * 3)
            trunc,  # (N,)
        )

        # Scatter the optimal scale and shift to the batch
        loss, aidx = scatter_minimum(
            size=B, dim=0, index=bidx, src=loss
        )  # (N,)

    # Reproduce by indexing for shorter compute graph
    idx2 = index[aidx]  # (B,), [0, 3n)
    idx1 = pidx[aidx] * 3 + idx2 % 3  # (B,), [0, 3n)

    # Gather the necessary values
    src1 = torch.gather(src_xyz.flatten(-2), dim=1, index=idx1[..., None]).squeeze(-1)  # (B, 3)
    tar1 = torch.gather(tar_xyz.flatten(-2), dim=1, index=idx1[..., None]).squeeze(-1)  # (B, 3)
    src2 = torch.gather(src_xyz.flatten(-2), dim=1, index=idx2[..., None]).squeeze(-1)  # (B, 3)
    tar2 = torch.gather(tar_xyz.flatten(-2), dim=1, index=idx2[..., None]).squeeze(-1)  # (B, 3)

    # Compute the optimal scale and shift
    scale = (tar2 - tar1) / torch.where(src2 != src1, src2 - src1, 1.0)  # (B, 3)
    shift = torch.gather(
        tar_xyz, dim=1, index=(idx1 // 3)[..., None, None].expand(-1, -1, 3)  # (B, P, 3)
    ).squeeze(-2) - scale[..., None] * torch.gather(
        src_xyz, dim=1, index=(idx1 // 3)[..., None, None].expand(-1, -1, 3)  # (B, P, 3)
    ).squeeze(-2)
    scale, shift = scale.reshape(sh), shift.reshape(*sh, 3)  # (...,), (..., 3)

    return scale, shift


def smooth(
    x: torch.FloatTensor,
    beta: float = 0.0
) -> torch.FloatTensor:
    if beta == 0:
        return x
    else:
        return torch.where(
            x < beta,
            0.5 * x.square() / beta,
            x - 0.5 * beta
        )


def weighted_mean(
    x: torch.Tensor,
    w: torch.Tensor = None,
    dim: Union[int, torch.Size] = None,
    keepdim: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(
            dim=dim, keepdim=keepdim
        ).add(eps)


def harmonic_mean(
    x: torch.Tensor,
    w: torch.Tensor = None,
    dim: Union[int, torch.Size] = None,
    keepdim: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    if w is None:
        return x.add(eps).reciprocal().mean(dim=dim, keepdim=keepdim).reciprocal()
    else:
        w = w.to(x.dtype)
        return (
            weighted_mean(x.add(eps).reciprocal(), w, dim=dim, keepdim=keepdim, eps=eps)
            .add(eps)
            .reciprocal()
        )


def align_xyz_scale_shift_roe(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    size: int = 32,
    trunc: float = 1.0,
):
    # Lazy import
    from easyvolcap.utils.moge_utils import mask_aware_nearest_resize

    # The input dimension must have (H, W)
    if mask.shape[-1] == 1:
        mask = mask.squeeze(-1)

    # Mask-aware nearest resize, we only need a small set of point to compute the alignment
    idx, msk = mask_aware_nearest_resize(mask, size, size)

    if msk.sum() == 0:
        return prediction

    # Compute the optimal scale and shift
    scale, shift = align_xyz_scale_xyz_shift(
        prediction[idx].flatten(-3, -2),  # (..., P, 3)
        target[idx].flatten(-3, -2),  # (..., P, 3)
        msk.flatten(-2, -1) / target[idx][..., 2].flatten(-2, -1).clamp_min(1e-2),  # (..., P),
        trunc=trunc
    )  # (...,), (..., 3)

    # Mask out the invalid scale and shift
    valid = scale > 0  # (...,)
    scale = torch.where(valid, scale, 0)  # (...,)
    shift = torch.where(valid[..., None], shift, 0)  # (..., 3)
    if valid.sum() != valid.numel():
        print(f'Invalid scale and shift: {valid.sum()}/{valid.numel()}')

    # Apply the scale and shift to the prediction
    align = scale[..., None, None, None] * prediction + shift[..., None, None, :]  # (..., H, W, C)
    return align


class OptimalAffineInvariantMAELoss(nn.Module):
    def __init__(
        self,
        beta=0.0,
        align=32,
        trunc=1.0,
        sparsity_aware: bool = False, 
        depth_normalize: bool = True,
        improve_weighting: bool = False,
        alignment_type: str = 'z',
    ):
        super().__init__()

        self.beta = beta
        self.align = align
        self.trunc = trunc
        self.sparsity_aware = sparsity_aware
        self.depth_normalize = depth_normalize
        self.improve_weighting = improve_weighting

        if alignment_type == 'z':
            self.fn = align_xyz_scale_z_shift
        elif alignment_type == 'xyz':
            self.fn = align_xyz_scale_xyz_shift

    def forward(self, prediction, target, mask, weight=None, batch: dotdict = None):
        # Lazy import
        from easyvolcap.utils.moge_utils import mask_aware_nearest_resize

        # The input dimension must have (B, H, W) or (B, H, W, C)
        if mask.shape[-1] == 1:
            mask = mask.squeeze(-1)

        # Total loss
        total = 0
        global_scale = []

        for i in range(prediction.shape[0]):
            # Mask-aware nearest resize, we only need a small set of point to compute the alignment
            idx, msk = mask_aware_nearest_resize(mask[i], self.align, self.align)

            # Compute the optimal scale and shift
            scale, shift = self.fn(
                prediction[i][idx].flatten(-3, -2),  # (..., P, 3)
                target[i][idx].flatten(-3, -2),  # (..., P, 3)
                msk.flatten(-2, -1) / target[i][idx][..., 2].flatten(-2, -1).clamp_min(1e-2),  # (..., P),
                trunc=self.trunc
            )  # (...,), (..., 3)

            # Mask out the invalid scale and shift
            valid = scale > 0  # (...,)
            scale = torch.where(valid, scale, 0)  # (...,)
            shift = torch.where(valid[..., None], shift, 0)  # (..., 3)

            # Apply the scale and shift to the prediction
            prediction_align = scale[..., None, None, None] * prediction[i] + shift[..., None, None, :]  # (..., H, W, C)

            # Determine the weight
            if self.improve_weighting:
                weight = (
                    valid[..., None, None] & mask[i]).float() / torch.maximum(
                    target[i][..., 2], prediction_align[..., 2].detach()
                )[..., None]  # (..., H, W, 1)
            elif self.depth_normalize:
                weight = (
                    valid[..., None, None] & mask[i]
                ).float() / target[i][..., 2].clamp_min(1e-5)  # (..., H, W)
                weight = weight.clamp_max(
                    10.0 * weighted_mean(weight, mask[i], dim=(-2, -1), keepdim=True)
                )[..., None]  # (..., H, W, 1)
            else:
                weight = torch.ones_like(target[i][..., :1])  # (..., H, W, 1)

            # Compute the MAE loss
            loss = smooth(
                (prediction_align - target[i]).abs() * weight,
                beta=self.beta
            )

            # Add sparsity-aware regularization
            if self.sparsity_aware:
                sparsity = mask[i].float().mean(dim=(-2, -1)) / msk.float().mean(dim=(-2, -1))
                loss /= (sparsity + 1e-7)

            total += loss.mean()
            global_scale.append(scale)

        # Record the global scale
        batch.global_scale = torch.stack(global_scale, dim=0).detach()  # (B, ...)

        return total


class LocalOptimalAffineInvariantMAELoss(nn.Module):
    def __init__(
        self,
        beta=0.0,
        trunc=1.0,
        sparsity_aware: bool = False, 
        depth_normalize: bool = True,
        improve_weighting: bool = False,
    ):
        super().__init__()

        self.beta = beta
        self.trunc = trunc
        self.sparsity_aware = sparsity_aware
        self.depth_normalize = depth_normalize
        self.improve_weighting = improve_weighting

        self.fn = align_xyz_scale_xyz_shift

    def forward(
        self,
        src_xyz: torch.Tensor,
        tar_xyz: torch.Tensor,
        mask: torch.Tensor,
        H: int,
        W: int,
        focal: torch.Tensor,
        level: Literal[4, 16, 64],
        align: Literal[16, 8, 4],
        n_patchs: Literal[64, 256, 4096],
        global_scale: torch.Tensor = None,
        min_xyz_per_patch: int = 32,
    ):
        # Lazy import
        from easyvolcap.utils.moge_utils import mask_aware_nearest_resize
        from easyvolcap.utils.moge_utils import compute_anchor_sampling_weight

        # Shape things
        sh = src_xyz.shape[:-3]
        B = math.prod(sh)
        dtype, device = src_xyz.dtype, src_xyz.device
        mask, focal = mask.reshape(-1, H, W), focal.reshape(-1)
        src_xyz, tar_xyz = src_xyz.reshape(-1, H, W, 3), tar_xyz.reshape(-1, H, W, 3)
        if global_scale is not None: global_scale = global_scale.reshape(-1)

        # Compute the sampling radius
        r2d = math.ceil(0.5 / level * (H ** 2 + W ** 2) ** 0.5)  # scalar
        r3d = 0.5 / level / focal[:, None, None] * tar_xyz[..., 2]  # (B, H, W)

        # Compuet the sampling weight
        weight = compute_anchor_sampling_weight(
            tar_xyz, mask,
            r2d, r3d,
            n_anchors=64
        )  # (B, H, W)

        # Sample the random patch with weight
        indice = torch.where(mask)  # (B * H * W,)
        sample = torch.multinomial(
            weight[indice],
            B * n_patchs,
            replacement=True
        )  # (B * n_patchs,)
        pbidx, phidx, pwidx = [idx[sample] for idx in indice]

        # Get sampled patch indices
        pu, pv = torch.meshgrid(
            torch.arange(-r2d, r2d + 1, device=device), 
            torch.arange(-r2d, r2d + 1, device=device),
            indexing='ij'
        )  # (2 * r2d + 1, 2 * r2d + 1), (2 * r2d + 1, 2 * r2d + 1)
        pu, pv = pu + phidx[:, None, None], pv + pwidx[:, None, None]  # (B * n_patchs, Hp, Wp)
        pmsk = (pu >= 0) & (pu < H) & (pv >= 0) & (pv < W)  # (B * n_patchs, Hp, Wp)
        pu, pv = pu.clamp(0, H - 1), pv.clamp(0, W - 1)  # (B * n_patchs, Hp, Wp)

        # Get the target anchor points and 3d radius
        ptar_anc = tar_xyz[pbidx, phidx, pwidx]  # (B * n_patchs, 3)
        ptar_r3d = 0.5 / level / focal[pbidx] * ptar_anc[..., 2]  # (B * n_patchs,)
        # Get the target patch points
        ptar_xyz = tar_xyz[pbidx[:, None, None], pu, pv]
        ptar_dist = (ptar_xyz - ptar_anc[:, None, None, :]).norm(dim=-1)
        # Get the patch masks
        pmsk &= mask[pbidx[:, None, None], pu, pv]
        pmsk &= ptar_dist <= ptar_r3d[:, None, None]

        # Pick only non-empty (occupied) patch indices
        oidx = torch.where(
            pmsk.sum(dim=(-2, -1)) >= min_xyz_per_patch
        )

        # Return 0.0 loss if no patch is found
        if oidx[0].shape[0] == 0:
            return torch.tensor(
                0.0, dtype=dtype, device=device
            )

        # Finalize all patch variables
        pbidx, pu, pv = pbidx[oidx], pu[oidx], pv[oidx]
        pmsk = pmsk[oidx]  # (N, Hp, Wp)
        ptar_r3d = ptar_r3d[oidx]  # (N,)
        ptar_xyz = ptar_xyz[oidx]  # (N, Hp, Wp, 3)
        psrc_xyz = src_xyz[pbidx[:, None, None], pu, pv]  # (N, Hp, Wp, 3)

        # Align patch points
        # Mask-aware nearest resize, we only need a small set of point to compute the alignment
        idx, msk = mask_aware_nearest_resize(pmsk, align, align)

        # Compute the optimal scale and shift
        scale, shift = self.fn(
            ptar_xyz[idx].flatten(-3, -2),  # (..., P, 3)
            psrc_xyz[idx].flatten(-3, -2),  # (..., P, 3)
            msk.flatten(-2) / ptar_r3d[:, None].add(1e-7),  # (..., P),
            trunc=self.trunc
        )  # (...,), (..., 3)

        # Mask out the invalid scale and shift
        if global_scale is not None:
            sdiff = scale / global_scale[pbidx]  # (N,)
            valid = (sdiff > 0.1) & (sdiff < 10.0) & (global_scale[pbidx] > 0)
        else:
            valid = scale > 0  # (...,)
        scale = torch.where(valid, scale, 0)  # (...,)
        shift = torch.where(valid[..., None], shift, 0)  # (..., 3)

        # Update the patch mask
        pmsk = pmsk & valid[:, None, None]  # (N, Hp, Wp)

        # Apply the scale and shift to the prediction
        psrc_xyz = scale[..., None, None, None] * psrc_xyz + shift[..., None, None, :]  # (..., Hp, Wp, C)

        # Determine the weight
        if self.improve_weighting:
            weight = pmsk.float() / torch.maximum(
                ptar_xyz[..., 2], psrc_xyz[..., 2].detach()
            )  # (..., H, W, 1)
        elif self.depth_normalize:
            mean = harmonic_mean(tar_xyz[..., 2], mask, dim=(-2, -1))
            weight = pmsk.float() / ptar_xyz[..., 2].clamp_min(
                0.1 * mean[pbidx, None, None]
            )  # (..., H, W, 1)
        else:
            weight = torch.ones_like(ptar_xyz[..., 0])  # (..., H, W)

        # Compute the MAE loss
        loss = smooth(
            (psrc_xyz - ptar_xyz).abs() * weight[..., None],
            beta=self.beta
        ).mean(dim=(-3, -2, -1))

        # Add sparsity-aware regularization
        if self.sparsity_aware:
            sparsity = pmsk.float().mean(dim=(-2, -1)) / msk.float().mean(dim=(-2, -1))
            loss /= (sparsity + 1e-7)

        loss = torch.scatter_reduce(
            torch.zeros(B, dtype=dtype, device=device),
            dim=0,
            index=pbidx,
            src=loss,
            reduce='sum'
        ) / n_patchs
        loss = loss.mean()

        return loss


# Adapted from MoGe
# https://github.com/microsoft/MoGe
def angle_diff_vec3(
    v1: torch.Tensor,
    v2: torch.Tensor,
    eps: float = 1e-12
):
    return torch.atan2(
        torch.cross(
            v1, v2, dim=-1
        ).norm(dim=-1) + eps,
        (v1 * v2).sum(dim=-1)
    )


def xyz_normal_diff(
    src_xyz: torch.Tensor,
    tar_xyz: torch.Tensor,
    msk: torch.Tensor,
    min_angle: float = math.radians(1),
    max_angle: float = math.radians(90),
    beta: float = math.radians(3),
):
    """ Compute the normal loss between the source and target xyz coordinates.

    Args:
        src_xyz (torch.Tensor), (..., H, W, 3): source xyz coordinates
        tar_xyz (torch.Tensor), (..., H, W, 3): target xyz coordinates
        msk (torch.Tensor), (..., H, W): mask for valid normal
        min_angle (float): minimum angle in radians
        max_angle (float): maximum angle in radians
        beta (float): beta for the smooth function
    
    Returns:
        loss (torch.Tensor): normal loss
    """
    # Compute the normal map
    src_lt, src_rt = (
        src_xyz[..., :-1, :-1, :],
        src_xyz[..., :-1, 1:, :],
    )  # (B, H-1, W-1, 3)
    src_lb, src_rb = (
        src_xyz[..., 1:, :-1, :],
        src_xyz[..., 1:, 1:, :],
    )  # (B, H-1, W-1, 3)
    src_txl = torch.cross(src_rt - src_rb, src_lb - src_rb, dim=-1)
    src_lxb = torch.cross(src_lt - src_rt, src_rb - src_rt, dim=-1)
    src_bxr = torch.cross(src_lb - src_lt, src_rt - src_lt, dim=-1)
    src_rxt = torch.cross(src_rb - src_lb, src_lt - src_lb, dim=-1)

    # Compute the ground truth normal map
    tar_lt, tar_rt = (
        tar_xyz[..., :-1, :-1, :],
        tar_xyz[..., :-1, 1:, :],
    )  # (B, H-1, W-1, 3)
    tar_lb, tar_rb = (
        tar_xyz[..., 1:, :-1, :],
        tar_xyz[..., 1:, 1:, :],
    )  # (B, H-1, W-1, 3)
    tar_txl = torch.cross(tar_rt - tar_rb, tar_lb - tar_rb, dim=-1)
    tar_lxb = torch.cross(tar_lt - tar_rt, tar_rb - tar_rt, dim=-1)
    tar_bxr = torch.cross(tar_lb - tar_lt, tar_rt - tar_lt, dim=-1)
    tar_rxt = torch.cross(tar_rb - tar_lb, tar_lt - tar_lb, dim=-1)

    # Get the mask for valid normal
    msk_lt, msk_rt = (
        msk[..., :-1, :-1],
        msk[..., :-1, 1:]
    )  # (B, H-1, W-1, 1)
    msk_lb, msk_rb = (
        msk[..., 1:, :-1],
        msk[..., 1:, 1:]
    )  # (B, H-1, W-1, 1)
    msk_txl = msk_rt & msk_lb & msk_rb  # (B, H-1, W-1, 1)
    msk_lxb = msk_lt & msk_rb & msk_rt  # (B, H-1, W-1, 1)
    msk_bxr = msk_lb & msk_rt & msk_lt  # (B, H-1, W-1, 1)
    msk_rxt = msk_rb & msk_lt & msk_lb  # (B, H-1, W-1, 1)

    # Compute the normal loss
    loss = msk_txl * smooth(
        angle_diff_vec3(src_txl, tar_txl).clamp(min_angle, max_angle),
        beta=beta
    )
    loss += msk_lxb * smooth(
        angle_diff_vec3(src_lxb, tar_lxb).clamp(min_angle, max_angle),
        beta=beta
    )
    loss += msk_bxr * smooth(
        angle_diff_vec3(src_bxr, tar_bxr).clamp(min_angle, max_angle),
        beta=beta
    )
    loss += msk_rxt * smooth(
        angle_diff_vec3(src_rxt, tar_rxt).clamp(min_angle, max_angle),
        beta=beta
    )
    loss = loss.mean() / (4 * max(src_xyz.shape[-3:-1]))

    return loss


def align_dpt_scale_shift_ransac(
    src_dpt: torch.Tensor,
    tar_dpt: torch.Tensor,
    msk: Optional[torch.Tensor] = None,
    inverse: bool = False,
    eps: float = 1e-8,
):
    """ Fit a 1D RANSAC regression (possibly on inverted values) to
    recover scale (a) and bias (b) such that `res_dpt ≈ a * src_dpt + b`.
    NOTE: this function does not support batching, use it in a loop if needed.
    NOTE: the output of this function is numpy array.

    Args:
        src_dpt (np.ndarray or torch.Tensor): source depth or disparity.
        tar_dpt (np.ndarray or torch.Tensor): target depth or disparity.
        msk (optional): additional boolean mask.
        inverse (bool): if True, operate on inverses (1/depth as disparity).
        eps (float): epsilon for the mask.

    Returns:
        res_dpt (np.ndarray): scaled & biased depth, same shape as `src_dpt`.
    """
    # Convert to numpy
    if isinstance(src_dpt, torch.Tensor):
        src_dpt = src_dpt.cpu().numpy()
    if isinstance(tar_dpt, torch.Tensor):
        tar_dpt = tar_dpt.cpu().numpy()

    # Use float32 for the regression
    src_dpt = src_dpt.astype(np.float32).squeeze()
    tar_dpt = tar_dpt.astype(np.float32).squeeze()

    # Deal with the mask
    if msk is not None:
        if isinstance(msk, torch.Tensor):
            msk = msk.cpu().numpy()
        msk = (msk > 0).astype(bool).squeeze()
    else:
        msk = np.ones_like(tar_dpt, dtype=bool).squeeze()
    msk = msk & (tar_dpt > eps)

    if msk.sum() == 0:
        log(red("No valid points for RANSAC regression; returning original depth."))
        return src_dpt

    # Extract valid points for the regression
    source = src_dpt[msk]
    target = tar_dpt[msk]

    # Invert the target depth if needed
    if inverse:
        target = 1.0 / np.clip(target, eps, None)

    # Prepare the regression pipeline (linear fit)
    model = make_pipeline(
        PolynomialFeatures(degree=1, include_bias=False),
        RANSACRegressor(random_state=0)
    )

    try:
        # Fit RANSAC
        model.fit(source[..., None], target[..., None])
        # Extract the underlying LinearRegression estimator
        # coef_ is an array [a], intercept_ is scalar b
        est = model.named_steps['ransacregressor'].estimator_
        a = est.coef_.item()
        b = est.intercept_.item()
    except ValueError as e:
        # Often thrown when too few inliers; fallback to identity
        log(yellow(f"RANSAC fit failed ({e}); using fallback a=1, b=0."))
        a, b = 1.0, 0.0
    except Exception as e:
        # Catch other unexpected errors but still fallback
        log(red(f"Unexpected error in align_dpt_scale_shift_ransac: {e}"))
        a, b = 1.0, 0.0

    # Apply the linear transform to the full prediction map
    if a > 0:
        res_dpt = a * src_dpt + b
    else:
        res_dpt = src_dpt * (
            np.mean(target) / np.mean(source)
        )

    # If we inverted at the start, invert back
    if inverse:
        res_dpt = 1.0 / np.clip(res_dpt, eps, None)
    return res_dpt
