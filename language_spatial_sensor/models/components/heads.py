"""Output distribution heads: pooled vector → spatial prediction.

All heads share the same signature:
    forward(x, coord_scale, coord_shift) -> <NamedTuple subclass>

Registered in HEAD_REGISTRY so the active head is selected by cfg.head_type.

Current registrations:
    "gaussian_cholesky"  — 3-D Gaussian parameterised via Cholesky decomposition
    "gaussian_diagonal"  — axis-aligned Gaussian: μ and per-axis σ; L is diagonal

Loss functions are registered in LOSS_REGISTRY:
    "bbox_cdf"           — Gaussian box mass vs. uniform baseline + distance penalty
    "center_nll"         — Multivariate Gaussian NLL + L1 + Mahalanobis + volume penalty

Planned (not yet registered):
    "flow"               — normalising flow over 3-D space
    "dit"                — Diffusion Transformer outputting denoised samples
"""

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LSSConfig
from ..registry import HEAD_REGISTRY, LOSS_REGISTRY


# ── Output containers ─────────────────────────────────────────────────────────

class GaussianPrediction(NamedTuple):
    """Output of Gaussian heads (Cholesky or diagonal).

    Attributes:
        mu:  (B, 3)    predicted distribution centre in world frame
        L:   (B, 3, 3) lower-triangular Cholesky factor such that Σ = L @ Lᵀ, in world frame.
             For ``gaussian_diagonal``, L is diagonal with marginal stds on the diagonal.
    """
    mu: torch.Tensor   # (B, 3)
    L:  torch.Tensor   # (B, 3, 3)  lower triangular, diagonal > 0


# ── FiLM layer ────────────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation conditioned on coord_scale and coord_shift.

    Computes γ * x + β where γ and β are produced by a small MLP over the
    concatenated [coord_scale (3,), coord_shift (3,)] = (6,) conditioning vector.

    Initialized so γ ≈ 1 and β ≈ 0, making the layer a near-identity at startup
    and allowing gradients to smoothly learn the conditioning signal.

    Args:
        x:            (B, D)  pooled scene-text context vector
        coord_scale:  (B, 3)  per-axis region size (world → region denominator)
        coord_shift:  (B, 3)  region centre (world → region subtracted value)
    Returns:
        (B, D)  conditioned context vector
    """

    def __init__(self, cfg: LSSConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, cfg.film_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.film_hidden_dim, 2 * cfg.hidden_dim),
        )
        # Bias init: [ones(D), zeros(D)] → γ=1, β=0 at startup
        with torch.no_grad():
            nn.init.zeros_(self.mlp[-1].weight)
            bias = torch.zeros(2 * cfg.hidden_dim)
            bias[: cfg.hidden_dim] = 1.0
            self.mlp[-1].bias.copy_(bias)

    def forward(
        self,
        x: torch.Tensor,            # (B, D)
        coord_scale: torch.Tensor,  # (B, 3)
        coord_shift: torch.Tensor,  # (B, 3)
    ) -> torch.Tensor:              # (B, D)
        cond = torch.cat([coord_scale, coord_shift], dim=-1)  # (B, 6)
        out = self.mlp(cond)                                   # (B, 2D)
        D = x.size(1)
        gamma = out[:, :D]   # (B, D)
        beta  = out[:, D:]   # (B, D)
        return gamma * x + beta


# ── Heads ─────────────────────────────────────────────────────────────────────

@HEAD_REGISTRY.register("gaussian_cholesky")
class GaussianCholeskyHead(nn.Module):
    """Regress a full-covariance 3-D Gaussian via Cholesky decomposition.

    The model predicts in region-normalized coordinates; coord_scale and
    coord_shift are used to inverse-transform the output to world frame:
        mu_world = mu_region * coord_scale + coord_shift
        L_world  = diag(coord_scale) @ L_region

    The network outputs 9 raw scalars per sample:
        mu_raw  (3,) → mu  (no activation — unconstrained location)
        L_raw   (6,) → L lower-triangular 3×3:
            diag:        softplus(L_raw[0:3]) to enforce > 0
            lower tri:   L_raw[3:6]  (unconstrained)

    Positive definiteness of Σ = L @ Lᵀ is guaranteed as long as diag(L) > 0.
    """

    def __init__(self, cfg: LSSConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 9),  # 3 (mu) + 6 (L entries)
        )
        # σ floor in region frame: prevents variance collapse → caps the 1/σ²
        # blowup that creates the NLL right-tail on bad-confident samples.
        # At least 1e-4 for numerical stability even when the user sets 0.
        self.min_sigma: float = max(float(cfg.head_min_sigma), 1e-4)
        # lower-triangular index pairs (row, col) for a 3×3 matrix
        self.register_buffer(
            "tril_idx",
            torch.tensor([[0, 1, 2, 1, 2, 2],   # row indices
                          [0, 1, 2, 0, 0, 1]],   # col indices
                         dtype=torch.long),
        )

    def forward(
        self,
        x: torch.Tensor,            # (B, D)  FiLM-conditioned context
        coord_scale: torch.Tensor,  # (B, 3)  per-axis region size
        coord_shift: torch.Tensor,  # (B, 3)  region centre in world frame
    ) -> GaussianPrediction:
        out   = self.mlp(x)                    # (B, 9)
        mu_region = torch.tanh(out[:, :3])   # (B, 3)  bounded to (-1, 1) in region frame
        l_raw     = out[:, 3:]               # (B, 6)

        B = x.size(0)
        # Match MLP output dtype (e.g. bfloat16 under autocast); x may still be float32.
        L_region = torch.zeros(B, 3, 3, device=x.device, dtype=out.dtype)

        rows, cols = self.tril_idx   # each (6,)
        diag_mask  = rows == cols
        off_mask   = ~diag_mask

        # softplus(·) + min_sigma keeps the diagonal strictly positive and
        # (optionally) floored to prevent variance collapse on hard samples.
        L_region[:, rows[diag_mask], cols[diag_mask]] = (
            F.softplus(l_raw[:, diag_mask]).to(L_region.dtype) + self.min_sigma
        )
        L_region[:, rows[off_mask], cols[off_mask]] = l_raw[:, off_mask]

        # ── Inverse coordinate transform → world frame ───────────────────────
        # mu_world = mu_region * coord_scale + coord_shift
        # L_world  = diag(coord_scale) @ L_region
        #   i.e. row i of L scaled by coord_scale[:, i]
        # Correctness: Σ_world = S Σ_region S^T = (S L)(S L)^T = L_world L_world^T
        # L_world remains lower-triangular with positive diagonal (coord_scale > 0)
        mu_world = mu_region * coord_scale + coord_shift         # (B, 3)
        L_world  = coord_scale.unsqueeze(-1) * L_region          # (B, 3, 3)

        return GaussianPrediction(mu=mu_world, L=L_world)


@HEAD_REGISTRY.register("gaussian_diagonal")
class DiagonalGaussianHead(nn.Module):
    """Axis-aligned 3-D Gaussian: μ and per-axis σ in region frame, then world transform.

    Σ_region = diag(σ²); Cholesky factor L_region = diag(σ). Same world mapping as
    ``GaussianCholeskyHead`` so ``bbox_cdf_loss`` and downstream code stay unchanged.

    FiLM conditioning (if enabled) is applied globally in LSSModel before this head
    receives x — no duplicate conditioning here.
    """

    def __init__(self, cfg: LSSConfig) -> None:
        super().__init__()
        D = cfg.hidden_dim
        hidden_sizes = list(cfg.head_hidden_sizes)
        dropout_p = cfg.head_dropout
        use_layernorm = cfg.head_use_layernorm
        self.min_sigma = cfg.head_min_sigma

        def make_mlp(input_dim: int, hidden: list[int], output_dim: int) -> nn.Module:
            layers: list[nn.Module] = []
            last_dim = input_dim
            for h in hidden:
                layers.append(nn.Linear(last_dim, h))
                if use_layernorm:
                    layers.append(nn.LayerNorm(h))
                layers.append(nn.ReLU())
                if dropout_p > 0.0:
                    layers.append(nn.Dropout(dropout_p))
                last_dim = h
            layers.append(nn.Linear(last_dim, output_dim))
            return nn.Sequential(*layers)

        self.mu_head = make_mlp(D, hidden_sizes, 3)
        self.sigma_head = make_mlp(D, hidden_sizes, 3)

    def forward(
        self,
        x: torch.Tensor,            # (B, D)  FiLM-conditioned context (or raw if use_film false)
        coord_scale: torch.Tensor,  # (B, 3)  per-axis region size
        coord_shift: torch.Tensor,  # (B, 3)  region centre in world frame
    ) -> GaussianPrediction:
        mu_region = torch.tanh(self.mu_head(x))
        sigma_region = self.min_sigma + F.softplus(self.sigma_head(x))

        L_region = torch.diag_embed(sigma_region)

        mu_world = mu_region * coord_scale + coord_shift
        L_world = coord_scale.unsqueeze(-1) * L_region

        return GaussianPrediction(mu=mu_world, L=L_world)


# ── Loss functions ────────────────────────────────────────────────────────────


@LOSS_REGISTRY.register("bbox_cdf")
def bbox_cdf_loss(
    pred: GaussianPrediction,
    bbox: torch.Tensor,           # (B, 6) [x_min, y_min, z_min, x_max, y_max, z_max]
    scale_factor: torch.Tensor,   # (B, 3) per-axis region size (used as uniform baseline)
    lambda_dist: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Likelihood-ratio box term plus distance to bbox centre.

    Uses per-axis Gaussian mass in the target AABB (via ``erf``), a uniform baseline
    over the region ``scale_factor`` (volume = ``scale_factor.prod(-1)``), and an L2
    penalty on ``mu - bbox_centre``.  Marginal σ are taken from ``L``'s diagonal
    (exact for diagonal ``L``; approximate for full Cholesky).
    """
    mu = pred.mu                                                   # (B, 3)
    sigma = pred.L.diagonal(dim1=-2, dim2=-1).clamp(min=eps)      # (B, 3)

    bbox_min = bbox[:, :3]
    bbox_max = bbox[:, 3:]

    bbox_center = (bbox_max + bbox_min) * 0.5                      # (B, 3)
    bbox_size   = (bbox_max - bbox_min).clamp(min=eps)             # (B, 3)

    # Gaussian mass inside target AABB (product of per-axis integrals)
    z_min = (bbox_min - mu) / (sigma * math.sqrt(2.0))
    z_max = (bbox_max - mu) / (sigma * math.sqrt(2.0))
    p_per_axis = 0.5 * (torch.erf(z_max) - torch.erf(z_min))
    p_gauss = p_per_axis.prod(dim=-1).clamp(min=eps)               # (B,)

    # Uniform baseline: probability a uniform draw over the region hits the box
    box_volume  = bbox_size.prod(dim=-1)
    room_volume = scale_factor.prod(dim=-1)
    p_uniform   = (box_volume / room_volume).clamp(min=eps)        # (B,)

    # Likelihood-ratio loss: -log(P_gauss / P_uniform)
    nll_loss = -torch.log(p_gauss) + torch.log(p_uniform)

    # L2 distance to bbox centre (unnormalized — scale invariance comes from coord frame)
    dist_loss = ((mu - bbox_center) ** 2).sum(dim=-1)

    return (nll_loss + lambda_dist * dist_loss).mean()


@LOSS_REGISTRY.register("center_nll")
def center_nll_loss(
    pred: GaussianPrediction,
    target_xyz: torch.Tensor,        # (B, 3) object center in world frame
    lambda_l1: float = 1.0,
    lambda_mahal: float = 0.1,
    lambda_vol: float = 0.01,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Multivariate Gaussian NLL on object centre with conformal-set auxiliary losses.

    Components:
        1. **Gaussian NLL** — standard negative log-likelihood under the predicted
           multivariate normal :math:`\\mathcal{N}(\\mu, \\Sigma)` with
           :math:`\\Sigma = L L^{\\mathsf T}`.
        2. **L1 loss** — :math:`\\|\\mu - y\\|_1`, encourages accurate mean predictions.
        3. **Mahalanobis penalty** — :math:`\\sqrt{(y-\\mu)^{\\mathsf T} \\Sigma^{-1} (y-\\mu)}`,
           penalises large normalised errors for better-calibrated uncertainty.
        4. **Covariance volume penalty** — :math:`\\log\\det\\Sigma`, encourages smaller
           predicted confidence ellipsoids (tighter conformal sets).
    """
    mu = pred.mu                                                    # (B, 3)
    L  = pred.L                                                     # (B, 3, 3)

    diff = target_xyz - mu                                          # (B, 3)

    # log det(Sigma) = 2 * sum(log(diag(L)))
    log_diag = L.diagonal(dim1=-2, dim2=-1).clamp(min=eps).log()   # (B, 3)
    log_det  = 2.0 * log_diag.sum(dim=-1)                          # (B,)

    # Quadratic form via triangular solve: L z = diff  =>  z^T z = diff^T Sigma^{-1} diff
    z    = torch.linalg.solve_triangular(
        L, diff.unsqueeze(-1), upper=False,
    ).squeeze(-1)                                                   # (B, 3)
    quad = (z * z).sum(dim=-1)                                      # (B,)

    # 1) Gaussian NLL
    k   = mu.shape[-1]                                              # 3
    nll = 0.5 * (log_det + quad + k * math.log(2.0 * math.pi))     # (B,)

    # 2) L1 loss
    l1 = diff.abs().sum(dim=-1)                                     # (B,)

    # 3) Mahalanobis distance
    mahal = quad.clamp(min=1e-12).sqrt()                            # (B,)

    # 4) Covariance volume penalty (log_det already computed)

    loss = nll + lambda_l1 * l1 + lambda_mahal * mahal + lambda_vol * log_det
    return loss.mean()


def marginal_cdf_at_gt(
    pred: GaussianPrediction,
    target_bbox_world: torch.Tensor,  # (B, 6) [x_min,y_min,z_min,x_max,y_max,z_max] world
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-axis Gaussian mass over the target AABB (product-of-marginals box diagnostic).

    For each axis *k*, with marginal :math:`X_k \\sim \\mathcal{N}(\\mu_k, \\sigma_k^2)` and
    :math:`\\sigma_k^2 = \\Sigma_{kk}` from :math:`\\Sigma = L L^{\\mathsf T}`:

        ``axis_mass[:, k]`` = :math:`\\mathbb{P}(\\mathrm{bbox}^{\\min}_k < X_k < \\mathrm{bbox}^{\\max}_k)`

    ``mass_prod`` = product over axes (independent-marginals approximation; matches the
    axis factorisation used in ``bbox_cdf_loss``).

    Returns:
        axis_mass:  (B, 3) per-axis interval masses in :math:`(0, 1)`.
        mass_prod:  (B,)   product over axes.
    """
    mu = pred.mu
    # Diagonal of L @ Lᵀ is the row-wise sum of squares of L — no need to materialise Σ.
    var = (pred.L ** 2).sum(dim=-1).clamp(min=1e-12)   # (B, 3)
    sigma = var.sqrt()

    bbox_min = target_bbox_world[:, :3]
    bbox_max = target_bbox_world[:, 3:]
    dist = torch.distributions.Normal(mu, sigma)
    axis_mass = dist.cdf(bbox_max) - dist.cdf(bbox_min)
    mass_prod = axis_mass.prod(dim=-1)
    return axis_mass, mass_prod
