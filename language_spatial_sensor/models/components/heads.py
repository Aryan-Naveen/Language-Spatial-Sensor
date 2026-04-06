"""Output distribution heads: pooled vector → spatial prediction.

All heads share the same signature:
    forward(x, coord_scale, coord_shift) -> <NamedTuple subclass>

Registered in HEAD_REGISTRY so the active head is selected by cfg.head_type.

Current registrations:
    "gaussian_cholesky"  — 3-D Gaussian parameterised via Cholesky decomposition

Planned (not yet registered):
    "flow"               — normalising flow over 3-D space
    "dit"                — Diffusion Transformer outputting denoised samples
"""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LSSConfig
from ..registry import HEAD_REGISTRY


# ── Output containers ─────────────────────────────────────────────────────────

class GaussianPrediction(NamedTuple):
    """Output of GaussianCholeskyHead.

    Attributes:
        mu:  (B, 3)    predicted distribution centre in world frame
        L:   (B, 3, 3) lower-triangular Cholesky factor such that Σ = L @ Lᵀ, in world frame
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
        out   = self.mlp(x)          # (B, 9)
        mu_region = out[:, :3]       # (B, 3)  region-normalized mean
        l_raw     = out[:, 3:]       # (B, 6)

        B = x.size(0)
        L_region = torch.zeros(B, 3, 3, device=x.device, dtype=x.dtype)

        rows, cols = self.tril_idx   # each (6,)
        diag_mask  = rows == cols
        off_mask   = ~diag_mask

        # softplus(·) + ε ensures diagonal stays strictly positive
        L_region[:, rows[diag_mask], cols[diag_mask]] = (
            F.softplus(l_raw[:, diag_mask]) + 1e-4
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


# ── Loss ──────────────────────────────────────────────────────────────────────

def bbox_cdf_loss(
    pred: GaussianPrediction,
    bbox: torch.Tensor,  # (B, 6)  [x_min,y_min,z_min,x_max,y_max,z_max] world frame
) -> torch.Tensor:
    """Negative log probability that the predicted Gaussian falls within the target bbox.

    Uses per-axis marginal CDFs (product of three 1-D Normal CDFs) as a
    closed-form differentiable approximation to the true multivariate box integral.

    Args:
        pred: GaussianPrediction with mu (B,3) and L (B,3,3) in world frame.
        bbox: (B, 6) target bounding box in world frame.

    Returns:
        Scalar loss (mean over batch).
    """
    mu    = pred.mu                                       # (B, 3)
    sigma = pred.L.diagonal(dim1=-2, dim2=-1)             # (B, 3) marginal std devs

    bbox_min = bbox[:, :3]                                # (B, 3)
    bbox_max = bbox[:, 3:]                                # (B, 3)

    dist = torch.distributions.Normal(mu, sigma.clamp(min=1e-6))
    p_per_axis = dist.cdf(bbox_max) - dist.cdf(bbox_min)  # (B, 3)
    p_box = p_per_axis.prod(dim=-1)                        # (B,)

    return -torch.log(p_box.clamp(min=1e-8)).mean()
