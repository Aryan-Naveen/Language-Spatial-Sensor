"""Shared GMM components for 3D grounding baselines.

Provides:
    GMMPrediction   — NamedTuple output format
    GMMCholeskyHead — K-component GMM head from a pooled feature vector
    gmm_nll_loss    — NLL loss with MAP distance penalty and entropy regularizer
    sample_from_gmm — draw samples from a single GMM instance (no batch dim)
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GMMPrediction(NamedTuple):
    logits: Tensor  # (B, K)       raw logits; softmax → mixture weights π_k
    mu:     Tensor  # (B, K, 3)    component means in world frame
    L:      Tensor  # (B, K, 3, 3) lower-triangular Cholesky factors; Σ_k = L_k @ L_k^T


class GMMCholeskyHead(nn.Module):
    """K-component full-covariance GMM head.

    Projects a pooled D-dim feature vector to K mixture components, each
    parameterized by a logit (weight), a 3D mean, and a 3×3 lower-triangular
    Cholesky factor.

    Total output scalars per sample: K * (1 logit + 3 μ + 6 L entries) = K * 10.

    Coordinate handling
    -------------------
    Means are predicted in a region-normalized frame and then mapped to world
    frame via the affine ``coord_scale`` / ``coord_shift`` tensors:

        μ_region = tanh(raw_μ)            # bounded in (-1, 1)³
        μ_world  = μ_region * coord_scale + coord_shift

    Cholesky factors are transformed the same way:

        L_world[k] = diag(coord_scale) @ L_region[k]

    The diagonal of each L is constrained positive via softplus + ``min_sigma``.
    Off-diagonal entries are unconstrained.

    Args:
        in_dim:         Input feature dimension D.
        num_components: Number of GMM components K (default 5).
        hidden_dim:     Hidden dimension of the two-layer projection MLP.
        min_sigma:      Floor added to L diagonal after softplus (region frame).
        dropout:        Dropout probability in the MLP.
    """

    def __init__(
        self,
        in_dim: int,
        num_components: int = 5,
        hidden_dim: int = 256,
        min_sigma: float = 0.001,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.K = num_components
        self.min_sigma = min_sigma

        out_dim = num_components * 10  # 1 logit + 3 mu + 6 L per component
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        # Pre-compute lower-triangular indices (3×3) as a persistent buffer.
        tril_rows, tril_cols = torch.tril_indices(3, 3)
        self.register_buffer("tril_rows", tril_rows)  # (6,)
        self.register_buffer("tril_cols", tril_cols)  # (6,)

    def forward(
        self,
        x: Tensor,           # (B, D)
    ) -> GMMPrediction:
        B, K = x.shape[0], self.K
        # Under autocast(bf16), Linear outputs are bf16 while upstream features can be fp32
        # (e.g. scene_feat from bf16 * float masks). Force fp32 so buffers and MVN loss agree.
        raw = self.mlp(x).float()                  # (B, K*10)
        raw = raw.view(B, K, 10)                   # (B, K, 10)

        raw_logits = raw[..., 0]                   # (B, K)
        raw_mu     = raw[..., 1:4]                 # (B, K, 3)
        raw_l      = raw[..., 4:]                  # (B, K, 6)

        # ── Means (region frame → world frame) ────────────────────────────────
        mu_world = torch.tanh(raw_mu)             # (B, K, 3), bounded in (-1,1)

        # ── Cholesky factors ───────────────────────────────────────────────────
        L_world = torch.zeros(B, K, 3, 3, device=x.device, dtype=raw.dtype)
        # Fill lower-triangular entries
        L_world[:, :, self.tril_rows, self.tril_cols] = raw_l
        # Constrain diagonal to be strictly positive
        diag_mask = self.tril_rows == self.tril_cols
        diag_r    = self.tril_rows[diag_mask]
        L_world[:, :, diag_r, diag_r] = (
            F.softplus(raw_l[:, :, diag_mask]) + self.min_sigma
        )

        return GMMPrediction(logits=raw_logits, mu=mu_world, L=L_world)


def gmm_nll_loss(
    pred: GMMPrediction,
    target_xyz: Tensor,       # (B, 3) ground-truth position, world frame
    lambda_dist: float = 0.1,
    entropy_reg: float = 0.01,
) -> Tensor:
    """Negative log-likelihood of the target under the GMM.

    Loss = NLL + distance penalty on MAP component + entropy regularizer.

    NLL
    ---
    Computed via logsumexp for numerical stability:

        NLL = -logsumexp_k( log π_k + log N(target; μ_k, Σ_k) )

    where log N is evaluated with ``torch.distributions.MultivariateNormal``.

    Distance penalty
    ----------------
    Encourages the highest-weight component to be near the target:

        dist_loss = ||μ_MAP - target||²

    Entropy regularizer
    -------------------
    Maximises mixture entropy to prevent all weight collapsing onto one component:

        entropy_bonus = -entropy_reg * H(π)   (added to loss, so minimising loss maximises H)

    Note: H(π) = -Σ_k π_k log π_k  ≥ 0, so subtracting it as a penalty minimises the loss
    when entropy is large, i.e. we ADD -entropy_reg * (-H) = entropy_reg * H to the objective.
    Wait — we want to encourage high entropy, so we subtract entropy from the loss:
        loss += -entropy_reg * H(π)
    Since H ≥ 0, this adds a negative term → encourages higher entropy.
    """
    B, K = pred.logits.shape
    device = pred.logits.device

    log_weights = F.log_softmax(pred.logits, dim=-1)  # (B, K)

    # Per-component log-probabilities at target
    # target_xyz: (B, 3) → expand to (B, K, 3) for batched MVN
    target_exp = target_xyz.unsqueeze(1).expand(B, K, 3)  # (B, K, 3)

    log_probs = torch.stack([
        torch.distributions.MultivariateNormal(
            pred.mu[:, k, :],           # (B, 3)
            scale_tril=pred.L[:, k, :, :],  # (B, 3, 3)
        ).log_prob(target_exp[:, k, :])  # (B,)
        for k in range(K)
    ], dim=1)  # (B, K)

    # NLL via logsumexp
    log_mixture = torch.logsumexp(log_weights + log_probs, dim=-1)  # (B,)
    nll = -log_mixture.mean()

    # Distance penalty: MAP component (highest weight) vs target
    with torch.no_grad():
        k_star = pred.logits.argmax(dim=-1)  # (B,)
    mu_map = pred.mu[torch.arange(B, device=device), k_star]  # (B, 3)
    dist_loss = ((mu_map - target_xyz) ** 2).sum(dim=-1).mean()

    # Entropy regularizer (negative entropy added to loss → minimising loss maximises entropy)
    weights = F.softmax(pred.logits, dim=-1)           # (B, K)
    entropy = -(weights * log_weights).sum(dim=-1).mean()  # scalar ≥ 0
    entropy_loss = -entropy_reg * entropy

    return nll + lambda_dist * dist_loss + entropy_loss


def sample_from_gmm(
    logits: Tensor,  # (K,)  raw logits (single sample, no batch dim)
    mu:     Tensor,  # (K, 3)
    L:      Tensor,  # (K, 3, 3)
    n_samples: int,
) -> Tensor:         # (n_samples, 3)
    """Draw ``n_samples`` points from a GMM (single instance, no batch dim).

    1. Sample component indices proportional to softmax(logits).
    2. For each index k, draw one point from MultivariateNormal(μ_k, scale_tril=L_k).

    Returns a (n_samples, 3) float32 tensor on the same device as ``mu``.
    """
    weights = torch.softmax(logits, dim=-1)  # (K,)
    component_ids = torch.multinomial(weights, num_samples=n_samples, replacement=True)  # (n_s,)

    # Sample per component
    samples = torch.zeros(n_samples, 3, device=mu.device, dtype=mu.dtype)
    for k in range(logits.shape[0]):
        mask = component_ids == k
        n_k  = mask.sum().item()
        if n_k == 0:
            continue
        dist = torch.distributions.MultivariateNormal(mu[k], scale_tril=L[k])
        samples[mask] = dist.sample((n_k,))

    return samples
