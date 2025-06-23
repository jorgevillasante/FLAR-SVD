# import gc
import sys

import torch
import torch.nn as nn


def lw_shrinkage(S):
    """Compute the Ledoit-Wolf shrinkage estimator of the covariance matrix."""
    n_features = S.shape[0]

    # Compute the shrinkage target (identity matrix)
    target = torch.eye(n_features, device="cuda") * torch.mean(torch.diag(S))

    # Compute the optimal shrinkage intensity
    shrinkage_intensity = torch.clip(
        1e10 * (torch.trace(S) - torch.trace(target)) / (torch.norm(S - target) ** 2),
        0,
        1,
    )

    # Compute the shrunk covariance matrix
    S_shrinked = (1 - shrinkage_intensity) * S + shrinkage_intensity * target

    return S_shrinked


class HookReg:
    def __init__(self, model: nn.Module):
        self.model = model

        self.profile = 0
        self.lwprofile = 0
        self.hooks = []

    def _hook_fn(self, lw):
        def get_scaling_mat(module, input, output):
            x = input[0].detach().float()

            out_prod = torch.matmul(x.transpose(1, 2), x)
            outpro_sum = torch.sum(out_prod, dim=0)
            lw_outpro_sum = lw_shrinkage(torch.mean(out_prod, dim=0))

            self.profile += outpro_sum
            self.lwprofile += lw_outpro_sum

            del module, input, x, out_prod, outpro_sum, output

        return get_scaling_mat

    def attach_hook(self, lw=False):
        self.hook = self.model.blocks[0].attn.qkv.register_forward_hook(
            self._hook_fn(lw)
        )

    def clear_hook(self):
        self.hook.remove()


def whitening(dev, raw_scale_diag):
    """
    The Cholesky decomposition is a convenient way to perform data whitening.
    The columns of the triangular matrix L form an orthogonal basis.
    Being triangular makes it easy and fast to handle and provides a numerically
    stable and efficient way to compute the inverse of the covariance matrix.
    """
    raw_scale_diag = raw_scale_diag.double().to(dev)
    # Perform Cholesky to obtain scaling matrix
    try:
        scale_diag = torch.linalg.cholesky(raw_scale_diag)
    except Exception:
        eigenvalues = torch.linalg.eigvalsh(raw_scale_diag)
        raw_scale_diag += (-eigenvalues[0] + 1e-6) * torch.eye(
            raw_scale_diag.shape[0]
        ).to(dev)
        try:
            scale_diag = torch.linalg.cholesky(raw_scale_diag)
        except Exception:
            # raise ValueError(f"Matrix not positive!: {name}")
            print("Warning: not positive!")
            # scale_diag = torch.linalg.qr(raw_scale_diag).R
            sys.exit()
        eigenvalues = None
        del eigenvalues
    raw_scale_diag = None
    del raw_scale_diag

    # Calculate the inverse of the scaling matrix
    try:
        scale_diag_inv = torch.linalg.inv(scale_diag)
    except Exception:
        scale_diag += 1e-6 * torch.eye(scale_diag.shape[0]).to(dev)
        try:
            scale_diag_inv = torch.linalg.inv(scale_diag)
        except Exception:
            # raise ValueError(f"Cannot invert matrix: {name}")
            print("Warning: not full rank!")
            sys.exit()

    return scale_diag.float().to(dev), scale_diag_inv.float().to(dev)


def low_rank_approx(module: nn.Linear, raw_profile, dev, rank):
    """Learning a low-rank decomposition for the given matrix.

    Args:
        module (nn.Linear): the given Linear module.
        raw_profile (nn.Tensor): the raw [activations] profile of the given module.
    """

    scale_diag, scale_diag_inv = whitening(dev, raw_profile)

    mat_scaled = torch.matmul(module.weight.detach(), scale_diag)
    # mat_scaled = mat_scaled.to(self.dev)

    u, s, vh = torch.linalg.svd(mat_scaled, full_matrices=False)
    s_val = torch.sqrt(torch.diag(s))  # half singular value
    mat_l = u @ s_val
    mat_l = mat_l[:, :rank].to(dev)
    mat_r = s_val @ torch.matmul(vh, scale_diag_inv)
    mat_r = mat_r[:rank, :].to(dev)

    return mat_l @ mat_r
