import torch

from ._interface import BaseFactorization
from ._interface import FactorizedMatrix


class SVDFactorization(BaseFactorization):
    def _factorize_matrix(self, matrix, name, eq_rank, rank, dev):
        if rank == 0:
            rank = eq_rank
        elif rank > eq_rank:
            print(f"Warning: {name} rank is larger than equivalent rank!")
            return

        mat_scaled = matrix.to(dev)

        u, s, vh = torch.linalg.svd(mat_scaled, full_matrices=False)
        s_val = torch.sqrt(torch.diag(s))  # half singular value
        mat_l = u @ s_val
        mat_l = mat_l[:, :rank]
        mat_r = s_val @ vh
        mat_r = mat_r[:rank, :]

        return FactorizedMatrix(
            mat_l=mat_l.cpu().float(),  # Left singular vectors
            mat_r=mat_r.cpu().float(),  # Right singular vectors
            eq_rank=eq_rank,  # Equivalent rank
            active_rank=rank,  # Active rank
        )
