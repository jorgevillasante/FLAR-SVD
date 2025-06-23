import gc
import sys

import torch

from ._interface import BaseFactorization
from ._interface import FactorizedMatrix
from ._interface import Hookstuff


def whitening(dev, raw_scale, name):
    """
    The Cholesky decomposition is a convenient way to perform data whitening.
    The columns of the triangular matrix L form an orthogonal basis.
    Being triangular makes it easy and fast to handle and provides a numerically
    stable and efficient way to compute the inverse of the covariance matrix.
    """
    raw_scale_diag = raw_scale[name]
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
            print(f"Warning: {name} is not positive!")
            # scale_diag = torch.linalg.qr(raw_scale_diag).R
            sys.exit()
        eigenvalues = None
        del eigenvalues
    raw_scale_diag = None
    del raw_scale_diag

    # Calculate the inverse of the scaling matrix
    try:
        # scale_diag_inv = torch.linalg.inv(scale_diag)
        scale_diag_inv = torch.linalg.pinv(scale_diag)
    except Exception:
        # scale_diag += 1e-4 * torch.eye(scale_diag.shape[0]).to(dev)
        scale_diag = torch.where(
            torch.isnan(scale_diag),
            torch.tensor(1e-10, device=scale_diag.device),
            scale_diag,
        )
        try:
            scale_diag_inv = torch.linalg.inv(scale_diag)
        except Exception:
            # raise ValueError(f"Cannot invert matrix: {name}")
            print(f"Warning: {name} is not full rank!")
            sys.exit()

    return scale_diag.float().to(dev), scale_diag_inv.float().to(dev)


class Cholesky_Hook(Hookstuff):
    def _hook_fn(self, layer_name, last_feat=False):
        def get_scaling_mat(module, input, output):
            x = input[0].detach().float()
            if x.dim() > 3:
                x.reshape(x.shape[0], -1, x.shape[-1])
            elif x.dim() == 2:
                x = x.unsqueeze(0)
            if self.dump_shape:
                self.input_shape[layer_name] = list(x.shape)
                self.input_shape[layer_name].extend([module.out_features, 0])
                return
            if last_feat:
                if "head" in layer_name:
                    self.model.last_feat = x.clone()
                return
            out_prod = torch.matmul(x.transpose(1, 2), x)
            outpro_sum = torch.sum(out_prod, dim=0)

            if layer_name not in self.profile:  # First run through each layer
                self.profile[layer_name] = outpro_sum
            else:
                self.profile[layer_name] += outpro_sum

            del module, input, x, out_prod, outpro_sum, output

        return get_scaling_mat


class SVD_LLMFactorization(BaseFactorization):
    def compute_scaling(self, model, name_omit, calib_data, mixup_fn, white_list=[]):
        print("\nObtaining activations")
        dev = torch.device(torch.cuda.current_device())

        with torch.cuda.device(dev):
            torch.cuda.empty_cache()

        model = model.eval().to(dev)
        extractor = Cholesky_Hook(model, name_omit, white_list=white_list)
        extractor.attach_hooks()

        with torch.no_grad():
            for data, target in calib_data:
                model_inps, targets = mixup_fn(data, target)
                model_inps = model_inps.to(dev)
                model(model_inps)
                del model_inps, targets

        extractor.clear_hooks()
        for key, value in extractor.profile.items():
            self.scaling_dict[key] = value
        del extractor

        shapes_getter = Cholesky_Hook(model, name_omit, True, white_list=white_list)
        shapes_getter.attach_hooks()
        dummy_input = torch.randn(20, 3, 224, 224).to(dev)
        model(dummy_input)
        shapes_getter.clear_hooks()
        for key, value in shapes_getter.input_shape.items():
            self.input_shapes[key] = value
        del shapes_getter, dummy_input

        gc.collect()

        return

    def _factorize_matrix(self, matrix, eq_rank, rank, name, dev):
        raw_profile = self.scaling_dict
        scale_diag, scale_diag_inv = whitening(dev, raw_profile, name)
        if rank == 0:
            rank = eq_rank
        elif rank > eq_rank:
            print(f"Warning: {name} rank is larger than equivalent rank!")
            return

        mat_scaled = torch.matmul(matrix.to(dev), scale_diag)

        u, s, vh = torch.linalg.svd(mat_scaled, full_matrices=False)
        s_val = torch.sqrt(torch.diag(s))  # half singular value
        mat_l = u @ s_val
        mat_l = mat_l[:, :rank].to(dev)
        mat_r = s_val @ torch.matmul(vh, scale_diag_inv)
        mat_r = mat_r[:rank, :].to(dev)

        return FactorizedMatrix(
            mat_l=mat_l.cpu(),  # Left singular vectors
            mat_r=mat_r.cpu(),  # Right singular vectors
            eq_rank=eq_rank,  # Equivalent rank
            active_rank=rank,  # Active rank
        )
