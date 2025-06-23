import gc

import torch

from ._interface import BaseFactorization
from ._interface import FactorizedMatrix
from ._interface import Hookstuff
from .svd_llm import whitening


def lw_shrinkage(S):
    """Compute the Ledoit-Wolf shrinkage estimator of the Gramm matrix."""
    n_features = S.shape[0]

    # Compute the shrinkage target (identity matrix)
    target = torch.eye(n_features, device="cuda") * torch.mean(torch.diag(S))

    # Approx the optimal shrinkage intensity
    shrinkage_intensity = torch.clip(
        1e10 * (torch.trace(S) - torch.trace(target)) / (torch.norm(S - target) ** 2),
        0,
        1,
    )
    # print(f"Shrinkage intensity: {shrinkage_intensity.item()}")
    # Compute the shrunk Gramm matrix
    S_shrinked = (1 - shrinkage_intensity) * S + shrinkage_intensity * target

    return S_shrinked


class Flar_Hook(Hookstuff):
    def _hook_fn(self, layer_name, last_feat=False):
        def get_scaling_mat(module, input, output):
            x = input[0].detach().clone().double()
            if x.dim() > 3:
                x = x.reshape(x.shape[0], -1, x.shape[-1])
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
            outpro_sum = torch.mean(out_prod, dim=0)
            outpro_sum = lw_shrinkage(outpro_sum)
            outpro_sum = outpro_sum.cpu()

            if layer_name not in self.profile:  # First run through each layer
                self.profile[layer_name] = outpro_sum
            else:
                self.profile[layer_name] += outpro_sum

            del module, input, x, out_prod, outpro_sum, output

        return get_scaling_mat


class FLAR_SVDFactorization(BaseFactorization):
    def compute_scaling(self, model, name_omit, calib_data, mixup_fn, white_list=[]):
        print("\nObtaining activations for FLAR SVD")
        dev = torch.device(torch.cuda.current_device())

        with torch.cuda.device(dev):
            torch.cuda.empty_cache()

        model = model.eval().to(dev)
        extractor = Flar_Hook(model, name_omit, white_list=white_list)
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

        shapes_getter = Flar_Hook(model, name_omit, True, white_list=white_list)
        shapes_getter.attach_hooks()
        dummy_input = torch.randn(20, 3, 224, 224).to(dev)
        model(dummy_input)
        shapes_getter.clear_hooks()
        for key, value in shapes_getter.input_shape.items():
            self.input_shapes[key] = value
        del shapes_getter, dummy_input

        gc.collect()

        return

    def _factorize_matrix(self, matrix, name, eq_rank, rank, dev):
        raw_profile = self.scaling_dict
        scale_diag, scale_diag_inv = whitening(dev, raw_profile, name)
        scale_diag, scale_diag_inv = scale_diag.float(), scale_diag_inv.float()
        if rank == 0:
            rank = eq_rank
        elif rank > eq_rank:
            print(f"Warning: {name} rank is larger than equivalent rank!")
            return

        mat_scaled = torch.matmul(matrix.to(dev), scale_diag)

        u, s, vh = torch.linalg.svd(mat_scaled, full_matrices=False)
        s_val = torch.sqrt(torch.diag(s))  # half singular value
        mat_l = u @ s_val
        mat_l = mat_l[:, :rank]
        mat_r = s_val @ torch.matmul(vh, scale_diag_inv)
        mat_r = mat_r[:rank, :]

        return FactorizedMatrix(
            mat_l=mat_l.cpu().float(),  # Left singular vectors
            mat_r=mat_r.cpu().float(),  # Right singular vectors
            eq_rank=eq_rank,  # Equivalent rank
            active_rank=rank,  # Active rank
        )
