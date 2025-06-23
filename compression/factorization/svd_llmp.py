import gc
import sys

import torch

from ._interface import BaseFactorization
from ._interface import FactorizedMatrix
from ._interface import Hookstuff


def lw_shrinkage(S):
    """Compute the Ledoit-Wolf shrinkage estimator of the Gramm matrix."""
    n_features = S.shape[0]

    # Compute the shrinkage target (identity matrix)
    target = torch.eye(n_features, device="cuda") * torch.mean(torch.diag(S))

    # Approx the optimal shrinkage intensity
    shrinkage_intensity = torch.clip(
        (torch.trace(S) - torch.trace(target)) / (torch.norm(S - target) ** 2), 0, 1
    )

    # Compute the shrunk Gramm matrix
    S_shrinked = (1 - shrinkage_intensity) * S + shrinkage_intensity * target

    return S_shrinked


def whitening(dev, raw_scale, name):
    """
    The Cholesky decomposition is a convenient way to perform data whitening.
    The columns of the triangular matrix L form an orthogonal basis.
    Being triangular makes it easy and fast to handle and provides a numerically
    stable and efficient way to compute the inverse of the covariance matrix.
    """
    raw_scale_diag = raw_scale[name]
    raw_scale_diag = raw_scale_diag.double().to(dev)
    # raw_scale_diag += 1e-1 * torch.eye(raw_scale_diag.shape[0]).to(dev)
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
        scale_diag_inv = torch.linalg.inv(scale_diag)
    except Exception:
        scale_diag += 1e-6 * torch.eye(scale_diag.shape[0]).to(dev)
        try:
            scale_diag_inv = torch.linalg.inv(scale_diag)
        except Exception:
            # raise ValueError(f"Cannot invert matrix: {name}")
            print(f"Warning: {name} is not full rank!")
            sys.exit()

    return scale_diag.float().to(dev), scale_diag_inv.float().to(dev)


class Cholesky_Hook(Hookstuff):
    def __init__(self, model, name_omit=[], dump_shape=False):
        super().__init__(model, name_omit, dump_shape)
        self.profile_out = {}
        self.output_shape = {}

    def _hook_fn(self, layer_name, last_feat=False):
        def get_scaling_mats(module, input, output):
            def get_scaling_mat_in(module, input, output):
                x = input[0].detach().float()
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
                # outpro_sum = torch.mean(out_prod, dim=0)
                # outpro_sum = lw_shrinkage(outpro_sum)

                if layer_name not in self.profile:  # First run through each layer
                    self.profile[layer_name] = outpro_sum
                else:
                    self.profile[layer_name] += outpro_sum

                del module, input, x, out_prod, outpro_sum, output

            def get_scaling_mat_out(module, input, output):
                x = output.detach().float()
                if self.dump_shape:
                    self.output_shape[layer_name] = list(x.shape)
                    self.output_shape[layer_name].extend([module.out_features, 0])
                    return
                if last_feat:
                    if "head" in layer_name:
                        self.model.last_feat = x.clone()
                    return
                # if "blocks.11.mlp.fc2" in layer_name:
                print(output.shape)
                print(input[0].shape)
                print(layer_name)
                # try:
                x_center = x - torch.mean(x, dim=1, keepdim=True)
                covariance = (
                    torch.matmul(x_center.transpose(1, 2), x_center) / x_center.shape[1]
                )
                outpro_sum = torch.sum(covariance, dim=0)
                # out_prod = torch.matmul(x.transpose(1, 2), x)
                # outpro_sum = torch.sum(out_prod, dim=0)
                # outpro_sum = torch.mean(out_prod, dim=0)
                # outpro_sum = lw_shrinkage(outpro_sum)
                # except:
                #     outpro_sum = out_prod = torch.matmul(x.transpose(0, 1), x)
                #     #torch.eye(x.shape[0], x.shape[1])

                if layer_name not in self.profile_out:  # First run through each layer
                    self.profile_out[layer_name] = outpro_sum
                else:
                    self.profile_out[layer_name] += outpro_sum

                del module, input, x, covariance, outpro_sum, output

            get_scaling_mat_in(module, input, output)
            get_scaling_mat_out(module, input, output)

        return get_scaling_mats


class SVD_LLMPFactorization(BaseFactorization):
    def compute_scaling(self, model, name_omit, calib_data, mixup_fn):
        print("\nObtaining activations")
        dev = torch.device(torch.cuda.current_device())

        with torch.cuda.device(dev):
            torch.cuda.empty_cache()

        model = model.eval().to(dev)
        extractor = Cholesky_Hook(model, name_omit)
        extractor.attach_hooks()

        with torch.no_grad():
            for data, target in calib_data:
                model_inps, targets = mixup_fn(data, target)

                model_inps = model_inps.to(dev)
                model(model_inps)
                del model_inps, targets

        extractor.clear_hooks()
        self.scaling_dict = extractor.profile
        self.scaling_dict_out = extractor.profile_out
        del extractor

        # if dist.is_initialized():
        #     dist_utils.dist_barrier()
        #     for key, val in raw_profile.items():
        #         raw_profile[key] = val = dist_utils.sync_tensor(
        #             val / len(self.calib_data), "mean"
        #         )
        #         raw_profile[key] = lw_shrinkage(val)
        #     dist_utils.dist_barrier()

        shapes_getter = Cholesky_Hook(model, name_omit, True)
        shapes_getter.attach_hooks()
        dummy_input = torch.randn(20, 3, 224, 224).to(dev)
        model(dummy_input)
        shapes_getter.clear_hooks()
        self.input_shapes = shapes_getter.input_shape
        del shapes_getter, dummy_input

        gc.collect()

        return

    def _factorize_matrix(self, matrix, eq_rank, rank, name, dev):
        raw_profile = self.scaling_dict
        raw_profile_out = self.scaling_dict_out
        scale_diag, scale_diag_inv = whitening(dev, raw_profile, name)
        scale_diag_out, scale_diag_out_inv = whitening(dev, raw_profile_out, name)
        if rank == 0:
            rank = eq_rank
        elif rank > eq_rank:
            print(f"Warning: {name} rank is larger than equivalent rank!")
            return

        mat_scaled = torch.matmul(
            torch.matmul(scale_diag_out_inv, matrix.to(dev)), scale_diag
        )

        u, s, vh = torch.linalg.svd(mat_scaled, full_matrices=False)
        s_val = torch.sqrt(torch.diag(s))  # half singular value
        mat_l = torch.matmul(scale_diag_out, u) @ s_val
        mat_l = mat_l[:, :rank].to(dev)
        mat_r = s_val @ torch.matmul(vh, scale_diag_inv)
        mat_r = mat_r[:rank, :].to(dev)

        return FactorizedMatrix(
            mat_l=mat_l.cpu(),  # Left singular vectors
            mat_r=mat_r.cpu(),  # Right singular vectors
            eq_rank=eq_rank,  # Equivalent rank
            active_rank=rank,  # Active rank
        )
