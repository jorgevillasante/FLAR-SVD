import gc

import torch

from ._interface import BaseFactorization
from ._interface import FactorizedMatrix
from ._interface import Hookstuff


class ASVD_Hook(Hookstuff):
    def _hook_fn(self, layer_name, last_feat=False):
        def get_scaling_mat(module, input, output):
            x = input[0].detach().float()
            if self.dump_shape:
                self.input_shape[layer_name] = list(x.shape)
                self.input_shape[layer_name].extend([module.out_features, 0])
                return
            if x.dim() > 3:
                x = x.reshape(x.size(0), -1, x.size(-1))
            elif x.dim() == 2:
                x = x.unsqueeze(0)
            outpro_sum = x.abs().amax(dim=-2).detach().amax(-2)

            if layer_name not in self.profile:  # First run through each layer
                self.profile[layer_name] = outpro_sum
            else:
                self.profile[layer_name] += outpro_sum

            del module, input, outpro_sum

        return get_scaling_mat


class ASVDFactorization(BaseFactorization):
    def __init__(self, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def compute_scaling(self, model, name_omit, calib_data, mixup_fn, white_list=[]):
        print("\nObtaining activations")
        dev = torch.device(torch.cuda.current_device())

        with torch.cuda.device(dev):
            torch.cuda.empty_cache()

        model = model.eval().to(dev)
        extractor = ASVD_Hook(model, name_omit, white_list=white_list)
        extractor.attach_hooks()

        with torch.no_grad():
            for samples, targets in calib_data:
                model_inps, targets = mixup_fn(samples, targets)

                model_inps = model_inps.to(dev)
                model(model_inps)
                del model_inps, targets

        extractor.clear_hooks()
        for key, value in extractor.profile.items():
            self.scaling_dict[key] = value

        del extractor

        shapes_getter = ASVD_Hook(model, name_omit, True, white_list=white_list)
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
        raw_profile = self.scaling_dict[name]
        scale_diag = raw_profile**self.alpha + 1e-6

        mat_scaled = matrix * scale_diag.view(1, -1)

        u, s, vh = torch.svd_lowrank(mat_scaled, q=rank)
        s_val = torch.sqrt(torch.diag(s))  # half singular value
        vh = (vh / scale_diag.view(-1, 1)).t()

        s_val = torch.sqrt(torch.diag(s))  # half singular value
        mat_l = u @ s_val
        mat_r = s_val @ vh
        mat_l = mat_l[:, :rank]
        mat_r = mat_r[:rank, :]

        return FactorizedMatrix(
            mat_l=mat_l.cpu(),  # Left singular vectors
            mat_r=mat_r.cpu(),  # Right singular vectors
            eq_rank=eq_rank,  # Equivalent rank
            active_rank=rank,  # Active rank
        )
