import gc

import torch
from torch import nn

from ._interface import BaseFactorization
from ._interface import FactorizedMatrix
from ._interface import Hookstuff


class FWSVD_Hook(Hookstuff):
    def _hook_fn(self, layer_name):
        def get_scaling_mat(module, input, output):
            x = input[0].detach().float()
            self.input_shape[layer_name] = list(x.shape)
            self.input_shape[layer_name].extend([module.out_features, 0])
            return

        return get_scaling_mat


class FWSVDFactorization(BaseFactorization):
    def __init__(self, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.loss_type = "CE"

    def compute_scaling(self, model, name_omit, calib_data, mixup_fn, white_list=[]):
        print("\nCollecting Fisher importance...")
        dev = torch.device(torch.cuda.current_device())
        loss_fn = nn.CrossEntropyLoss()

        with torch.cuda.device(dev):
            torch.cuda.empty_cache()

        model = model.eval().to(dev)

        for data, target in calib_data:
            model_inputs, target_mix = mixup_fn(data, target)
            model_inputs, target_mix = model_inputs.to(dev), target_mix.to(dev)
            out = model(model_inputs)
            if self.loss_type == "CE":
                loss = loss_fn(out, target_mix)
            else:
                loss = out
            loss.mean().backward()
            del model_inputs, target_mix
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    if name not in self.scaling_dict:
                        tmp = module.weight.grad.detach()
                        self.scaling_dict[name] = tmp.pow(2).mean(0)
                    else:
                        tmp = module.weight.grad.detach()
                        self.scaling_dict[name] += tmp.pow(2).mean(0)
            model.zero_grad()
            with torch.cuda.device(torch.cuda.current_device()):
                torch.cuda.empty_cache()

        for key, val in self.scaling_dict.items():
            self.scaling_dict[key] = (val / len(calib_data)).sqrt()

        shapes_getter = FWSVD_Hook(model, name_omit, True, white_list=white_list)
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
        dev = torch.device(torch.cuda.current_device())
        raw_profile = self.scaling_dict[name]
        scale_diag = raw_profile**self.alpha + 1e-6

        if rank == 0:
            rank = eq_rank
        elif rank > eq_rank:
            print(f"Warning: {name} rank is larger than equivalent rank!")
            return

        mat_scaled = matrix * scale_diag.view(1, -1)

        u, s, vh = torch.svd_lowrank(mat_scaled, q=rank)
        s_val = torch.sqrt(torch.diag(s))  # half singular value
        vh = (vh / scale_diag.view(-1, 1)).t()

        s_val = torch.sqrt(torch.diag(s))  # half singular value
        mat_l = u @ s_val
        mat_l = mat_l[:, :rank].to(dev)
        mat_r = s_val @ vh
        mat_r = mat_r[:rank, :].to(dev)

        return FactorizedMatrix(
            mat_l=mat_l.cpu(),  # Left singular vectors
            mat_r=mat_r.cpu(),  # Right singular vectors
            eq_rank=eq_rank,  # Equivalent rank
            active_rank=rank,  # Active rank
        )
