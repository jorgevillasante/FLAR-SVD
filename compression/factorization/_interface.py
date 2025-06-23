from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class FactorizedMatrix:
    mat_l: torch.Tensor = None
    mat_r: torch.Tensor = None
    eq_rank: int = 0
    active_rank: int = 0


class BaseFactorization:
    def __init__(self):
        self.scaling_dict = {}
        self.input_shapes = {}

    def compute_scaling(self, model, name_omit, calib_data, mixup_fn, white_list=[]):
        print("\nNo scaling method implemented.")
        pass

    def factorize_matrix(self, matrix, name, rank=-1, ratio=-1) -> FactorizedMatrix:
        # function that applies the svd technique to a single matrix and return the
        # compressed one (+ meta data?)
        print(f"Factorizing {name} matrix")
        if rank == -1 and ratio == -1:
            print(f"Warning: {name} rank or ratio must be defined!")
            return
        elif rank != -1 and ratio != -1:
            print(
                f"Warning: {name} rank and ratio are both defined, "
                "only one can be used at a time!"
            )
            return
        dev = torch.device(torch.cuda.current_device())
        eq_rank = (
            matrix.shape[0] * matrix.shape[1] // (matrix.shape[0] + matrix.shape[1])
        )
        if rank == 0:
            rank = eq_rank
        elif ratio != -1:
            # rank = int(np.round(eq_rank * ratio))
            rank = int(eq_rank * ratio)
        elif rank > eq_rank:
            print(f"Warning: {name} rank is larger than equivalent rank!")
            return
        fact_matrix = self._factorize_matrix(
            matrix=matrix, name=name, eq_rank=eq_rank, rank=rank, dev=dev
        )
        return fact_matrix

    def _factorize_matrix(self, matrix, name, eq_rank, rank, dev) -> FactorizedMatrix:
        # function that applies the svd technique to a single matrix and return the
        # compressed one (+ meta data?)
        raise NotImplementedError("Subclasses should implement this method.")

    def create_factorized_sequential(
        self, factorized_matrix: FactorizedMatrix, original_module
    ) -> nn.Module:
        dev = original_module.weight.device
        module_l = nn.Linear(
            original_module.in_features,
            factorized_matrix.active_rank,
            bias=False,
        )
        module_r = nn.Linear(
            factorized_matrix.active_rank,
            original_module.out_features,
            # bias=(original_module.bias is not None),
            bias=False,
        )
        module_l = module_l.to(dev)
        module_r = module_r.to(dev)

        weight_l, weight_r = factorized_matrix.mat_l, factorized_matrix.mat_r
        module_l.weight.data.copy_(weight_r[: factorized_matrix.active_rank, :].to(dev))
        module_r.weight.data.copy_(weight_l[:, : factorized_matrix.active_rank].to(dev))
        # if hasattr(original_module, "bias") and original_module.bias is not None:
        #     module_r.bias.data.copy_(original_module.bias)
        module = weight_l = weight_r = None
        del weight_l, weight_r, module

        with torch.cuda.device(dev):
            torch.cuda.empty_cache()

        # return nn.Sequential(module_l, module_r).to(dev)
        return SeqSVD(module_l, module_r, original_module.bias if hasattr(original_module, "bias")
                      else None).to(dev)

    def factorize_model(self, uncom_model, rank_dict, name_omit) -> dict:
        """
        Apply low-rank decomposition to the model in place. Note that name omit
        is supported implicitly as removing or not mentioning something in the
        compression ratio dict will resul in it not being compressed.

        Args:
            name (str): module name
            module (nn.Linear): the given Linear module
            raw_profile (dict): the raw profile of the given module
        """
        print("\nApplying factorization")
        dev = torch.device(torch.cuda.current_device())
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()

        model = uncom_model.eval().to(dev)
        copied_modules = {
            name: module_sub
            for name, module_sub in model.named_modules()
            if all(omit not in name for omit in name_omit)
            and isinstance(module_sub, nn.Linear)
        }
        print(rank_dict)
        for name, module_sub in copied_modules.items():
            # condition for not applying low rank
            if (
                module_sub.out_features < 10
                or name in name_omit
                or rank_dict[name] == -1
                or rank_dict[name] == 1.0
            ):
                continue

            factorized = self.factorize_matrix(
                matrix=module_sub.weight,
                rank=rank_dict[name] if isinstance(rank_dict[name], int) else -1,
                ratio=rank_dict[name] if isinstance(rank_dict[name], float) else -1,
                name=name,
            )

            svd_replacement = self.create_factorized_sequential(
                factorized_matrix=factorized, original_module=module_sub
            )

            print(f"Applying low rank on {name:^10}, rank {rank_dict[name]}")

            base, localname = model, name
            while "." in localname:
                prefix, localname = localname.split(".", 1)
                base = base.__getattr__(prefix)

            setattr(base, localname, svd_replacement)


class SeqSVD(nn.Module):
    def __init__(self, mod_a, mod_b, bias=None):
        super().__init__()
        self.mod_a = mod_a
        self.mod_b = mod_b
        self.bias = bias

    def forward(self, x):
        x = self.mod_a(x)
        x = self.mod_b(x)
        if self.bias is not None:
            x += self.bias
        return x


class Hookstuff:
    # this is one unified hook to obtain the scalings for everybody.
    def __init__(self, model: nn.Module, name_omit=[], dump_shape=False, white_list=[]):
        self.model = model
        self.name_omit = name_omit
        self.white_list = white_list
        self.dump_shape = dump_shape

        self.profile = {}
        self.input_shape = {}
        self.hooks = []
        self.cp_modules = reversed(
            [
                (name, module_sub)
                for name, module_sub in model.named_modules()
                # if all(omit not in name for omit in name_omit)
                if isinstance(module_sub, nn.Linear)
            ]
        )

    def _hook_fn(self, layer_name):
        def get_scaling_mat(module, input, output):
            pass

        return get_scaling_mat

    def _register_hooks_recursive(self, cp_modules: dict, prefix=""):
        for name, layer in cp_modules:
            if isinstance(layer, nn.Linear):
                if any(n in name for n in self.name_omit):
                    continue
                if layer.out_features < 10:
                    continue
                try:
                    if self.white_list and not any(n in name for n in self.white_list):
                        continue
                except Exception:
                    if self.white_list and not any(
                        n in name for n, _ in self.white_list
                    ):
                        continue
                hook = layer.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)

    def attach_hooks(self):
        self._register_hooks_recursive(self.cp_modules)

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
