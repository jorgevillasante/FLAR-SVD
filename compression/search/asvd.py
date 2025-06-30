from copy import deepcopy

import all_utils.dist_training as dist_utils
import torch
from torch import nn

from ..factorization._interface import BaseFactorization
from ._interface import BaseSearch


class ASVDSearch(BaseSearch):
    def __init__(self, eval_data, mixup_fn, name_omit=[], ratio_target=0.5):
        self.eval_data = tuple(data for data in eval_data)
        self.name_omit = name_omit
        self.mixup_fn = mixup_fn
        # sensitivity dict needed for ASVD search
        self.sensitivity_dict = {}
        self.lrd_method = None
        self.ratio_target = ratio_target

    def initialize_search(
        self, lrd_method: BaseFactorization, model: nn.Module, spec_tensor=None
    ):
        self.lrd_method = lrd_method
        layer_sensitivity = self._get_layer_sensitivity(model, spec_tensor)
        self.sensitivity_dict = layer_sensitivity

    def search(self, model: nn.Module):
        module_dict = {name: module for name, module in model.named_modules()}

        default_param_ratio = 1.0

        # create and sort sensitivity list required for search
        sensitivity_list = []
        for layername, v in self.sensitivity_dict.items():
            for param_ratio, ppl in v.items():
                if param_ratio >= 1:
                    continue
                sensitivity_list.append((layername, param_ratio, ppl))
        sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])

        # binary search start
        high = len(sorted_sensitive_list) - 1
        low = 0

        while low < high:
            mid = (low + high) // 2
            layers_min_ratio = {
                layername: default_param_ratio
                for layername in self.sensitivity_dict.keys()
            }
            for layername, param_ratio, ppl in sorted_sensitive_list[mid:]:
                layers_min_ratio[layername] = min(
                    layers_min_ratio[layername], param_ratio
                )
            tot_params = 0
            compress_params = 0

            for layername, param_ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * param_ratio
            now_ratio = compress_params / tot_params
            if now_ratio > self.ratio_target:
                high = mid
            else:
                low = mid + 1

        print("=== Searching done, decomposing layers... ===")
        layers_min_ratio = {
            layername: default_param_ratio for layername in self.sensitivity_dict.keys()
        }
        for layername, param_ratio, ppl in sorted_sensitive_list[mid:]:
            if layers_min_ratio[layername] is None:
                layers_min_ratio[layername] = param_ratio
            else:
                layers_min_ratio[layername] = min(
                    layers_min_ratio[layername], param_ratio
                )
        # return dict with per layer compression ratio
        return layers_min_ratio

    def _eval_model(self, model):
        model.eval()
        dev = torch.device(torch.cuda.current_device())
        model = model.to(dev)
        model = model.eval()
        loss_fc = nn.CrossEntropyLoss()
        ppl = torch.tensor(0.0, device=dev)
        with torch.no_grad():
            for batch_idx, (samples, targets) in enumerate(self.eval_data):
                model_inputs, labels = samples.to(dev), targets.to(dev)
                logits = model(model_inputs)
                del model_inputs
                loss = loss_fc(logits, labels)
                ppl += loss
                # ppl = torch.exp(loss)
                del loss, logits

                with torch.cuda.device(torch.cuda.current_device()):
                    torch.cuda.empty_cache()

        dist_utils.dist_barrier()
        if dist_utils.is_dist_initialized():
            ppl = dist_utils.sync_tensor(ppl, "mean")
        dist_utils.dist_barrier()

        return ppl.item()

    def _get_layer_sensitivity(self, model: nn.Module, spec_tensor=None):
        copied_modules = {
            name: module_sub for name, module_sub in model.named_modules()
        }
        sensitivity_dict = {}
        for name, module_sub in copied_modules.items():
            if isinstance(module_sub, nn.Linear):
                if any(n in name for n in self.name_omit):
                    continue
                if module_sub.out_features < 10:
                    continue  # for some head matrix, such as image-text match head

                print(f"Evaluating sensitivity for layer {name}")
                cp_model = deepcopy(model)

                base, localname = cp_model, name
                while "." in localname:
                    prefix, localname = localname.split(".", 1)
                    base = base.__getattr__(prefix)

                sensitivity_dict[name] = {}
                # only do factorize once to avoid overhead
                factorized_matrix = self.lrd_method.factorize_matrix(
                    name=name,
                    matrix=module_sub.weight,
                    ratio=1.0,  # spec_tensor[name],
                )
                mat_l = factorized_matrix.mat_l.clone()
                mat_r = factorized_matrix.mat_r.clone()
                for ratio in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    eval_rank = int(factorized_matrix.eq_rank * ratio)
                    factorized_matrix.mat_l = mat_l[:, :eval_rank]
                    factorized_matrix.mat_r = mat_r[:eval_rank, :]
                    factorized_matrix.active_rank = eval_rank
                    seq_replacement = self.lrd_method.create_factorized_sequential(
                        factorized_matrix=factorized_matrix, original_module=module_sub
                    )
                    setattr(base, localname, seq_replacement)
                    metric = self._eval_model(cp_model)
                    sensitivity_dict[name][ratio] = metric
        return sensitivity_dict