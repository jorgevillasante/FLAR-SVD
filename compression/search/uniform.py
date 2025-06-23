import torch
from torch import nn

from ._interface import BaseSearch


class UNIFORMSearch(BaseSearch):
    def __init__(self, eval_data, mixup_fn, name_omit=[], ratio_target=0.5, stage_name_in_current_model="stages"):
        self.eval_data = eval_data
        self.name_omit = name_omit
        self.mixup_fn = mixup_fn
        self.dev = torch.device(torch.cuda.current_device())
        self.stage_name_in_current_model = stage_name_in_current_model
        # sensitivity dict needed for ASVD search
        self.sensitivity_dict = {}
        self.lrd_method = None
        self.ratio_target = ratio_target

    def search(self, model: nn.Module):
        default_param_ratio = 1.0
        layer_compression_dict = {
            name: self.ratio_target for name, _ in model.named_modules()
        }
        # replace name omit layer compression with 1.0
        for name in layer_compression_dict.keys():
            if any(n in name for n in self.name_omit):
                layer_compression_dict[name] = default_param_ratio

        return layer_compression_dict

    def search_blockwise(self, model: nn.Module, stage_name: str, calib_data=None):
        default_param_ratio = 1.0
        compression_dict = {
            name: default_param_ratio for name, _ in model.named_modules()
        }
        blocks, blocks_layer_names = self.get_model_blocks(model, stage_name)
        for block, block_layer_names in zip(blocks, blocks_layer_names):
            self.lrd_method.compute_scaling(
                model,
                name_omit=self.name_omit,
                calib_data=calib_data,
                mixup_fn=self.mixup_fn,
                white_list=block_layer_names,
            )
            for layer_name, _ in block_layer_names:
                compression_dict[layer_name] = self.ratio_target
                layer_name_within_block = max(
                    (
                        name
                        for name in dict(block.named_modules()).keys()
                        if name in layer_name
                    ),
                    key=len,
                )
                smodule: nn.Linear = dict(block.named_modules())[
                    layer_name_within_block
                ]
                factorized_matrix = self.lrd_method.factorize_matrix(
                    smodule.weight, ratio=self.ratio_target, name=layer_name
                )
                smodule.weight.data.copy_(
                    factorized_matrix.mat_l.to(self.dev)
                    @ factorized_matrix.mat_r.to(self.dev)
                )
        return compression_dict
