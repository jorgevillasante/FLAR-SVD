import importlib
import time

import all_utils.dist_training as dist_utils
import torch.nn as nn
from torch.utils import data

from .factorization._interface import BaseFactorization
from .search._interface import BaseSearch


def get_lr_module(module_name: str):
    parent_module = "compression.factorization"
    module = importlib.import_module(f"{parent_module}.{module_name}")
    return getattr(module, f"{module_name.upper()}Factorization")


def get_search_module(module_name: str):
    parent_module = "compression.search"
    module = importlib.import_module(f"{parent_module}.{module_name}")
    return getattr(module, f"{module_name.upper()}Search")


class ModelFactorizer:
    """
    Implementation of Low-Rank Decomposition for compressing the model's weights.
    """

    def __init__(
        self,
        svd_method: str,
        svd_method_args: dict,
        search_method: str,
        search_method_args: dict,
    ) -> None:
        self.svd_method = svd_method
        self.svd_method_args = svd_method_args
        self.search_method = search_method
        self.search_method_args = search_method_args

    def factorize_and_search(
        self,
        model: nn.Module,
        calib_data: data.DataLoader,
        eval_data: data.DataLoader,
        mixup_fn,
        name_omit: list = [],
        blockwise_search: bool = False,
    ):
        """
        This function applies SVD decomposition to the models layers.
        """
        print(f"{' Compressing model ':=^115}")
        start_time = time.time()
        LrdModule = get_lr_module(self.svd_method)
        SearchMethod = get_search_module(self.search_method)

        lrd_method: BaseFactorization = LrdModule(**self.svd_method_args)
        lrd_method.compute_scaling(
            model=model, name_omit=name_omit, calib_data=calib_data, mixup_fn=mixup_fn
        )

        search_method: BaseSearch = SearchMethod(
            name_omit=name_omit,
            eval_data=eval_data,
            mixup_fn=mixup_fn,
            **self.search_method_args,
        )
        search_method.initialize_search(lrd_method, model)
        if not blockwise_search:
            layerwise_rank_dict = search_method.search(model)
        else:
            layerwise_rank_dict = search_method.search_blockwise(
                model,
                stage_name=self.search_method_args["stage_name_in_current_model"],
                calib_data=calib_data,
            )

        # inplace factorization based on layerwise rank dict
        lrd_method.factorize_model(model, layerwise_rank_dict, name_omit=name_omit)

        dist_utils.dist_barrier()
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n\nNumber of parameters in model: {num_params}")
        print(f"\nTook {(time.time() - start_time)/60:.3} min to compress model\n")
        dist_utils.dist_barrier()
