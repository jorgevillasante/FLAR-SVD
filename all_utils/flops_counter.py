# !usr/bin/env python
"""
 Description  :
 Version      : 1.0
 Author       : MrYXJ
 Mail         : yxj2017@gmail.com
 Github       : https://github.com/MrYxJ
 Date         : 2023-08-19 10:28:55
 LastEditTime : 2023-09-07 23:39:17
 Copyright (C) 2023 mryxj. All rights reserved.
"""
import torch
import torch.nn as nn
from calflops.calculate_pipline import CalFlopsPipline
from calflops.utils import flops_to_string
from calflops.utils import generate_transformer_input
from calflops.utils import macs_to_string
from calflops.utils import params_to_string

DEFAULT_PRECISION = 2


def calculate_flops(
    model, input_shape=(1, 3, 224, 224), detail=False, print_results=True, args=[], kwargs={}
):

    """Returns the total floating-point operations, MACs, and parameters of a model."""

    forward_mode = "forward"
    include_backPropagation = False
    compute_bp_factor = 2.0
    # print_results = True
    output_as_string = True
    output_precision = 2
    output_unit = None
    ignore_modules = None
    transformer_tokenizer = None

    assert isinstance(model, nn.Module), "model must be a PyTorch module"
    model.to("cuda").eval()

    is_Transformer = True if "transformers" in str(type(model)) else False

    calculate_flops_pipline = CalFlopsPipline(
        model=model,
        include_backPropagation=include_backPropagation,
        compute_bp_factor=compute_bp_factor,
    )
    calculate_flops_pipline.start_flops_calculate(ignore_list=ignore_modules)

    device = next(model.parameters()).device
    model = model.to(device)

    if input_shape is not None:
        assert len(args) == 0 and len(kwargs) == 0
        assert type(input_shape) is tuple, "input_shape must be a tuple"
        assert len(input_shape) >= 1, "input_shape must have at least one element"

        if transformer_tokenizer is None:  # model is not transformers model
            assert is_Transformer is False
            try:
                input = torch.ones(()).new_empty(
                    (*input_shape,),
                    dtype=next(model.parameters()).dtype,
                    device=device,
                )
            except StopIteration:
                input = torch.ones(()).new_empty((*input_shape,))
            args = [input]
        else:
            assert len(input_shape) == 2
            kwargs = generate_transformer_input(
                input_shape=input_shape,
                model_tokenizer=transformer_tokenizer,
                device=device,
            )
    else:
        assert transformer_tokenizer or (len(args) > 0 or len(kwargs) > 0)
        if transformer_tokenizer:
            kwargs = generate_transformer_input(
                input_shape=None, model_tokenizer=transformer_tokenizer, device=device
            )

    if kwargs:
        for key, value in kwargs.items():
            kwargs[key] = value.to(device)

        if forward_mode == "forward":
            _ = model(*args, **kwargs)
        if forward_mode == "generate":
            _ = model.generate(*args, **kwargs)
    else:
        for index in range(len(args)):
            args[index] = args[index].to(device)

        if forward_mode == "forward":
            _ = model(*args)
        if forward_mode == "generate":
            _ = model.generate(*args)

    flops = calculate_flops_pipline.get_total_flops()
    macs = calculate_flops_pipline.get_total_macs()
    params = calculate_flops_pipline.get_total_params()

    if print_results:
        calculate_flops_pipline.print_model_pipline(
            units=output_unit, precision=output_precision, print_detailed=detail
        )
    calculate_flops_pipline.stop_flops_calculate()
    # calculate_flops_pipline.end_flops_calculate()

    if include_backPropagation:
        flops = flops * (1 + compute_bp_factor)
        macs = macs * (1 + compute_bp_factor)

    if output_as_string:
        return (
            flops_to_string(flops, units=output_unit, precision=output_precision),
            macs_to_string(macs, units=output_unit, precision=output_precision),
            params_to_string(params, units=output_unit, precision=output_precision),
        )

    return flops, macs, params, model
