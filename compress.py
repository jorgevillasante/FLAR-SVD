# --------------------------------------------------------
# Main (train/validate)
# Adapted from LeViT and Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# --------------------------------------------------------
import argparse
import random
from pathlib import Path

import models

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from all_utils.flops_counter import calculate_flops
from compression.svd_core import ModelFactorizer
from data.datasets import build_dataset
from engine import evaluate
from timm.data import Mixup
from timm.models import create_model
from torch.utils.data import Subset
from all_utils.benchmark_pip import benchmark

import utils


def get_args_parser():
    parser = argparse.ArgumentParser(
        "FLAR-SVD compression and evaluation script", add_help=False
    )
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--calib_bs", default=128, type=int, help="Batchsize for data based svd calibration.")
    parser.add_argument("--epochs", default=300, type=int)

    # Model parameters
    parser.add_argument(
        "--model",
        default="deit_base_patch16_224.fb_in1k",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--input-size", default=224, type=int, help="images input size")

    # Augmentation parameters
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.3,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train-interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )
    parser.add_argument("--repeated-aug", action="store_true")
    parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")
    parser.set_defaults(repeated_aug=True)

    # Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # Dataset parameters
    parser.add_argument(
        "--data-path",
        default="datasets/imagenet-1k",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--data-set",
        default="IMNET",
        choices=["CIFAR", "IMNET", "INAT", "INAT19"],
        type=str,
        help="dataset type",
    )
    parser.add_argument(
        "--inat-category",
        default="name",
        choices=[
            "kingdom",
            "phylum",
            "class",
            "order",
            "supercategory",
            "family",
            "genus",
            "name",
        ],
        type=str,
        help="semantic granularity",
    )
    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=457, type=int) #34,280, 457
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--save_freq", default=1, type=int, help="frequency of model saving"
    )

    # SVD parameters
    parser.add_argument(
        "--svd_method",
        default="flar_svd",
        choices=["flar_svd", "svd_llm", "pela", "svd", "fwsvd", "asvd"],
        type=str,
        help="SVD compression method to use.",
    )
    parser.add_argument(
        "--search_method",
        default="flar_svd",
        choices=["flar_svd", "asvd", "uniform"],
        type=str,
        help="SVD compression method to use.",
    )
    parser.add_argument(
        "--name_omit", default=["norm", "head", "patch_embed", "downsample"], type=list
    #     "-no", "--name_omit", default=["norm", "head", "patch_embed", "downsample"], action="append", type=str
    )
    # SVD settings
    parser.add_argument("--compression_target", default=0.5, type=float, help="compression target ratio")
    parser.add_argument(
        "--target_metric",
        default="params",
        choices=["params", "flops"],
        type=str,
        help="Metric to optimize based on target.",
    )
    # search settings
    parser.add_argument("--blockwise", action="store_true", default=False, help="whether to do blockwise search or not.")
    # FLAR SVD search settings
    parser.add_argument("--error_threshold", default=None, type=float, help="error threshold for flar_svd search")
    parser.add_argument("--stage_name", default="layers", type=str, help="Name of stages in model to compress.")
    parser.add_argument("--progressive_comp", action="store_true", help="Use progressive compression")
    # ASVD settings
    parser.add_argument("--asvd_alpha", default=1.0, type=float, help="alpha for ASVD")

    return parser


def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    # Standard datasets
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    # Calibration datasets as subset of train dataset
    indices = list(range(len(dataset_train)))
    random.shuffle(indices)
    total_calib_set = indices[:4608] # 1152 4608 16896
    split = 512
    svd_calib_indices, svd_eval_indices = total_calib_set[split:], total_calib_set[:split]
    dataset_svd_calib = Subset(dataset_train, svd_calib_indices)
    dataset_svd_eval = Subset(dataset_train, svd_eval_indices)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_svd_calib = torch.utils.data.DistributedSampler(
        dataset_svd_calib,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
    )
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_svd_eval = torch.utils.data.SequentialSampler(dataset_svd_eval)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_svd_calib = torch.utils.data.DataLoader(
        dataset_svd_calib,
        sampler=sampler_svd_calib,
        batch_size=args.calib_bs,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    data_loader_svd_eval = torch.utils.data.DataLoader(
        dataset_svd_eval,
        sampler=sampler_svd_eval,
        batch_size=128,  # lower to 64 if OOM
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    mixup_fn = Mixup(
        mixup_alpha=2.0,  # Higher alpha for more aggressive mixup
        cutmix_alpha=2.0,  # Higher alpha for more aggressive cutmix
        cutmix_minmax=(0.2, 0.8),  # Wider range for cutmix
        prob=1.0,  # Always apply mixup/cutmix
        switch_prob=1.0,  # Always switch between mixup and cutmix
        mode="elem",  # Apply mixup/cutmix to individual elements
        label_smoothing=0.0,  # No label smoothing
        num_classes=1000,
    )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        num_classes=args.nb_classes,
        # distillation=False,
        pretrained=True,
        # fuse=False,
    )
    model = model.to("cuda")
    example_inputs = torch.randn(20, 3, 224, 224).to("cuda")
    torch.onnx.export(model, example_inputs, "model_out.onnx", input_names = ['input'], output_names=['output'])#, dynamic_axes={'input' : {0 : 'batch_size'},})
    
    latency_uncompressed = benchmark(onnx_model_path="model_out.onnx", fp16=False)
    test_stats = evaluate(data_loader_val, model, device)
    print(f"Accuracy of the network on the {len(dataset_val)} test images (BEFORE compression): {test_stats['acc1']:.1f}%")
    calculate_flops(model)
    svd_method_args = {}
    if args.svd_method == "asvd" or args.svd_method == "fwsvd":
        svd_method_args["alpha"] = args.asvd_alpha
    
    search_args = {}
    search_args["ratio_target"] = args.compression_target
    if args.search_method == "flar_svd":
        search_args["threshold"] = args.error_threshold
        search_args["target_metric"] = args.target_metric
        search_args["progressive_comp"] = args.progressive_comp
    if args.blockwise:
        search_args["stage_name_in_current_model"] = args.stage_name

    factorizor = ModelFactorizer(
        svd_method=args.svd_method,
        svd_method_args=svd_method_args,
        search_method=args.search_method,
        search_method_args=search_args,
    )
    factorizor.factorize_and_search(
        model=model,
        calib_data=data_loader_svd_calib,
        eval_data=data_loader_svd_eval,
        mixup_fn=mixup_fn,
        name_omit=args.name_omit,
        blockwise_search=args.blockwise,
    )
    calculate_flops(model)
    test_stats = evaluate(data_loader_val, model, device)
    torch.onnx.export(model.to("cuda"), example_inputs, "model_out.onnx", input_names = ['input'])
    latency_compressed = benchmark(onnx_model_path="model_out.onnx", fp16=False)
    print(f"Accuracy of the network on the {len(dataset_val)} test images (AFTER compression): {test_stats['acc1']:.1f}%")
    print(f"Latency (uncompressed): {latency_uncompressed}, latency (compressed): {latency_compressed}, reduction: {latency_compressed/latency_uncompressed:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "FLAR-SVD compression and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
