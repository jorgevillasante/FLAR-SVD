# from copy import deepcopy
import cosine_sim_utils as cosim_utils
import torch
import torch.nn as nn
from data.datasets import build_dataset
from timm.data import Mixup
from timm.models import create_model
from torch.utils.data import Subset

import utils


def main_loop(
    model: nn.Module,
    calib_batches,
    batch_size,
    dataset_train,
    datargs,
    mixup_fn,
    dev,
    rank,
):

    calib_size = int(calib_batches * batch_size)

    # Calibration datasets as subset of train dataset
    indices = list(range(len(dataset_train)))
    svd_calib_indices = indices[:calib_size]  # 1152 4608 16896
    dataset_svd_calib = Subset(dataset_train, svd_calib_indices)
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_svd_calib = torch.utils.data.DistributedSampler(
        dataset_svd_calib,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
    )

    data_loader_svd_calib = torch.utils.data.DataLoader(
        dataset_svd_calib,
        sampler=sampler_svd_calib,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    print("Obtaining activations")
    with torch.cuda.device(dev):
        torch.cuda.empty_cache()

    extractor = cosim_utils.HookReg(model)
    extractor.attach_hook()

    with torch.no_grad():
        for data, target in data_loader_svd_calib:
            model_inps, targets = mixup_fn(data, target)
            model_inps = model_inps.to(dev)
            model(model_inps)
            del model_inps, targets

    extractor.clear_hook()

    layer = model.blocks[0].attn.qkv
    # comp = deepcopy(layer)

    recon = cosim_utils.low_rank_approx(layer, extractor.profile, dev, rank)
    print(
        "Cholesky:",
        nn.functional.cosine_similarity(
            recon.flatten(), layer.weight.detach().flatten(), dim=0
        ),
    )

    recon = cosim_utils.low_rank_approx(layer, extractor.lwprofile, dev, rank)
    print(
        "FLAR-SVD:",
        nn.functional.cosine_similarity(
            recon.flatten(), layer.weight.detach().flatten(), dim=0
        ),
    )


class DatArgs:
    def __init__(self):
        self.data_set = "IMNET"
        self.data_path = "/data/datasets/PytorchDatasets/ImageNet-pytorch/"
        self.inat_category = "name"
        self.color_jitter = 0.3
        self.input_size = 224
        self.aa = "rand-m9-mstd0.5-inc1"
        self.train_interpolation = "bicubic"
        self.reprob = 0.25
        self.remode = "pixel"
        self.recount = 1

        self.nb_classes = None


if __name__ == "__main__":
    torch.manual_seed(310)  # 280
    device = torch.device("cuda")
    batch_size = 128
    rank = 64  # 85
    datargs = DatArgs()

    dataset_train, datargs.nb_classes = build_dataset(is_train=True, args=datargs)
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

    model = create_model(
        "deit_base_patch16_224.fb_in1k",
        num_classes=datargs.nb_classes,
        # distillation=False,
        pretrained=True,
        # fuse=False,
    )

    model.eval().to(device)

    calib_sizes = [16, 32, 78, 118, 156.25]
    for calib_batches in calib_sizes:
        print(f"Using {calib_batches} batches of {batch_size}")
        main_loop(
            model,
            calib_batches,
            batch_size,
            dataset_train,
            datargs,
            mixup_fn,
            device,
            rank,
        )
