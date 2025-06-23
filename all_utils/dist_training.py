import os
import typing

import torch

if typing.TYPE_CHECKING:
    from session.trainer.trainers._base_trainer import BaseTrainer

import torch.nn as nn
import torch.distributed as dist
from .logging import logger

from all_utils.list import list_sum, list_mean


def dist_init() -> None:
    if is_dist_initialized():
        return
    try:
        dist.init_process_group(backend="nccl")
        assert dist.is_initialized()
    except Exception:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        logger.warning(
            "Distributed environment not initialized, moving on without it ..."
        )


def is_dist_initialized() -> bool:
    return dist.is_initialized()


def get_dist_rank() -> int:
    return int(os.environ["RANK"])


def is_master() -> bool:
    return get_dist_rank() == 0


def dist_barrier() -> None:
    if is_dist_initialized():
        dist.barrier()


def get_dist_local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def setup_cuda_env() -> None:
    torch.cuda.set_device(get_dist_local_rank())


def sync_model(trainer: "BaseTrainer"):
    print("Sync model")
    trainer.save_model(model_name="sync.pt")
    dist_barrier()
    checkpoint = torch.load(
        os.path.join(trainer.checkpoint_path, "sync.pt"), map_location="cpu"
    )
    dist_barrier()
    if is_master():
        os.remove(os.path.join(trainer.checkpoint_path, "sync.pt"))
    dist_barrier()

    # load checkpoint
    trainer.network.load_state_dict(checkpoint["state_dict"], strict=False)
    if "optimizer" in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    if "lr_scheduler" in checkpoint:
        trainer.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if "ema" in checkpoint and trainer.ema is not None:
        trainer.ema.load_state_dict(checkpoint["ema"])
    if "scaler" in checkpoint and trainer.fp16:
        trainer.scaler.load_state_dict(checkpoint["scaler"])


def is_parallel(model: nn.Module) -> bool:
    return isinstance(
        model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    )


def get_dist_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def sync_tensor(
    tensor: torch.Tensor | float, reduce="mean"
) -> torch.Tensor | list[torch.Tensor]:
    if not is_dist_initialized():
        return tensor
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.Tensor(1).fill_(tensor).cuda()
    tensor_list = [torch.empty_like(tensor) for _ in range(get_dist_size())]
    dist.all_gather(tensor_list, tensor.contiguous(), async_op=False)
    if reduce == "mean":
        return list_mean(tensor_list)
    elif reduce == "sum":
        return list_sum(tensor_list)
    elif reduce == "cat":
        return torch.cat(tensor_list, dim=0)
    elif reduce == "root":
        return tensor_list[0]
    else:
        return tensor_list
