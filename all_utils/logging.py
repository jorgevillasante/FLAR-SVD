import os
import sys
import typing

if typing.TYPE_CHECKING:
    from loguru import Logger
else:
    Logger = None

__all__ = ["logger"]


def redirect_all_std_out_of_subprocesses():
    rank = os.environ.get("RANK", None)
    if rank is None:
        # Not distributed training
        return
    else:
        if not rank == "0":
            print(f"Suppressing print for rank: {rank}")
            f = open(os.devnull, "w")
            sys.stdout = f


def __get_logger() -> Logger:
    from loguru import logger

    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        filter=lambda record: record["level"].no < 40,
        colorize=True,
        format=(
            "<green>[{time:YYYY-MM-DD HH:mm:ss.SSS}]</green> "
            "<level>{message}</level>"
        ),
    )
    return logger


redirect_all_std_out_of_subprocesses()
logger = __get_logger()
