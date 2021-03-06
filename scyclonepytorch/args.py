from argparse import ArgumentParser, Namespace
from typing import List, Optional


def parseArgments(
    parser: ArgumentParser, input: Optional[List[str]] = None
) -> Namespace:
    """
    Parse Scyclone-PyTorch arguments
    """

    # path for logging & checkpointing
    parser.add_argument("--dir_root", default="scyclonepytorch/logs", type=str)
    parser.add_argument("--name_exp", default="default", type=str)
    parser.add_argument("--name_version", default="version_-1", type=str)
    # max: from Scyclone poster (check my Scyclone summary blog post)
    parser.add_argument("--max_epochs", default=8000, type=int)
    #6240
    parser.add_argument("--noiseless_d", action="store_true")
    parser.add_argument("--sampling_rate", default=16000, type=int)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--val_interval_epoch", default=50, type=int)
    parser.add_argument("--adress_data_root", type=str)
    # DataLoaderPerformance
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--profiler", type=str, choices=["simple", "advanced"])

    return parser.parse_args() if input is None else parser.parse_args(input)
