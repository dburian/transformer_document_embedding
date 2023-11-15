from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


OUTPUT_DIR = "./results"


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    output_base_path: str = OUTPUT_DIR,
) -> None:
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to yaml experiment file.",
        required=True,
    )
    parser.add_argument(
        "--output_base_path",
        type=str,
        default=output_base_path,
        help=(
            "Path to directory containing all experiment results. Default:"
            f" '{output_base_path}'."
        ),
    )

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help="Name of the experiment. If no name is given, one is generated.",
    )
