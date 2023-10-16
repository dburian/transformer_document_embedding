from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


RESULTS_DIR = "./results"


def add_common_args(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        action="extend",
        help=(
            "Experiment configurations to be run described by YAML files. Required"
            " syntax to be found here:"
            " github.com/dburian/transformer_document_embedding/"
            "blob/master/log/experiment_files.md."
        ),
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--output_base_path",
        type=str,
        default=RESULTS_DIR,
        help=(
            "Path to directory containing all experiment results. Default:"
            f" '{RESULTS_DIR}'."
        ),
    )
