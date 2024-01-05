"""My attempt to get rid of code duplicaiton in scripts."""

from __future__ import annotations

import pprint
import os
import logging
from abc import abstractmethod
from typing import TYPE_CHECKING
import argparse

import coolname

from transformer_document_embedding.scripts.utils import init_type, save_config


if TYPE_CHECKING:
    from transformer_document_embedding.scripts.config_specs import ExperimentSpec
    from transformer_document_embedding.models.experimental_model import (
        ExperimentalModel,
    )
    from transformer_document_embedding.tasks.experimental_task import ExperimentalTask
    from typing import Optional


class Pipeline:
    def add_args(self, parser: argparse.ArgumentParser) -> None:
        pass

    @abstractmethod
    def run(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError()


class TrainingPipeline(Pipeline):
    def __init__(
        self,
        load_model_path: Optional[str] = None,
        load_model_strictly: bool = True,
        save_trained: bool = False,
        train: bool = False,
    ) -> None:
        super().__init__()

        self._load_model_path = load_model_path
        self._save_trained = save_trained
        self._train = train
        self._load_model_strictly = load_model_strictly

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--load_model_path",
            type=str,
            default=self._load_model_path,
            help="Path from which to load the fitted model before training.",
        )

        parser.add_argument(
            "--save_trained",
            type=bool,
            action=argparse.BooleanOptionalAction,
            default=self._save_trained,
            help="Whether to save trained model.",
        )

        parser.add_argument(
            "--train",
            type=bool,
            action=argparse.BooleanOptionalAction,
            default=self._train,
            help="Whether to train the model before generating embeddings.",
        )

        parser.add_argument(
            "--load_model_strictly",
            type=bool,
            action=argparse.BooleanOptionalAction,
            default=self._load_model_strictly,
            help="Whether to fail for unknown or missing parameters when loading a "
            "model",
        )

    def run(
        self,
        args: argparse.Namespace,
        model: ExperimentalModel,
        task: ExperimentalTask,
        exp_path: str,
        config: ExperimentSpec,
    ) -> None:
        if args.load_model_path is not None:
            logging.info("Loading model from %s.", args.load_model_path)
            model.load(args.load_model_path, strict=args.load_model_strictly)

        if args.train:
            logging.info("Training model...")

            model.train(
                task,
                log_dir=exp_path,
                **config.model.train_kwargs,
            )
            logging.info("Training done.")

        if args.save_trained:
            trained_path = os.path.join(exp_path, "model")
            logging.info(
                "Saving trained model to %s.",
                trained_path,
            )
            os.makedirs(trained_path, exist_ok=True)
            model.save(trained_path)


class InitializeModelAndTask(Pipeline):
    def run(
        self,
        exp_name: str,
        exp_path: str,
        config: ExperimentSpec,
    ) -> tuple[ExperimentalModel, ExperimentalTask]:
        logging.info(
            "Starting experiment '%s' with config:\n%s",
            exp_name,
            pprint.pformat(config, indent=1),
        )
        save_config(config, exp_path)

        task = init_type(config.task)
        model = init_type(config.model)
        return model, task


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
        default="_".join(coolname.generate(2)),
        help="Name of the experiment. If no name is given, one is generated.",
    )
