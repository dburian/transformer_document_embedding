"""My attempt to get rid of code duplicaiton in scripts."""

from __future__ import annotations

import pprint
import os
import logging
from abc import abstractmethod
from typing import TYPE_CHECKING
import argparse


if TYPE_CHECKING:
    from transformer_document_embedding.experiments.config import ExperimentConfig
    from transformer_document_embedding.baselines.baseline import Baseline
    from transformer_document_embedding.tasks.experimental_task import ExperimentalTask
    from typing import Optional


class Pipeline:
    @abstractmethod
    def add_args(self, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError()

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
        save_trained: bool = False,
        train: bool = False,
    ) -> None:
        super().__init__()

        self._load_model_path = load_model_path
        self._save_trained = save_trained
        self._train = train

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

    def run(
        self,
        args: argparse.Namespace,
        model: Baseline,
        task: ExperimentalTask,
        config: ExperimentConfig,
    ) -> None:
        if args.load_model_path is not None:
            logging.info("Loading model from %s.", args.load_model_path)
            model.load(args.load_model_path)

        if args.train:
            logging.info("Training model...")

            model.train(
                task,
                log_dir=config.experiment_path,
                model_dir=config.model_path,
                **config.values["model"].get("train_kwargs", {}),
            )
            logging.info("Training done.")

            if args.save_trained:
                trained_path = os.path.join(config.model_path, "trained")
                logging.info(
                    "Saving trained model to %s.",
                    trained_path,
                )
                os.makedirs(trained_path, exist_ok=True)
                model.save(trained_path)


class InitializeModelAndTask(Pipeline):
    def add_args(self, _: argparse.ArgumentParser) -> None:
        pass

    def run(
        self,
        config: ExperimentConfig,
    ) -> tuple[Baseline, ExperimentalTask]:
        logging.info(
            "Starting experiment '%s' with config:\n%s",
            config.name,
            pprint.pformat(config.values, indent=1),
        )
        config.save()
        model = config.get_model_type()(**config.values["model"].get("kwargs", {}))
        task = config.get_task_type()(**config.values["task"].get("kwargs", {}))
        return model, task
