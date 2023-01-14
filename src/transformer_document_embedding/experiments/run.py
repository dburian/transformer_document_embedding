"""Runs experiment for given model and task.

# TODO: Document this module once the interface settles...
"""
import argparse
import logging
import pprint
from typing import Optional

import transformer_document_embedding as tde

EXPERIMENTS_DIR = "./results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
        "--grid_search_config",
        type=str,
        default=None,
        help=(
            "Run grid search over all arguments specified in grid search YAML"
            " file for each experiment. Syntax documented here: TODO."
        ),
    )
    parser.add_argument(
        "--output_base_path",
        type=str,
        default=EXPERIMENTS_DIR,
        help=(
            "Path to directory containing all experiment results. Default:"
            f" '{EXPERIMENTS_DIR}'."
        ),
    )
    parser.add_argument(
        "--save_model",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save model after training.",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help=(
            "Path from which to load the fitted model instead of fitting it (which is"
            " the default behaviour)."
        ),
    )

    args = parser.parse_args()
    return args


def run_single(
    config: tde.experiments.ExperimentConfig,
    save_model: bool,
    load_model_path: Optional[str],
) -> None:
    logging.info(
        "Starting experiment with config:\n%s",
        pprint.pformat(config.values, indent=1),
    )
    model = config.get_model_type()(
        log_dir=config.experiment_path,
        **config.values["model"].get("kwargs", {}),
    )
    task = config.get_task_type()(**config.values["task"].get("kwargs", {}))

    if load_model_path is not None:
        logging.info("Loading model from %s.", load_model_path)
        model.load(load_model_path)
    else:
        logging.info("Training model...")
        model.train(task.train)
        logging.info("Training done.")

    if save_model:
        logging.info("Saving trained model to %s", config.model_path)
        model.save(config.model_path)

    logging.info("Evaluating on test data...")
    test_predictions = model.predict(task.test)
    results = task.evaluate(test_predictions)
    logging.info("Evaluation done. Results:\n%s", results)

    tde.evaluation.save_results(results, config.experiment_path)

    config.save()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )

    for exp_file in args.config:
        config = tde.experiments.ExperimentConfig.parse(exp_file, args.output_base_path)

        if args.grid_search_config is None:
            run_single(
                config,
                args.save_model,
                args.load_model_path,
            )
            return

        for experiment_instance in config.grid_search(args.grid_search_config):
            run_single(
                experiment_instance,
                args.save_model,
                args.load_model_path,
            )


if __name__ == "__main__":
    main()
