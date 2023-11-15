# Hyperparameter logging

Implementation-wise there are some gotchas. I will describe not-so-working
approaches to hp logging which I've tried below the division line. What is above
is relevant to the current implementation.

## Updates:

### Only logging the currently searched over hyperparameters

I've decided to only log the hyperparameters which are different from the base
configuration. This may turn out to be a mistake, so here is my reasoning:

- Too many parameters over-crowds the 'HPARAMS' screen in tensorboard.
- All the parameters are visible in the experiment config, so they are
  recoverable.
- In almost all the cases I am only interested in the hyperparameters I am
  searching over.
- The only disadvantage is one-search. In current version only the one parameter
  that is different is logged. I may want to see all that will be changed (in
  other experiments in the current search). But that may be easily implemented.

---

This turned out to be quite a problem. I've tried fairly advanced approach,
which did not work at the end (due to some unexplicable inconsistencies in
`tensorboard`). So here I'll explain the approaches I tried and reasons why they
do not work.

## Approaches

### The default

I had a directory structure like so:

```
experiment_dir
    hparams
        hparams' values event file
    train
        ...train event files
    val
        ... val event files
```

This did not work out of the box, because HParams plugin did not know where the
metrics are for given hparam values and critically [**HParam plugin does not show
hparams without any
metric**](https://github.com/tensorflow/tensorboard/issues/5476).

### The all-in approach

Then because I did not know what was wrong, I tried to recreate a setup
described in `tensorboard`'s
[guide](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams).

The used directory structure was as so:
```
hp_search_root_dir
    hparams config event file
    experiment_1_dir
        hparams' values event file
        metric event files
    experiment_2_dir
        hparams' values event file
        metric event files
    ...
```

So I tried the following:

1. Have separate directories for experiment results and hparam tuning.
2. Directory structure for results was as it was always been. Hparam tuning had
   experiments organised by hp search.
3. To generate the hparam config we had to know two things:
    - hparams and their types
    - metric names
Hparams and their types could be inferred from YAML file, while metric names could be
inferred from experiments' logs (interpreting all scalars as metric names) after
all experiments have finished.
4. To have the metrics in the hparam tuning directory structure, the scalars
   were copied (while changing their names) to a single event file under the
   experiment dir in the hparam tuning dir structure.
5. Then the hparam config scanned for all scalar names, and written them to the
   hparam config log as metric names.

The result was I could do:
```
tensorboard --logdir .
```

to have both the loss/accuracy curves and hparams. The only negative was that
there were duplicate metrics.


#### Problem:

The problem of this approach is that for some reason the copied metrics got
written as tensors instead of scalars. This complicated the scanning for
metrics and to be honest was the last drop for me.

I've written a lot of code for this approach so I'll share it at the end of this
file.

### The easy, no-hparam-config approach

The solution was just to log the hparam values at the root of the experiment
directory. That's it. HParams on their own infer all the hparam types, and
metrics from children directories. No duplicate metrics required, less code, no
hacks.

## Code for all-in approach

Logging helpers in `experiments/logs.py`:
```python
import os
from typing import Iterable, Optional

from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
from tensorboard.backend.event_processing.io_wrapper import IsSummaryEventsFile


def list_event_files(
    dirpath: str, breadcrumbs: Optional[list[str]] = None
) -> Iterable[tuple[str, list[str]]]:
    if breadcrumbs is None:
        breadcrumbs = []

    with os.scandir(dirpath) as direntry_it:
        for entry in direntry_it:
            if entry.is_dir():
                for found_event_file in list_event_files(
                    entry.path, breadcrumbs + [entry.name]
                ):
                    yield found_event_file
            elif IsSummaryEventsFile(entry.path):
                yield entry.path, breadcrumbs


def get_all_scalar_names(log_dir: str) -> list[str]:
    scalar_names = set()
    for event_path, _ in list_event_files(log_dir):
        scalars = get_last_scalars(event_path)
        scalar_names.update(scalars.keys())

    return list(scalar_names)


def get_all_last_scalars(log_dir: str) -> dict[str, float]:
    all_scalars = {}
    for event_path, breadcrumbs in list_event_files(log_dir):
        event_scalars = get_last_scalars(event_path)

        prefix = "_".join(breadcrumbs)
        for key, value in event_scalars.items():
            all_scalars[f"{prefix}_{key}"] = value

    return all_scalars


def get_last_scalars(event_file_path: str) -> dict[str, float]:
    scalars = {}
    accumulator = EventAccumulator(event_file_path)

    # Reads events
    accumulator.Reload()
    for scalar_name in accumulator.Tags()["scalars"]:
        events = accumulator.Scalars(scalar_name)
        if len(events) > 0 and hasattr(events[-1], "value"):
            scalars[scalar_name] = events[-1].value

    return scalars
```

New type of config with hparam logging:
```python
class HPSearchExperimentConfig(ExperimentConfig):
    def __init__(
        self,
        values: dict[str, Any],
        base_results_dir: str,
        *,
        hp_search_identifier: str,
        hp_results_dir: str,
    ) -> None:
        super().__init__(values, base_results_dir)

        self._hp_search_identifier = hp_search_identifier
        self._hp_results_dir = hp_results_dir

    def log_hparams(self) -> None:
        with tf.summary.create_file_writer(
            os.path.join(self._hp_results_dir, self._hp_search_identifier)
        ).as_default():
            hparams = flatten_dict(self.values)
            hp.hparams(
                hparams,
                os.path.relpath(self.experiment_path, start=self.base_results_dir),
            )

            tf.summary.flush()
```

Extra methods for `HyperparameterSearch`:
```python
    @property
    def hp_search_base_path(self) -> str:
        if self._root_path is None:
            self._root_path = os.path.join(
                self.output_base_path,
                f'{datetime.now().strftime("%Y-%m-%d_%H%M%S")}',
            )

            os.makedirs(self._root_path, exist_ok=True)

        return self._root_path

    def log_hparam_config(self) -> None:
        metric_names = get_all_scalar_names(self.hp_search_base_path)
        print(metric_names)
        config_path = self.hp_search_base_path

        hparams = self._create_hparams_types()

        with tf.summary.create_file_writer(config_path).as_default():
            hp.hparams_config(
                hparams=hparams, metrics=[hp.Metric(name) for name in metric_names]
            )
            tf.summary.flush()

    def _create_hparams_types(self) -> list[hp.HParam]:
        def _construct_hparam_type(
            values: list[Any],
        ) -> Union[hp.Discrete, hp.RealInterval, hp.IntInterval]:
            if min(isinstance(val, int) for val in values) is True:
                return hp.IntInterval(min(values), max(values))
            if min(isinstance(val, (float, int)) for val in values) is True:
                return hp.RealInterval(float(min(values)), float(max(values)))

            return hp.Discrete(values)

        hparams_config = []
        for key, values in flatten_dict(self.hparams).items():
            hparams_config.append(hp.HParam(key, _construct_hparam_type(values)))

        return hparams_config

    @classmethod
    def get_identifier(cls, hparams: dict[str, Any]) -> str:
        identifier = []
        for key, value in hparams.items():
            str_key = ".".join(
                "_".join(re.sub("(.)[^_]*_?", r"\1", breadcrumb))
                for breadcrumb in key.split(".")
            )
            identifier.append(f"{str_key}={value}")

        return ",".join(identifier)
```

Grid-search went as so:
```python
    def based_on(
        self, reference_config: ExperimentConfig
    ) -> Iterable[ExperimentConfig]:
        for combination in self._all_combinations(self.hparams):
            new_values = deepcopy(reference_config.values)
            for gs_key, gs_value in combination.items():
                self._update_with_hparam(new_values, gs_key, gs_value)

            yield HPSearchExperimentConfig(
                new_values,
                reference_config.base_results_dir,
                hp_search_identifier=self._get_identifier(combination),
                hp_results_dir=self.hp_search_base_path,
            )
```

Then the script received new argument:
```python
parser.add_argument(
    "--hp_search_base_path",
    type=str,
    default=HP_SEARCH_DIR,
    help="Path to directory containing all logs for hyperparameter searching.",
)
```

And the experiments were run as so:
```python
        search = search_cls.from_yaml(config_path, args.hp_search_base_path)
        for exp_file in args.config:
            config_path = tde.experiments.ExperimentConfig.from_yaml(
                exp_file, args.output_base_path
            )

            for experiment_instance in search.based_on(config_path):
                # Also logged hparam configs by calling
                `experiment_instance.log_hparams()`
                run_single(experiment_instance, args.early_stop)

        search.log_hparam_config()
```
