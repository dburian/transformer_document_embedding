from typing import Any, Callable, Optional

import pynvml
import tensorflow as tf
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import batch_to_device
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torcheval.metrics import Metric


class TBMetricsEvaluator(SentenceEvaluator):
    def __init__(
        self,
        *,
        summary_writer: SummaryWriter,
        data_loader: DataLoader,
        metrics: dict[str, Metric],
        steps_in_epoch: int,
        metric_transforms: Optional[
            dict[str, Callable[[list[torch.Tensor]], Any]]
        ] = None,
        decisive_metric: Optional[str] = None,
        decisive_metric_higher_is_better: bool = True,
    ) -> None:
        self._writer = summary_writer
        self._data_loader = data_loader
        self._metrics = metrics
        self._steps_in_epoch = steps_in_epoch
        self._decisive_metric = decisive_metric
        self._decisive_metric_hib = decisive_metric_higher_is_better

        if metric_transforms is None:
            metric_transforms = {}

        for name in metrics:
            if name not in metric_transforms:
                metric_transforms[name] = lambda x: x
        self._metric_transforms = metric_transforms

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: Optional[str] = None,
        epoch: int = -1,
        step: int = -1,
    ) -> float:
        dev = model.device

        for name, metric in self._metrics.items():
            metric.reset()
            self._metrics[name] = metric.to(dev)

        self._data_loader.collate_fn = model.smart_batching_collate
        for batch in self._data_loader:
            siam_features, labels = batch
            for i, _ in enumerate(siam_features):
                siam_features[i] = batch_to_device(siam_features[i], dev)
            labels = labels.to(dev)
            with torch.no_grad():
                outputs = [
                    model(features)["sentence_embedding"] for features in siam_features
                ]

            for name, metric in self._metrics.items():
                transform = self._metric_transforms[name]
                metric.update(transform(outputs), labels)

        results = {name: metric.compute() for name, metric in self._metrics.items()}

        epoch, step, _ = _parse_epoch_step(epoch, step, self._steps_in_epoch)
        for name, res in results.items():
            self._writer.add_scalar(name, res, epoch * self._steps_in_epoch + step)
            self._metrics[name].reset()

        self._writer.flush()
        self._writer.close()
        if self._decisive_metric is None:
            return float("-inf")

        decisive_result = results[self._decisive_metric]

        if not self._decisive_metric_hib:
            decisive_result *= -1

        return decisive_result


def _parse_epoch_step(
    epoch: int, step: int, steps_in_epoch: int
) -> tuple[int, int, bool]:
    """Parses arguments given to `SentenceEvaluator.__call__` into meanigful values.

    Args:
        epoch: Number of epoch; -1 if evaluating on test data.
        step: Number of step in epoch; -1 if last step in epoch.
        steps_in_epoch: Number of steps in each epoch.
    Returns:
        A tuple [epoch, step, test_data], where:
            - epoch is number of epoch; 0 if evaluating test data,
            - step is number of steps in epoch; 0 if evaluating on test data,
            - test_data is True if evaluating test data.
    """
    if epoch == -1:
        return (0, 0, True)
    if step == -1:
        return (epoch, steps_in_epoch, False)

    return (epoch, step, False)


class VMemEvaluator(SentenceEvaluator):
    def __init__(
        self, log_dir: str, *, gpu_index: int = 0, name: str = "used_vmem_MB"
    ) -> None:
        pynvml.nvmlInit()
        self._writer = tf.summary.create_file_writer(log_dir)
        self._name = name
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: Optional[str] = None,
        epoch: int = -1,
        step: int = -1,
    ) -> float:
        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        used_MB = info.used // 1024**2
        with self._writer.as_default():
            tf.summary.scalar(self._name, used_MB, epoch)
        return float("-inf")
