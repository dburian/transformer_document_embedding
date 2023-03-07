from typing import Optional
import datasets
import tensorflow as tf
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
import pynvml


class LossEvaluator(SentenceEvaluator):
    def __init__(
        self,
        labeled_texts: datasets.Dataset,
        log_dir: str,
        loss: torch.nn.Module,
        *,
        beta: float = 0.9,
        name: str = "loss",
        batch_size: int = 32,
    ) -> None:
        self._writer = tf.summary.create_file_writer(log_dir)
        self._loss = loss
        self._labeled_texts = labeled_texts.with_format("torch")
        self._beta = beta
        self._batch_size = batch_size
        self._last_step = 0
        self._name = name

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: Optional[str] = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> float:
        norm = 1
        running_avg = torch.tensor(0, device=model.device)
        for i in range(0, len(self._labeled_texts), self._batch_size):
            batch = self._labeled_texts[i : i + self._batch_size]
            pred_labels = model.encode(
                batch["text"],
                batch_size=self._batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            pred_labels = pred_labels[:, 0]

            loss = self._loss(
                pred_labels, batch["label"].float().to(device=model.device)
            )
            running_avg = self._beta * running_avg + (1 - self._beta) * loss
            norm *= self._beta

        avg_loss = running_avg / (1 - norm)
        avg_loss = avg_loss.numpy(force=True)

        with self._writer.as_default():
            tf.summary.scalar(self._name, avg_loss, step=epoch)

        return -avg_loss


# Only tested for binary loss
class AccuracyEvaluator(SentenceEvaluator):
    def __init__(
        self, labeled_texts: datasets.Dataset, log_dir: str, *, batch_size: int = 32
    ) -> None:
        self._labeled_texts = labeled_texts.with_format("torch")
        self._writer = tf.summary.create_file_writer(log_dir)
        self._batch_size = batch_size

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: Optional[str] = None,
        epoch: int = -1,
        step: int = -1,
    ) -> float:
        total = 0
        correct = 0

        for i in range(0, len(self._labeled_texts), self._batch_size):
            batch = self._labeled_texts[i : i + self._batch_size]

            pred = model.encode(
                batch["text"],
                batch_size=self._batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            pred = torch.round(pred[:, 0])
            true = batch["label"].to(pred.device)

            correct += torch.sum(pred == true)
            total += len(batch)

        correct = correct.numpy(force=True)

        accuracy = correct / total

        with self._writer.as_default():
            tf.summary.scalar("accuracy", accuracy, step=epoch)

        return accuracy


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
        return 0
