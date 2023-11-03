from __future__ import annotations

from collections import deque
import numpy as np

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar, cast

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.trainer_pt_utils import get_parameter_names

if TYPE_CHECKING:
    from transformers.utils import PaddingStrategy
    from transformers.tokenization_utils import TruncationStrategy
    import datasets
    from typing import Iterable, Iterator, Sized, Any, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class FastDataCollator:
    """Custom data collator to be used with FastTokenizers.

    Implemented in order to:
        - avoid the warning caused by first encoding with fast tokenizer and
          then padding with it on a separate call.
        - have the feature of ensuring encoded input is of minimal length.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    min_length: Optional[int] = None
    truncation: Union[bool, str, TruncationStrategy] = "longest_first"
    return_length: bool = False

    def __post_init__(self):
        self._max_special_token_id = max(
            self.tokenizer.pad_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.unk_token_id,
            self.tokenizer.cls_token_id,
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # Convert from list of dicts to dict of lists
        batch = {}
        for key in features[0].keys():
            batch[key] = [example[key] for example in features]

            # Prefer 2d tensors over list of tensors
            if isinstance(batch[key][0], torch.Tensor):
                batch[key] = torch.stack(batch[key])

        texts = batch["text"]
        del batch["text"]

        tokenized_batch = cast(
            BatchEncoding,
            self.tokenizer(
                texts,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=None,
                truncation=self.truncation,
                return_length=False,  # Here return_length returns length after padding
            ),
        )

        if self.min_length:
            batch_size = len(tokenized_batch["input_ids"])
            for i in range(batch_size):
                input_length = len(tokenized_batch["input_ids"][i])
                if input_length >= self.min_length:
                    continue
                difference = self.min_length - input_length

                if "attention_mask" in tokenized_batch:
                    tokenized_batch["attention_mask"][i] += [0] * difference

                if "token_type_ids" in tokenized_batch:
                    tokenized_batch["token_type_ids"][i] += [
                        self.tokenizer.pad_token_type_id
                    ] * difference
                if "special_tokens_mask" in tokenized_batch:
                    tokenized_batch["special_tokens_mask"][i] += [1] * difference

                tokenized_batch["input_ids"][i] += [
                    self.tokenizer.pad_token_id
                ] * difference

        tokenized_batch.convert_to_tensors(
            self.return_tensors, prepend_batch_axis=False
        )

        if self.return_length:
            tokenized_batch["length"] = torch.sum(
                tokenized_batch["input_ids"] > self._max_special_token_id, axis=1
            )

        batch.update(tokenized_batch)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


class ConsistentNumberOfTokensSampler(Sampler):
    def __init__(
        self,
        dataset: Sized,
        lengths: Iterable[int],
        tokens_in_batch: int,
    ) -> None:
        self._dataset = dataset
        self._lengths = lengths
        self._tokens_in_batch = tokens_in_batch
        super().__init__(None)

    def __len__(self) -> int:
        # TODO: This is complicated. I don't know:
        # - what exactly is the equivalent in fairseq
        # - isn't recompilation for different batch size a problem?
        # - how to compute number of batches then?
        # - There isn't a HF thing that would create dynamic batch sizes based
        # on number of tokens in each batch
        return (sum(self._lengths) // self._tokens_in_batch) + 1

    def __iter__(self) -> Iterable[Any]:
        raise NotImplementedError()


class ConsistentLenghtDistSampler(Sampler):
    def __init__(
        self,
        dataset: datasets.Dataset,
        bucket_limits: list[int],
        effective_batch_size: int,
        mega_batch_size: int = 1000,
        generator: Optional[np.random.Generator] = None,
    ) -> None:
        """Creates a sampler that tries to be consistent in input length.
        Parameters:
        -----------
        """
        super().__init__(None)
        self._dataset = dataset
        self._mega_batch_size = mega_batch_size
        self._buffer = BucketedBuffer(
            limits=bucket_limits,
            generator=generator,
        )
        self._effective_batch_size = effective_batch_size

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self) -> Iterator[int]:
        idx_length_iter = enumerate(self._dataset["length"])

        # Add an input to buffer so we kick-start the pipeline
        idx, length = next(idx_length_iter)
        self._buffer.append(idx, bucketing_criterion=length)

        while len(self._buffer) > 0:
            feeding_rounds = min(
                # Do not fill up more than mega_batch_size
                self._mega_batch_size - len(self._buffer),
                # Feed only as long as we have data
                len(self._dataset) - idx - 1,
            )
            for _ in range(feeding_rounds):
                idx, length = next(idx_length_iter)
                self._buffer.append(idx, bucketing_criterion=length)

            yield from self._buffer.sample(count=self._effective_batch_size)


class TargetLenghtDistSampler(Sampler):
    def __init__(
        self,
        dataset: datasets.Dataset,
        effective_batch_size: int,
        length_limits: list[int],
        target_dist: list[float],
        generator: Optional[np.random.Generator] = None,
        max_bucket_size_factor: int = 3,
    ) -> None:
        """Creates a sampler that tries to be match desired distribution of
        input lengths.

        Can drop data.

        Parameters:
        -----------
        target_dist: list[float]
            Desired distribution of lengths in a single effective batch.
        max_bucket_size_factor: float
            How many effective batches should fit into a single bucket. The
            larger the number the larger possible memory costs, but the less
            inputs are going to be dropped.
        """
        super().__init__(None)
        self._dataset = dataset
        self._effective_batch_size = effective_batch_size
        self._target_dist = target_dist

        self._buffer = BucketedBuffer(
            limits=length_limits,
            generator=generator,
            max_bucket_size=max_bucket_size_factor * effective_batch_size,
        )

    def __len__(self) -> int:
        logger.warn(
            "%s can drop inputs. Dataset length is probably not equal to number of "
            "sampled inputs.",
            TargetLenghtDistSampler.__name__,
        )
        return len(self._dataset)

    def __iter__(self) -> Iterator[Any]:
        idx_length_iter = enumerate(self._dataset["length"])
        desired_counts = np.ceil(
            self._effective_batch_size * np.array(self._target_dist)
        )

        # Add an input to buffer so we kick-start the pipeline
        idx, length = next(idx_length_iter)
        self._buffer.append(idx, bucketing_criterion=length)

        while len(self._buffer) > 0:
            # Fill up the buffer until
            while (
                # each bucket has at least as many inputs as we need
                self._buffer.bucket_sizes < desired_counts
                # or we have exhausted the dataset
                and idx < len(self._dataset) - 1
            ):
                idx, length = next(idx_length_iter)
                self._buffer.append(idx, bucketing_criterion=length)

            yield from self._buffer.sample(
                count=self._effective_batch_size,
                target_dist=self._target_dist,
            )


T = TypeVar("T")


class BucketedBuffer(Generic[T]):
    def __init__(
        self,
        limits: list[int],
        generator: Optional[np.random.Generator] = None,
        max_bucket_size: Optional[int] = None,
    ) -> None:
        """Creates buffer bucketed according to a criterion.

        Parameters:
        -----------
        limits: list[int]
            Sorted upper bounds for bucketing criterion.

        max_bucket_size: int, optional
            Maximum size per bucket. Elements appended to full buckets will be
            discarded.
        """
        self._bucket_limits = limits + [float("inf")]
        self._rand_gen = generator if generator is not None else np.random.default_rng()

        def create_bucket():
            return deque() if max_bucket_size is None else deque(maxlen=max_bucket_size)

        self._buckets = [create_bucket() for _ in range(len(self._bucket_limits))]

    @property
    def bucket_sizes(self) -> list[int]:
        return [len(bucket) for bucket in self._buckets]

    @property
    def current_dist(self) -> np.ndarray:
        return np.array([len(bucket) / len(self) for bucket in self._buckets])

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self._buckets)

    def append(self, element: T, bucketing_criterion: int) -> None:
        bucket_idx = 0
        while bucketing_criterion >= self._bucket_limits[bucket_idx]:
            bucket_idx += 1

        self._buckets[bucket_idx].appendleft(element)

    def sample(
        self, count: int, *, target_dist: Optional[list[float]] = None
    ) -> Iterable[T]:
        # Target distribution of sample
        desired_sample_dist = (
            self.current_dist if target_dist is None else np.array(target_dist)
        )

        bucket_sizes = np.array(self.bucket_sizes)

        # We cannot give what we don't have
        count = min(count, bucket_sizes.sum())
        bucket_counts = np.minimum(
            np.floor(count * desired_sample_dist).astype("int32"),
            bucket_sizes,
        )

        while bucket_counts.sum() < count:
            # Redistribute the sampling probability to non-empty buckets
            non_empty_bucket_idxs = np.nonzero(bucket_sizes - bucket_counts)[0]
            sample_dist = desired_sample_dist[non_empty_bucket_idxs]
            sample_dist /= sample_dist.sum()

            rand_bucket_idx = self._rand_gen.choice(
                non_empty_bucket_idxs, p=sample_dist
            )
            bucket_counts[rand_bucket_idx] += 1

        assert bucket_counts.sum() == count

        for bucket, size in zip(self._buckets, bucket_counts, strict=True):
            yield from (bucket.pop() for _ in range(size))


def create_tokenized_data_loader(
    data: datasets.Dataset,
    batch_size: Optional[int] = None,
    batch_size_in_tokens: Optional[int] = None,
    sampling: str = "default",
    training: bool = True,
    sampler_kwargs: Optional[dict[str, Any]] = None,
    **kwargs,
) -> DataLoader:
    """Creates DataLoder giving batches of tokenized text.

    Parameters:
    -----------
    data: datasets.Dataset
        Dataset from which to create a DataLoader
    batch_size: int, optional
        Number of inputs to put in a batch. Mutually exclusive with
        `batch_size_in_tokens`.
    batch_size_in_tokens: int, optional
        Number of tokens in each batch. Mutually exclusive with `batch_size`.
    sampling: str
        Method of data sampling. Mutually exclusive with `batch_size_in_tokens`.
    sampler_kwargs: dict[str, Any], optional
        Kwargs for sampler.
    training: bool
        Whether the input data are for training or for testing. In testing no
        augmentation, nor shuffling should happened.
    """
    if batch_size is None and batch_size_in_tokens is None:
        raise TypeError("Both `batch_size` and `batch_size_in_tokens` cannot be None.")

    data = data.with_format("torch")
    data = data.remove_columns(["id"])

    if training:
        data = data.shuffle()

    collator = FastDataCollator(
        padding="longest",
        **kwargs,
    )

    dataloader_kwargs: dict[str, Any] = {
        "collate_fn": collator,
    }

    if batch_size is not None:
        dataloader_kwargs["batch_size"] = batch_size

        if sampler_kwargs is None:
            sampler_kwargs = {}

        if sampling == "consistent":
            dataloader_kwargs["sampler"] = ConsistentLenghtDistSampler(
                **sampler_kwargs,
                dataset=data,
            )
        elif sampling == "target":
            dataloader_kwargs["sampler"] = TargetLenghtDistSampler(
                **sampler_kwargs,
                dataset=data,
            )

    elif batch_size_in_tokens is not None:
        raise NotImplementedError()
        batch_sampler = ...
        dataloader_kwargs["batch_sampler"] = batch_sampler

    dataloader = DataLoader(data, **dataloader_kwargs)
    return dataloader


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> None:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)


def get_optimizer_params(
    model: torch.nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]


def get_linear_lr_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)
