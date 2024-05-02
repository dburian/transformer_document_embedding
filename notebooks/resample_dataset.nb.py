# %% [markdown]
# # Create a resampled copy of dataset
#
# 1. Loads a `Dataset`
# 2. Initializes a sampler
# 3. Iterates over the sampler to the desired target length
# 4. Saves the resulting dataset to disk

# %%
from __future__ import annotations
from datasets import load_from_disk, Dataset, DatasetDict

from typing import TYPE_CHECKING
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from typing import Iterator, Any
    from torch.util.data import Sampler

# %%
DS_PATH = "/mnt/data/datasets/wikipedia_resampled_eval"
NEW_DS_PATH = "/mnt/data/datasets/wikipedia_sample"
NEW_DS_SIZE = 150000
NUM_PROC = 6

# %%
ds = load_from_disk(DS_PATH)


# %%
def new_ds_gen(split: Dataset, sampler: Sampler) -> Iterator[Any]:
    for count, idx in enumerate(sampler, start=1):
        yield split[idx]

        if count == NEW_DS_SIZE:
            break


# %%
from collections import deque
import numpy as np

import logging
from typing import TYPE_CHECKING, Generic, TypeVar

from torch.utils.data import Sampler

from transformer_document_embedding.datasets import col

if TYPE_CHECKING:
    import datasets
    from typing import Iterable, Iterator, Any, Optional

logger = logging.getLogger(__name__)


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
        Parameters
        ----------
        dataset: datasets.Dataset
            Dataset to sample
        bucket_limits: list[int]
            List of length thresholds that define buckets.
        effective_batch_size: int
            Number of documents that will be used in effective batch. Typically
            this would be `batch_size * gradient_accumulation_steps`.
        mega_batch_size: int
            Number of documents to cache to


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
        idx_length_iter = enumerate(self._dataset[col.LENGTH])

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
        bucket_limits: list[int],
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
            limits=bucket_limits,
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
            # rand_bucket_idx = non_empty_bucket_idxs[np.argmin(sample_dist)]

            sample_dist /= sample_dist.sum()

            rand_bucket_idx = self._rand_gen.choice(
                non_empty_bucket_idxs, p=sample_dist
            )
            bucket_counts[rand_bucket_idx] += 1

        assert bucket_counts.sum() == count

        for bucket, size in zip(self._buckets, bucket_counts, strict=True):
            yield from (bucket.pop() for _ in range(size))


# %%
new_splits = {}
for split_name, split in ds.items():
    sampler = ConsistentLenghtDistSampler(
        dataset=split,
        effective_batch_size=64,
        bucket_limits=[512 + i * 512 for i in range(8)],
    )

    new_split = Dataset.from_generator(
        new_ds_gen, gen_kwargs={"split": split, "sampler": sampler}
    )
    new_splits[split_name] = new_split
new_ds = DatasetDict(new_splits)


# %%
def plot_length_dist(ds):
    batch_size = 1024 * 8
    splits = ["train"]
    if "validation" in ds:
        splits.append("validation")

    for split in tqdm(splits, "Splits"):
        batch_iter = tqdm(
            ds[split].select_columns("length").with_format("np").iter(batch_size),
            desc="Batches",
            total=len(ds[split]) // batch_size + 1,
        )
        data = np.array(
            [
                [batch["length"].mean(), *np.percentile(batch["length"], [25, 75])]
                for batch in batch_iter
            ]
        )
        xs = np.arange(data.shape[0]) * batch_size
        lines = plt.plot(xs, data[:, 0], label=f"{split}")
        plt.fill_between(
            xs, data[:, 1], data[:, 2], color=lines[0].get_color(), alpha=0.1
        )

    plt.legend()


# %%
plot_length_dist(new_ds)

# %%
plot_length_dist(new_ds)

# %%
new_ds.save_to_disk(NEW_DS_PATH, max_shard_size="1GB")
