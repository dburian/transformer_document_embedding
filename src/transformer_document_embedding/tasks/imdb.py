from datasets.arrow_dataset import Dataset
from datasets.combine import concatenate_datasets
from datasets.load import load_dataset

from transformer_document_embedding.tasks.experimental_task import \
    ExperimentalTask

IMDBData = Dataset


class IMDBClassification(ExperimentalTask):
    """Classification task done using the IMDB dataset.

    The dataset is specified as `datasets.Dataset` with 'train', 'test' and
    'unsupervised' splits.
    """

    def __init__(self) -> None:
        self._train = None
        self._test = None
        self._unsuper = None
        self._all_train = None
        self._test_inputs = None
        self._train_test_size = 25000

    @property
    def train(self) -> IMDBData:
        """
        Returns datasets.Dataset of both train and unsupervised training
        documents. Each document is dictionary with keys:
            - 'text' (str) - text of the document,
            - 'label' (int) - 1/0 sentiment class index,
            - 'id' (int) - document id unique among all the documents in the dataset.
        """
        if self._all_train is None:
            self._train = load_dataset("imdb", split="train").map(
                lambda _, idx: {"id": self._get_id_from_index(idx, "train")},
                with_indices=True,
            )
            self._unsuper = load_dataset("imdb", split="unsupervised").map(
                lambda _, idx: {"id": self._get_id_from_index(idx, "unsuper")},
                with_indices=True,
            )

            self._all_train = concatenate_datasets([self._train, self._unsuper])

        return self._all_train

    @property
    def test(self) -> IMDBData:
        """
        Returns datasets.Dataset of testing documents. Each document is
        dictionary with keys:
            - 'text' (str) - text of the document,
            - 'id' (int) - document id unique among all the documents in the dataset.
        """
        if self._test_inputs is None:
            self._test = load_dataset("imdb", split="test").map(
                lambda _, idx: {"id": self._get_id_from_index(idx, "test")},
                with_indices=True,
            )
            self._test_inputs = self._test.remove_columns("label")

        return self._test_inputs

    def evaluate(self, test_predictions) -> dict[str, float]:
        pass

    def _get_id_from_index(self, idx: int, split: str) -> int:
        """
        Assings global id from document index based on data split.
        """
        split_ind = ["test", "train", "unsuper"].index(split)
        return split_ind * self._train_test_size + idx


Task = IMDBClassification
