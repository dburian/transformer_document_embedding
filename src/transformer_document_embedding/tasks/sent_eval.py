from __future__ import annotations
from typing import TYPE_CHECKING
from transformer_document_embedding.tasks.experimental_task import ExperimentalTask

if TYPE_CHECKING:
    from typing import Union, Any


PREDEFINED_PARAMS = {
    "debug": {
        "usepytorch": True,
        "kfold": 5,
        "classifier": {
            "nhid": 0,
            "optim": "rmsprop",
            "batch_size": 128,
            "tenacity": 3,
            "epoch_size": 2,
        },
    },
    "default": {
        "usepytorch": True,
        "kfold": 10,
        "classifier": {
            "nhid": 0,
            "optim": "adam",
            "batch_size": 64,
            "tenacity": 5,
            "epoch_size": 4,
        },
    },
}


class SentEval(ExperimentalTask):
    def __init__(
        self,
        path: str,
        tasks: list[str],
        params: Union[dict[str, Any], str],
        add_ids: bool = True,
    ) -> None:
        super().__init__()

        self.tasks = tasks
        self.add_ids = add_ids
        self.params = PREDEFINED_PARAMS[params] if isinstance(params, str) else params
        self.params["task_path"] = path
