"""Doc2Vec's implementation from the `gensim` package.


"""
import logging
import os
import random
from typing import Any, Optional

import numpy as np
import tensorflow as tf
from gensim.models import doc2vec
from gensim.models.callbacks import CallbackAny2Vec


class EmbeddingDifferencesCallback(CallbackAny2Vec):
    def __init__(
        self,
        *,
        log_dir: str,
        dm: bool,
        min_doc_id: int,
        max_doc_id: int,
        num_samples: int,
        seed: Optional[int] = None,
    ) -> None:
        self._tb_writer = tf.summary.create_file_writer(log_dir)
        self._epoch = 0
        self._scalar_name = f"doc2vec_{'dm' if dm else 'dbow'}_embed_diff"
        if seed:
            random.seed(seed)
        self._sample_doc_ids = [
            random.randint(min_doc_id, max_doc_id) for _ in range(num_samples)
        ]
        self._last_embed = None

    def on_epoch_end(self, model: doc2vec.Doc2Vec) -> None:
        new_embed = np.stack([model.dv[id] for id in self._sample_doc_ids], axis=0)
        new_embed = new_embed / np.reshape(np.linalg.norm(new_embed, axis=1), (-1, 1))
        if self._last_embed is not None:
            with self._tb_writer.as_default():
                distances = 1 - np.sum(new_embed * self._last_embed, axis=1)
                # Disregarding linearly decreasing learning rate
                distances /= model.alpha
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)
                logging.info(
                    "%s, Epoch %s: %.5f+-%.5f",
                    self._scalar_name,
                    self._epoch,
                    mean_distance,
                    std_distance,
                )
                tf.summary.scalar(
                    f"{self._scalar_name}_mean", mean_distance, self._epoch
                )
                tf.summary.scalar(f"{self._scalar_name}_std", std_distance, self._epoch)

        self._epoch += 1
        self._last_embed = new_embed


class PV_DBOW(doc2vec.Doc2Vec):
    def __init__(self, **kwargs) -> None:
        kwargs["dm"] = 0
        super().__init__(**kwargs)


class PV_DM(doc2vec.Doc2Vec):
    def __init__(self, **kwargs) -> None:
        kwargs["dm"] = 1
        super().__init__(**kwargs)


class ParagraphVector:
    def __init__(
        self,
        dm_kwargs: Optional[dict[str, Any]] = None,
        dbow_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.modules = []
        if dm_kwargs is not None:
            self.modules.append(PV_DM(**dm_kwargs))
        if dbow_kwargs is not None:
            self.modules.append(PV_DBOW(**dbow_kwargs))

        assert (
            len(self.modules) > 0
        ), f"{ParagraphVector.__name__} must be used with DM or DBOW architecture."

    @property
    def vector_size(self) -> int:
        return sum([module.vector_size for module in self.modules])

    def get_vector(self, id: Any) -> np.ndarray:
        vectors = [module.dv.get_vector(id) for module in self.modules]

        return np.concatenate(vectors)

    def save(self, dir_path: str) -> None:
        for module in self.modules:
            module_filepath = os.path.join(dir_path, "dbow" if module.dbow else "dm")
            module.save(module_filepath)

    def load(self, dir_path: str) -> None:
        new_modules = []
        for module_type in ["dm", "dbow"]:
            module_filepath = os.path.join(dir_path, module_type)
            if os.path.exists(module_filepath):
                module = PV_DM.load(module_filepath)
                assert module.dbow == (module_type == "dbow"), (
                    f"{ParagraphVector.load.__name__}: Loaded module does not"
                    " correspond to assumed architecture."
                )
                new_modules.append(module)

        assert (
            len(self.modules) > 0
        ), f"{ParagraphVector.load.__name__}: no model found."
        self.modules = new_modules
