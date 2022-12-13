# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %%
import functools
import logging
import os
from typing import Any, Callable, Iterable

import datasets
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import tensorflow as tf

import transformer_document_embedding as tde

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
# %%
imdb = datasets.load_dataset("imdb")
print(imdb)
# %%
def get_id(id: int, split: str) -> int:
    split_ind = ["test", "train", "unsuper"].index(split)
    return split_ind * 25000


def add_doc_id(
    example: dict[str, Any], idx: int, get_id: Callable[[int], int]
) -> dict[str, Any]:
    return {"id": get_id(idx), "text": example["text"], "label": example["label"]}


train_add_doc_id = functools.partial(
    add_doc_id, get_id=lambda idx: get_id(idx, "train")
)
unsuper_add_doc_id = functools.partial(
    add_doc_id, get_id=lambda idx: get_id(idx, "unsuper")
)
test_add_doc_id = functools.partial(add_doc_id, get_id=lambda idx: get_id(idx, "train"))
train = imdb["train"].map(train_add_doc_id, with_indices=True)
unsuper = imdb["unsupervised"].map(unsuper_add_doc_id, with_indices=True)
test = imdb["test"].map(test_add_doc_id, with_indices=True)
# %%
for i, x in enumerate(train):
    print(x)
    if i > 5:
        break
# %%
def preprocess_document(text: str) -> list[str]:
    """Preprocesses document according to Paragraph Vector paper."""
    words = text.split()
    while len(words) < 10:
        words.insert(0, "NULL")

    return words


class MyGensimCorpus:
    def __iter__(self) -> Iterable[TaggedDocument]:
        for doc in train:
            yield TaggedDocument(preprocess_document(doc["text"]), [doc["id"]])

        for doc in unsuper:
            yield TaggedDocument(preprocess_document(doc["text"]), [doc["id"]])


for i, x in enumerate(MyGensimCorpus()):
    print(x)
    if i > 3:
        break

# %%
model = Doc2Vec(vector_size=400, window=5, dm_concat=1, workers=6)
model.build_vocab(MyGensimCorpus())
# %%
model.train(MyGensimCorpus(), total_examples=model.corpus_count, epochs=1)
# %%
model.dv[25000]

# %%


def add_feature_vec(doc: dict[str, Any]) -> dict[str, Any]:
    feature_vector = model.dv[doc["id"]]
    return {"features": feature_vector, "label": doc["label"]}


softmax_train = train.map(add_feature_vec, remove_columns=["text", "id"]).to_tf_dataset(
    1
)
softmax_train = softmax_train.unbatch()
softmax_train = softmax_train.map(lambda doc: (doc["features"], doc["label"]))
softmax_train = softmax_train.shuffle(1000)
softmax_train = softmax_train.batch(2)
# tf_dataset_softmax_train = tf.data.Dataset.from_generator(
#     lambda: softmax_train,
#     output_types=(tf.float32, tf.int32),
#     output_shapes=((1,), (0,)),
# )
for x in softmax_train.take(5):
    print(x)
# %%
softmax_net = tf.keras.Sequential(
    [
        tf.keras.layers.Input(400),
        tf.keras.layers.Dense(50, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ]
)
softmax_net.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.losses.BinaryCrossentropy(),
    metrics=[tf.metrics.BinaryAccuracy()],
)
# %%
softmax_net.fit(softmax_train)
