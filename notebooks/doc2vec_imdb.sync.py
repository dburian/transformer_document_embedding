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
import importlib
import logging
import os

import datasets
import tensorflow as tf

import transformer_document_embedding as tde

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
# %%
tde = importlib.reload(tde)
# %%
LOG_DIR = "../results/notebooks/doc2vec_imdb"
task = tde.tasks.IMDBClassification(data_size_limit=100)
model = tde.models.Doc2VecIMDB(log_dir=LOG_DIR, use_dm=False, dbow_kwargs={"epochs": 1})
# %%
doc2vec_train = datasets.combine.concatenate_datasets(
    [task.train, task.unsupervised, task.test]
)
doc2vec_train = doc2vec_train.shuffle()
model._doc2vec.train(doc2vec_train)
# %%
ds = model._cls_head_dataset(task.train, training=False)

# %%
label_ds = task.train.to_tf_dataset(1, columns=["label"]).unbatch()
list(label_ds.take(4).as_numpy_iterator())
# %%
ds = tf.data.Dataset.zip((ds, label_ds))

# %%
ds = ds.shuffle(25000)
# %%
ds = ds.batch(2)
# %%
list(ds.take(4).as_numpy_iterator())
# %%
ds = model._cls_head_dataset(task.train)
