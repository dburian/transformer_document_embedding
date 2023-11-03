import tensorflow_datasets as tfds

train = tfds.load("c4/en", split="train[:1%]")

builder = tfds.builder("c4/en")
info = builder.info
