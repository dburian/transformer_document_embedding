import os

# Importing torch after tensorflow causes wierd stuff to happen on AIC
# TODO: File an issue
import torch

# Report only TF errors by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from . import baselines, experiments, models, tasks
