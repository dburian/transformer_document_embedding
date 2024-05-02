# %%
from __future__ import annotations
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np

# %%
ds_size = 10000
positives = ds_size * 0.5
negatives = ds_size * 0.5

samples = 50

# %%


def accuracy(tp: float, tn: float, fp: float, fn: float) -> float:
    return (tp + tn) / (fp + fn + tp + tn)


def f1(tp: float, tn: float, fp: float, fn: float) -> float:
    return 2 * tp / (2 * tp + fp + fn)


def precision(tp: float, tn: float, fp: float, fn: float) -> float:
    return tp / (tp + fp)


def recall(tp: float, tn: float, fp: float, fn: float) -> float:
    return tp / (tp + fn)


def fp(tn: float) -> float:
    return negatives - tn


def fn(tp: float) -> float:
    return positives - tp


def draw_metric(
    metric_fn: Callable[[float, float, float, float], float],
    fig: plt.figure.Figure,
    ax: plt.axes.Axes,
) -> None:
    tps = np.linspace(0.01, positives, samples)
    tns = np.linspace(0.01, negatives, samples)

    metric = np.array([[metric_fn(tp, tn, fp(tn), fn(tp)) for tp in tps] for tn in tns])

    ax.set_xlabel("TP")
    ax.set_ylabel("TN")

    contour_set = ax.contour(tps, tns, metric, levels=samples)
    fig.colorbar(contour_set)


# %%
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.reshape(-1)

for metric, ax in zip([f1, accuracy, precision, recall], axes, strict=True):
    draw_metric(metric, fig=fig, ax=ax)
    ax.set_title(metric.__name__)


# %%
