from __future__ import annotations
from functools import partial

from torcheval.metrics import Max, Mean, Metric, Sum
from transformer_document_embedding.datasets import col

from transformer_document_embedding.pipelines.torch.train import TorchTrainPipeline
from transformer_document_embedding.utils import cca_losses
from transformer_document_embedding.utils.metrics import (
    EmbeddingCosineDistanceWithCol,
    EmbeddingMSEWithCol,
    TrainingMetric,
    WindowedAbsCorrelationMetric,
    WindowedAbsCrossCorrelationMetric,
    WindowedCCAMetricZoo,
)
from typing import TYPE_CHECKING, Optional

import torch

from transformer_document_embedding.utils.similarity_losses import ContrastiveLoss

if TYPE_CHECKING:
    from transformer_document_embedding.models.transformer import (
        TransformerEmbedder,
    )
    from transformer_document_embedding.heads.structural_contextual_head import (
        StructuralContextualHead,
    )


def log_max_abs_grad(
    metric: Metric,
    *_,
    param_name: str,
    model: torch.nn.Module,
) -> None:
    param = model.get_parameter(param_name)
    grad = param.grad
    if grad is None:
        metric.update(torch.tensor([torch.nan], device=param.device))
    else:
        metric.update(grad.abs().max())


class _Student(torch.nn.Module):
    def __init__(
        self,
        encoder: TransformerEmbedder,
        head: StructuralContextualHead,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.head = head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        if (
            kwargs.get(col.STRUCTURAL_EMBED, None) is not None
            or kwargs.get(col.CONTEXTUAL_EMBED, None) is not None
        ):
            head_kwargs = {
                col.EMBEDDING: outputs[col.EMBEDDING],
                col.LENGTH: kwargs[col.LENGTH],
                col.STRUCTURAL_EMBED: kwargs.get(col.STRUCTURAL_EMBED, None),
                col.CONTEXTUAL_EMBED: kwargs.get(col.CONTEXTUAL_EMBED, None),
            }
            outputs.update(self.head(**head_kwargs))

        return outputs


class StudentTrainPipeline(TorchTrainPipeline):
    def get_encompassing_model(
        self,
        model: TransformerEmbedder,
        head: StructuralContextualHead,
    ) -> torch.nn.Module:
        return _Student(model, head)

    def get_train_metrics(self, log_freq: int, model: _Student) -> list[TrainingMetric]:
        supers_metrics = super(StudentTrainPipeline, self).get_train_metrics(
            log_freq, model
        )

        return (
            supers_metrics
            + self.get_evaluation_metrics(log_freq, model)
            + self.get_loss_metrics(log_freq, model)
            + self.get_projection_metrics(log_freq, model)
            + self.get_grad_metrics(log_freq, model)
            + [
                TrainingMetric(
                    "mean_length",
                    Mean(),
                    log_freq,
                    lambda metric, _, batch: metric.update(batch[col.LENGTH]),
                ),
            ]
        )

    def get_evaluation_metrics(
        self, log_freq: int, model: _Student
    ) -> list[TrainingMetric]:
        eval_metrics = [
            EmbeddingMSEWithCol("structural", log_freq, col_name=col.STRUCTURAL_EMBED),
            EmbeddingCosineDistanceWithCol(
                "structural", log_freq, col_name=col.STRUCTURAL_EMBED
            ),
        ]

        structural_length_thres = model.head.max_structural_length
        if structural_length_thres is not None:
            eval_metrics.extend(
                [
                    EmbeddingMSEWithCol(
                        "structural",
                        log_freq,
                        col_name=col.STRUCTURAL_EMBED,
                        max_input_length=structural_length_thres,
                    ),
                    EmbeddingCosineDistanceWithCol(
                        "structural",
                        log_freq,
                        col_name=col.STRUCTURAL_EMBED,
                        max_input_length=structural_length_thres,
                    ),
                ]
            )

        return eval_metrics

    def get_loss_metrics(self, log_freq: int, model: _Student) -> list[TrainingMetric]:
        """Metrics logging parts of loss."""
        loss_metrics = []

        if model.head.contextual_head is not None:
            loss_metrics.append(
                TrainingMetric(
                    "mean_contextual_loss",
                    Mean(),
                    log_freq,
                    lambda metric, outputs, _: metric.update(
                        outputs["contextual_loss"]
                    ),
                )
            )

            if isinstance(model.head.contextual_head.loss_fn, cca_losses.SoftCCALoss):
                loss_metrics.extend(
                    [
                        TrainingMetric(
                            "mean_projection_l2_norm",
                            Mean(),
                            log_freq,
                            lambda metric, outputs, _: metric.update(
                                outputs["contextual_l2"]
                            ),
                        ),
                        TrainingMetric(
                            "mean_projection_sdl1",
                            Mean(),
                            log_freq,
                            lambda metric, outputs, _: metric.update(
                                outputs["contextual_sdl1"]
                            ),
                        ),
                        TrainingMetric(
                            "mean_projection_sdl2",
                            Mean(),
                            log_freq,
                            lambda metric, outputs, _: metric.update(
                                outputs["contextual_sdl2"]
                            ),
                        ),
                    ]
                )

            if isinstance(model.head.contextual_head.loss_fn, ContrastiveLoss):
                loss_metrics.extend(
                    [
                        TrainingMetric(
                            # TODO: Rename contrastive to MaxMarginals
                            "contextual_contrastive_positive",
                            Mean(),
                            log_freq,
                            lambda metric, outputs, _: metric.update(
                                outputs["contextual_contrastive_positive"]
                            ),
                        ),
                        TrainingMetric(
                            "contextual_contrastive_negative",
                            Mean(),
                            log_freq,
                            lambda metric, outputs, _: metric.update(
                                outputs["contextual_contrastive_negative"]
                            ),
                        ),
                    ]
                )

        if model.head.structural_head is not None:
            loss_metrics.extend(
                [
                    TrainingMetric(
                        "mean_structural_loss",
                        Mean(),
                        log_freq,
                        lambda metric, outputs, _: metric.update(
                            outputs["structural_loss"]
                        ),
                    ),
                    TrainingMetric(
                        "mean_structural_mask",
                        Mean(),
                        log_freq,
                        lambda metric, outputs, _: metric.update(
                            outputs["structural_mask"]
                        ),
                    ),
                    TrainingMetric(
                        "structural_steps",
                        Sum(),
                        log_freq,
                        lambda metric, outputs, _: metric.update(
                            outputs["structural_mask"] / self.batch_size
                        ),
                        reset_after_log=False,
                    ),
                ]
            )

        if isinstance(model.head.structural_head, ContrastiveLoss):
            loss_metrics.extend(
                [
                    TrainingMetric(
                        "structural_contrastive_positive",
                        Mean(),
                        log_freq,
                        lambda metric, outputs, _: metric.update(
                            outputs["structural_contrastive_positive"]
                        ),
                    ),
                    TrainingMetric(
                        "structural_contrastive_negative",
                        Mean(),
                        log_freq,
                        lambda metric, outputs, _: metric.update(
                            outputs["structural_contrastive_negative"]
                        ),
                    ),
                ]
            )

        return loss_metrics

    def get_projection_metrics(
        self, log_freq: int, model: _Student
    ) -> list[TrainingMetric]:
        if not isinstance(model.head.contextual_head, cca_losses.ProjectionLoss):
            return []

        projection_metrics = []

        def update_with_projection(
            metric, outputs, _, *, net1_layer_ind: int, net2_layer_ind: int
        ) -> None:
            metric.update(
                outputs["contextual_projected_views1"][net1_layer_ind],
                outputs["contextual_projected_views2"][net2_layer_ind],
            )

        def update_with_projection1(metric, outputs, _, layer_ind):
            metric.update(outputs["contextual_projected_views1"][layer_ind])

        def update_with_projection2(metric, outputs, _, layer_ind):
            metric.update(outputs["contextual_projected_views2"][layer_ind])

        net1_feats = model.head.contextual_head.net1.features
        net2_feats = model.head.contextual_head.net2.features

        def reverse_projcetion_layers():
            layer_counts = (len(net1_feats), len(net2_feats))
            for back_steps in range(1, max(*layer_counts) + 1):
                yield (max(0, layer_c - back_steps) for layer_c in layer_counts)

        for net1_layer_ind, net2_layer_ind in reverse_projcetion_layers():
            min_dim = min(net1_feats[net1_layer_ind], net2_feats[net2_layer_ind])

            both_views_update_fn = partial(
                update_with_projection,
                net1_layer_ind=net1_layer_ind,
                net2_layer_ind=net2_layer_ind,
            )
            view1_update_fn = partial(update_with_projection1, layer_ind=net1_layer_ind)
            view2_update_fn = partial(update_with_projection2, layer_ind=net2_layer_ind)

            layer_specifier = f"[{net1_layer_ind},{net2_layer_ind}]"
            for multiplier in [5, 10]:
                cca_metric = WindowedCCAMetricZoo(
                    n_components=min_dim, window_size=min_dim * multiplier
                )

                metric_name = (
                    f"cca_projection{layer_specifier}_"
                    f"{cca_metric.n_components}x{cca_metric.window_size}"
                )

                projection_metrics.append(
                    TrainingMetric(
                        metric_name,
                        cca_metric,
                        log_freq,
                        both_views_update_fn,
                        reset_after_log=False,
                    ),
                )

            crosscorr_window = min_dim * 5
            net1_window = net1_feats[net1_layer_ind] * 5
            net2_window = net2_feats[net2_layer_ind] * 5
            projection_metrics.extend(
                [
                    TrainingMetric(
                        f"crosscorr_projection{layer_specifier}_x{crosscorr_window}",
                        WindowedAbsCrossCorrelationMetric(crosscorr_window),
                        log_freq,
                        both_views_update_fn,
                        reset_after_log=False,
                    ),
                    TrainingMetric(
                        f"corr_student_projection[{net1_layer_ind}]_x{net1_window}",
                        WindowedAbsCorrelationMetric(net1_window),
                        log_freq,
                        view1_update_fn,
                        reset_after_log=False,
                    ),
                    TrainingMetric(
                        f"corr_contextual_projection[{net2_layer_ind}]_x{net2_window}",
                        WindowedAbsCorrelationMetric(net2_window),
                        log_freq,
                        view2_update_fn,
                        reset_after_log=False,
                    ),
                ]
            )

        return projection_metrics

    def get_grad_metrics(self, log_freq: int, model: _Student) -> list[TrainingMetric]:
        grad_metrics = [
            TrainingMetric(
                "max_abs_student_grad",
                Max(),
                log_freq,
                partial(
                    log_max_abs_grad,
                    param_name="encoder.transformer.encoder.layer.11.output.dense.weight",
                    model=model,
                ),
            ),
        ]

        for net_name in ["net1", "net2"]:
            net = getattr(model.head.contextual_head, net_name)
            layer_count = len(net.layers)
            sample_projection_weight_path = (
                f"head.contextual_head.{net_name}.layers.{layer_count-1}.0.weight"
            )

            grad_metrics.append(
                TrainingMetric(
                    f"max_abs_{net_name}_grad",
                    Max(),
                    log_freq,
                    partial(
                        log_max_abs_grad,
                        param_name=sample_projection_weight_path,
                        model=model,
                    ),
                )
            )

        return grad_metrics
