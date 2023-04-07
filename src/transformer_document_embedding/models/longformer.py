import logging
from typing import Optional

import torch
from transformers import (LongformerConfig, LongformerModel,
                          LongformerPreTrainedModel)
from transformers.activations import get_activation
from transformers.models.longformer.modeling_longformer import \
    LongformerSequenceClassifierOutput

logger = logging.getLogger(__name__)


class ClsLongformerConfig(LongformerConfig):
    def __init__(
        self,
        classifier_activation: Optional[str] = None,
        classifier_hidden_size: Optional[int] = None,
        classifier_dropout_prob: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if classifier_hidden_size is None:
            classifier_hidden_size = self.hidden_size
        self.classifier_hidden_size = classifier_hidden_size

        if classifier_activation is None:
            classifier_activation = self.hidden_act
        self.cls_head_activation = classifier_activation

        if classifier_dropout_prob is None:
            classifier_dropout_prob = self.hidden_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob


class TDELongformerClassificationHead(torch.nn.Module):
    """My configurable head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.classifier_hidden_size)
        self.dropout = torch.nn.Dropout(config.classifier_dropout_prob)
        self.out_proj = torch.nn.Linear(
            config.classifier_hidden_size, config.num_labels
        )
        self.activation_fn = get_activation(config.classifier_activation)

    def forward(self, hidden_states, **_):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output


class TDELongformerForSequenceClassification(LongformerPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # TODO: Classification on pooled hidden states?
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.classifier = TDELongformerClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    # pylint: disable=too-many-locals
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> LongformerSequenceClassifierOutput:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if global_attention_mask is None:
            logger.info("Initializing global attention on CLS token...")
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return LongformerSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )
