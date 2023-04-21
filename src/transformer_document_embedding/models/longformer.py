import logging
from typing import Optional, Union, cast

import torch
import transformers.models.longformer.modeling_longformer as hf_longformer
from transformers import PreTrainedModel
from transformers.activations import get_activation

logger = logging.getLogger(__name__)


class LongformerConfig(hf_longformer.LongformerConfig):
    def __init__(
        self,
        classifier_activation: Optional[str] = None,
        classifier_hidden_size: Optional[int] = None,
        classifier_dropout_prob: Optional[float] = None,
        pooler_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Due to lacking interface in HF's PreTrainedConfig, we take the
        # approach: "If bad config value is set, something will crash later on."
        if classifier_hidden_size is None:
            classifier_hidden_size = kwargs.get("hidden_size", None)
        self.classifier_hidden_size = cast(int, classifier_hidden_size)

        if classifier_activation is None:
            classifier_activation = kwargs.get("hidden_act", None)
        self.classifier_activation = cast(str, classifier_activation)

        if classifier_dropout_prob is None:
            classifier_dropout_prob = kwargs.get("hidden_dropout_prob", None)
        self.classifier_dropout_prob = cast(float, classifier_dropout_prob)

        assert (
            pooler_type is None or pooler_type in AVAILABLE_POOLERS
        ), f"pooler_type must be one of {AVAILABLE_POOLERS}"
        self.pooler_type = pooler_type

        super().__init__(
            classifier_hidden_size=classifier_hidden_size,
            classifier_activation=classifier_activation,
            classifier_dropout_prob=classifier_dropout_prob,
            pooler_type=pooler_type,
            **kwargs,
        )


class LongformerMeanPooler(torch.nn.Module):
    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        summed = torch.sum(last_hidden_state * attention_mask[:, :, None], 1)
        row_lengths = torch.sum(attention_mask, 1)
        return summed / row_lengths[:, None]


class LongformerSumPooler(torch.nn.Module):
    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sum(last_hidden_state * attention_mask[:, :, None], 1)


class LongformerClsPooler(torch.nn.Module):
    def __init__(self, config: LongformerConfig) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.max_position_embeddings)

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # TODO: Transform CLS token's hidden state to weights with softmax.
        # Then do weighted sum.
        pass


AVAILABLE_POOLERS = {
    "sum": LongformerSumPooler,
    "mean": LongformerMeanPooler,
}


class LongformerClassificationHead(torch.nn.Module):
    """My configurable head for sentence-level classification tasks."""

    def __init__(self, config: LongformerConfig):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.classifier_hidden_size)
        self.dropout = torch.nn.Dropout(config.classifier_dropout_prob)
        self.out_proj = torch.nn.Linear(
            config.classifier_hidden_size, config.num_labels
        )
        self.activation_fn = get_activation(config.classifier_activation)

    def forward(self, hidden_state: torch.Tensor):
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.dense(hidden_state)
        hidden_state = self.activation_fn(hidden_state)
        hidden_state = self.dropout(hidden_state)
        output = self.out_proj(hidden_state)
        return output


class LongformerForTextEmbedding(hf_longformer.LongformerPreTrainedModel):
    def __init__(self, config: LongformerConfig) -> None:
        super().__init__(config)

        self.longformer = hf_longformer.LongformerModel(config)
        self.pooler = None
        if config.pooler_type is not None:
            self.pooler = AVAILABLE_POOLERS[config.pooler_type]()

        self.post_init()

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **_,  # Unused arguments which are given to models in HF framework.
    ) -> hf_longformer.LongformerBaseModelOutputWithPooling:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if global_attention_mask is None:
            # logger.info("Initializing global attention on CLS token...")
            # TODO: Better passing global_attention_mask explicitely
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1

        outputs = cast(
            hf_longformer.LongformerBaseModelOutputWithPooling,
            self.longformer(
                input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                head_mask=head_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            ),
        )

        outputs.pooler_output = (
            self.pooler(
                last_hidden_state=outputs.last_hidden_state,
                attention_mask=attention_mask,
            )
            if self.pooler is not None
            else outputs.last_hidden_state[:, 0]
        )

        return outputs


class LongformerForSequenceClassification(LongformerForTextEmbedding):
    def __init__(self, config: LongformerConfig) -> None:
        super().__init__(config)

        self.classifier = LongformerClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **_,  # Unused arguments which are given to models in HF framework.
    ) -> hf_longformer.LongformerSequenceClassifierOutput:
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        logits = self.classifier(outputs.pooler_output)

        return hf_longformer.LongformerSequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )
