import logging
from typing import Optional, cast

import torch
import transformers.models.big_bird.modeling_big_bird as hf_bigbird

from transformer_document_embedding.models.hf_layers import (
    AVAILABLE_POOLERS,
    ClassificationConfigMixin,
    ClassificationHead,
    PooledConfigMixin,
)

logger = logging.getLogger(__name__)


class BigBirdConfig(
    PooledConfigMixin, ClassificationConfigMixin, hf_bigbird.BigBirdConfig
):
    pass


# pylint: disable=abstract-method
class BigBirdForTextEmbedding(hf_bigbird.BigBirdPreTrainedModel):
    def __init__(self, config: BigBirdConfig) -> None:
        super().__init__(config)

        self.bert = hf_bigbird.BigBirdModel(config, add_pooling_layer=False)
        self.pooler = None
        if config.pooler_type is not None:
            self.pooler = AVAILABLE_POOLERS[config.pooler_type]()

        self.post_init()

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **_,  # Unused arguments which are usually given to models in HF framework.
    ) -> hf_bigbird.BaseModelOutputWithPoolingAndCrossAttentions:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = cast(
            hf_bigbird.BaseModelOutputWithPoolingAndCrossAttentions,
            self.bert(
                input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            ),
        )

        outputs.pooler_output = outputs.last_hidden_state[:, 0]

        if self.pooler is not None:
            outputs.pooler_output = self.pooler(
                last_hidden_state=outputs.last_hidden_state,
                attention_mask=attention_mask,
            )

        return outputs


class BigBirdForSequenceClassification(BigBirdForTextEmbedding):
    def __init__(self, config: BigBirdConfig) -> None:
        super().__init__(config)

        self.classifier = ClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **_,  # Unused arguments which are given to models in HF framework.
    ) -> hf_bigbird.SequenceClassifierOutput:
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        logits = self.classifier(outputs.pooler_output)

        return hf_bigbird.SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
