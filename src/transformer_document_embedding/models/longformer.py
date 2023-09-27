import logging
from typing import Optional, cast

import torch
import transformers.models.longformer.modeling_longformer as hf_longformer

from transformer_document_embedding.models.hf_layers import (
    AVAILABLE_POOLERS,
    ClassificationConfigMixin,
    ClassificationHead,
    PooledConfigMixin,
)

logger = logging.getLogger(__name__)


class LongformerConfig(
    PooledConfigMixin, ClassificationConfigMixin, hf_longformer.LongformerConfig
):
    pass


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
        # TODO: Check how outputs are handled during training.

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if global_attention_mask is None:
            # logger.info("Initializing global attention on CLS token...")
            # TODO: Better passing global_attention_mask explicitly
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

        self.classifier = ClassificationHead(config)

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
