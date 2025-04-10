"""PyTorch KDAdapter model."""

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from model.configuration_kdadapter import KDAdapterConfig
from model.modeling_llama import LlamaForCausalLM
from model.modeling_qformer import QFormerModel
from transformers import AutoConfig, BertModel, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.utils import ModelOutput, logging

logger = logging.get_logger(__name__)


@dataclass
class KDAdapterForConditionalGenerationOutput(ModelOutput):
    loss: Optional[tuple[torch.FloatTensor]] = None
    logits: Optional[tuple[torch.FloatTensor]] = None
    bert_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[tuple[torch.FloatTensor]] = None

    def to_tuple(self):
        return tuple(
            self[k]
            if k not in ["bert_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class KDAdapterPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = KDAdapterConfig
    base_model_prefix = "kdadapter"
    supports_gradient_checkpointing = True
    _no_split_modules = ["QFormerAttention", "LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _keep_in_fp32_modules = ["wo"]

    def _init_weights(self, module):
        """Initialize the weights"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class KDAdapterForConditionalGeneration(KDAdapterPreTrainedModel, GenerationMixin):
    config_class = KDAdapterConfig
    _keys_to_ignore_on_load_unexpected = [
        r"query_visual_projection",
        r"logit_scale",
        r"visual_tb",
        r"visual_decoder",
        r"text_tb",
        r"text_decoder",
    ]
    _keys_to_ignore_on_load_missing = [r"ka_", r"attn_"]

    def __init__(
        self,
        config: KDAdapterConfig = None,
        bert_model: BertModel = None,
        qformer: QFormerModel = None,
        language_model: LlamaForCausalLM = None,
    ):
        if config is None and (
            bert_model is None or qformer is None or language_model is None
        ):
            raise ValueError("Either a configuration or models have to be provided")

        if config is None:
            config = KDAdapterConfig.from_configs(
                bert_model.config, qformer.config, language_model.config
            )
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(
                    f"config: {config} has to be of type {self.config_class}"
                )

        super().__init__(config)

        if bert_model is None:
            bert_model = BertModel(config.bert_config)

        if qformer is None:
            qformer = QFormerModel(config.qformer_config)

        if language_model is None:
            language_model = LlamaForCausalLM(config.llm_config)

        self.bert_model = bert_model
        self.query_tokens = nn.Parameter(
            torch.empty(
                1, config.num_query_tokens, config.qformer_config.hidden_size
            ).normal_()
        )
        self.qformer = qformer

        self.language_projection = nn.Linear(
            config.qformer_config.hidden_size, config.llm_config.hidden_size
        )
        # Update _tied_weights_keys using the base model used.
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"language_model.{k}" for k in language_model._tied_weights_keys
            ]

        self.language_model = language_model

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.bert_model.config = self.config.bert_config
        self.qformer.config = self.config.qformer_config
        self.language_model.config = self.config.llm_config

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if (
            len(hf_device_map) > 1
            and "language_model" not in hf_device_map
            and torch.cuda.device_count() > 1
        ):
            # warn users about unexpected behavior when using multi-GPU + BLIP-2 + `accelerate`.
            logger.warning(
                "The `language_model` is not in the `hf_device_map` dictionary and you are running your script" 
                " in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`."
                " Please pass a `device_map` that contains `language_model` to remove this warning."
                " Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for"
                " more details on creating a `device_map` for large models.",
            )

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = (
                True  # For `generate` compatibility
            )

    def forward(
        self,
        llm_input_ids: torch.LongTensor,
        llm_attention_mask: Optional[torch.LongTensor] = None,
        bert_input_ids: Optional[torch.LongTensor] = None,
        bert_attention_mask: Optional[torch.LongTensor] = None,
        t2k_llm_input_ids: Optional[torch.LongTensor] = None,
        t2k_llm_attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, KDAdapterForConditionalGenerationOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # forward the texts through the text encoder to get text embeddings of shape (batch_size, seq_len, hidden_size)
        bert_outputs = self.bert_model(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        bert_embeds = bert_outputs[0]  # last_hidden_state

        # forward the query tokens through the QFormer, using the text embeddings for cross-attention
        query_tokens = self.query_tokens.expand(bert_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=bert_embeds,
            encoder_attention_mask=bert_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # use the language model, conditioned on the query outputs and the prompt
        query_embeds = self.language_projection(query_output)
        expected_device = query_embeds.device
        query_attention_mask = torch.ones(
            query_embeds.size()[:-1], dtype=torch.long, device=expected_device
        )

        llm_inputs_embeds = self.language_model.get_input_embeddings()(
            llm_input_ids
        ).to(expected_device)
        llm_inputs_embeds = torch.cat(
            [llm_inputs_embeds[:, :27, :], query_embeds, llm_inputs_embeds[:, 27:, :]],
            dim=1,
        )

        if llm_attention_mask is None:
            llm_attention_mask = torch.ones_like(llm_input_ids)
        llm_attention_mask = llm_attention_mask.to(expected_device)
        llm_attention_mask = torch.cat(
            [
                llm_attention_mask[:, :27],
                query_attention_mask,
                llm_attention_mask[:, 27:],
            ],
            dim=1,
        )

        outputs = self.language_model(
            inputs_embeds=llm_inputs_embeds,
            attention_mask=llm_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            know_prompts=query_embeds,
        )
        logits = outputs.logits if return_dict else outputs[0]
        loss = None
        # we compute the loss here since we need to take into account the sequence length of the query embeds
        if labels is not None:
            logits = logits[:, -labels.size(1) :, :]

            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)

            loss_fct = CrossEntropyLoss(reduction="mean")
            loss = loss_fct(
                shift_logits.view(-1, self.language_model.config.vocab_size),
                shift_labels.view(-1),
            )

        #  use the language model, generate the knowledge prompts
        if t2k_llm_input_ids is not None:
            t2k_llm_embeds = self.language_model.get_input_embeddings()(
                t2k_llm_input_ids
            ).to(expected_device)
            if t2k_llm_attention_mask is None:
                t2k_llm_attention_mask = torch.ones_like(t2k_llm_input_ids)
            t2k_llm_attention_mask = t2k_llm_attention_mask.to(expected_device)

            t2k_input_embeds = []
            t2k_attention_masks = []
            llm_ones_counts = t2k_llm_attention_mask.sum(dim=1)

            # append the query_embeds before the pad token
            for i, ones_count in enumerate(llm_ones_counts):
                t2k_input_embeds.append(
                    torch.cat(
                        [
                            t2k_llm_embeds[i, :ones_count],
                            query_embeds[i],
                            t2k_llm_embeds[i, ones_count:],
                        ],
                        dim=0,
                    )
                )

                t2k_attention_masks.append(
                    torch.cat(
                        [
                            t2k_llm_attention_mask[i, :ones_count],
                            query_attention_mask[i],
                            t2k_llm_attention_mask[i, ones_count:],
                        ],
                        dim=0,
                    )
                )

            # stack the results
            t2k_input_embeds = torch.stack(t2k_input_embeds)
            t2k_attention_masks = torch.stack(t2k_attention_masks)

            t2k_outputs = self.language_model.model(
                inputs_embeds=t2k_input_embeds,
                attention_mask=t2k_attention_masks,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            query_len = query_embeds.size(1)
            query_llm_output = torch.stack(
                [
                    t2k_outputs[0][i, ones_count : ones_count + query_len]
                    for i, ones_count in enumerate(llm_ones_counts - 1)
                ]
            )

            loss_t2k = nn.functional.mse_loss(query_llm_output, query_embeds)
            if loss is not None:
                loss += loss_t2k * 0.5
            else:
                loss = loss_t2k * 0.5

        if not return_dict:
            output = (logits, bert_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return KDAdapterForConditionalGenerationOutput(
            loss=loss,
            logits=logits,
            bert_outputs=bert_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
        self,
        llm_input_ids: torch.LongTensor,
        llm_attention_mask: Optional[torch.LongTensor] = None,
        bert_input_ids: Optional[torch.LongTensor] = None,
        bert_attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size = bert_input_ids.shape[0]

        bert_outputs = self.bert_model(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask,
            return_dict=True,
        )
        bert_embeds = bert_outputs[0]  # last_hidden_state

        query_tokens = self.query_tokens.expand(bert_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=bert_embeds,
            encoder_attention_mask=bert_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs[0]

        query_embeds = self.language_projection(query_output)
        expected_device = query_embeds.device
        query_attention_mask = torch.ones(
            query_embeds.size()[:-1], dtype=torch.long, device=expected_device
        )

        if llm_input_ids is None:
            llm_input_ids = (
                torch.LongTensor([[self.config.llm_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(bert_embeds.device)
            )
        if llm_attention_mask is None:
            llm_attention_mask = torch.ones_like(llm_input_ids)

        llm_inputs_embeds = self.language_model.get_input_embeddings()(
            llm_input_ids
        ).to(expected_device)
        llm_inputs_embeds = torch.cat(
            [llm_inputs_embeds[:, :27, :], query_embeds, llm_inputs_embeds[:, 27:, :]],
            dim=1,
        )
        llm_attention_mask = torch.cat(
            [
                llm_attention_mask[:, :27],
                query_attention_mask,
                llm_attention_mask[:, 27:],
            ],
            dim=1,
        )

        outputs = self.language_model.generate(
            inputs_embeds=llm_inputs_embeds,
            attention_mask=llm_attention_mask,
            know_prompts=query_embeds,
            **generate_kwargs,
        )

        return outputs

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_bert_qformer_llm_pretrained(
        cls,
        bert_model_name_or_path: str,
        qformer_name_or_path: str,
        language_model_name_or_path: str,
        cross_attention_frequency=2,
        num_query_tokens=32,
        attn_mode=False,
        attn_scalar=1.0,
        ffn_mode=False,
        ffn_bottleneck=512,
        ffn_adapter_scalar=1.0,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        bert_model = BertModel.from_pretrained(bert_model_name_or_path)

        qformer_config = AutoConfig.from_pretrained(qformer_name_or_path)
        qformer_config.encoder_hidden_size = bert_model.config.hidden_size
        qformer_config.cross_attention_frequency = cross_attention_frequency
        qformer_config.query_length = num_query_tokens
        qformer = QFormerModel.from_pretrained(
            qformer_name_or_path, config=qformer_config
        )

        llm_config = AutoConfig.from_pretrained(language_model_name_or_path)
        llm_config.attn_mode = attn_mode
        llm_config.attn_query_len = num_query_tokens
        llm_config.attn_scalar = attn_scalar
        llm_config.ffn_mode = ffn_mode
        llm_config.ffn_bottleneck = ffn_bottleneck
        llm_config.ffn_adapter_scalar = ffn_adapter_scalar
        language_model = LlamaForCausalLM.from_pretrained(
            language_model_name_or_path, config=llm_config, *model_args, **kwargs
        )

        # instantiate config with corresponding kwargs
        config = KDAdapterConfig.from_configs(
            bert_model.config, qformer.config, language_model.config
        )

        # init model
        model = cls(
            config=config,
            bert_model=bert_model,
            qformer=qformer,
            language_model=language_model,
        )

        return model
