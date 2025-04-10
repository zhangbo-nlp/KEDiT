# coding=utf-8

"""KDAdapter model configuration"""

from transformers import Blip2QFormerConfig, BertConfig, LlamaConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class KDAdapterConfig(PretrainedConfig):
    model_type = "kdadapter"

    def __init__(
            self,
            bert_config=None,
            qformer_config=None,
            llm_config=None,
            cross_attention_frequency=2,
            num_query_tokens=32,
            attn_mode=False,
            attn_scalar=1.,
            ffn_mode=False,
            ffn_bottleneck=512,
            ffn_adapter_scalar=1.,
            **kwargs
    ):
        super().__init__(**kwargs)

        if bert_config is None:
            bert_config = {}
            logger.info("bert_config is None. initializing the BertConfig with default values.")

        if qformer_config is None:
            qformer_config = {}
            logger.info("qformer_config is None. Initializing the Blip2QFormerConfig with default values.")

        if llm_config is None:
            llm_config = {}
            logger.info("llm_config is None. Initializing the LlamaConfig with default values.")

        self.bert_config = BertConfig(**bert_config)

        self.qformer_config = BertConfig(**qformer_config)
        self.qformer_config.encoder_hidden_size = getattr(self.qformer_config, "encoder_hidden_size",  self.bert_config.hidden_size)
        self.qformer_config.cross_attention_frequency = getattr(self.qformer_config, "cross_attention_frequency", cross_attention_frequency)
        self.qformer_config.query_length = getattr(self.qformer_config, "query_length", num_query_tokens)

        self.llm_config = LlamaConfig(**llm_config)
        self.llm_config.attn_mode = getattr(self.llm_config, "attn_mode", attn_mode)
        self.llm_config.attn_query_len = getattr(self.llm_config, "attn_query_len", num_query_tokens)
        self.llm_config.attn_scalar = getattr(self.llm_config, "attn_scalar", attn_scalar)
        self.llm_config.ffn_mode = getattr(self.llm_config, "ffn_mode", ffn_mode)
        self.llm_config.ffn_bottleneck = getattr(self.llm_config, "ffn_bottleneck", ffn_bottleneck)
        self.llm_config.ffn_adapter_scalar = getattr(self.llm_config, "ffn_adapter_scalar", ffn_adapter_scalar)

        self.tie_word_embeddings = self.llm_config.tie_word_embeddings
        self.is_encoder_decoder = self.llm_config.is_encoder_decoder

        self.num_query_tokens = self.qformer_config.query_length
        assert self.qformer_config.query_length == self.llm_config.attn_query_len, "Query length must be equal"

        self.initializer_range = 0.02

    @classmethod
    def from_configs(
            cls,
            bert_config: BertConfig,
            qformer_config: Blip2QFormerConfig,
            llm_config: LlamaConfig,
            **kwargs,
    ):
        return cls(
            bert_config=bert_config.to_dict(),
            qformer_config=qformer_config.to_dict(),
            llm_config=llm_config.to_dict(),
            **kwargs,
        )
