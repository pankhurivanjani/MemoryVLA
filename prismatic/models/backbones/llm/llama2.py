"""
llama2.py

Class definition for all LLMs derived from LlamaForCausalLM.
"""

import os
from typing import Optional, Sequence, Type

import torch
from torch import nn as nn
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import (
    LLaMa2ChatPromptBuilder,
    PromptBuilder,
    PurePromptBuilder,
    VicunaV15ChatPromptBuilder,
)

# [JSC] Which repo the 7B base is pulled from. prismatic rebuilds the LLM from this repo and
# always takes the tokenizer from it, but the CogACT/MemoryVLA checkpoint then overwrites every
# LLM weight -- so in a fine-tuning run only the config and tokenizer actually come from here.
# meta-llama/Llama-2-7b-hf is gated; on an account without access `HfApi().model_info()` still
# succeeds and only a real file fetch returns 403. Set MEMVLA_LLAMA2_REPO to a mirror
# (e.g. NousResearch/Llama-2-7b-hf, a verbatim re-upload) to run before access is granted, and
# unset it afterwards to go back to the official repo with no code change.
_LLAMA2_7B_REPO = os.environ.get("MEMVLA_LLAMA2_REPO", "meta-llama/Llama-2-7b-hf")

# [JSC] flash-attn has no aarch64 wheel and is not built in this env. The class default below is
# True and `prismatic/models/materialize.py` never passes the flag, so TRAINING (which runs with
# inference_mode=False) hard-fails in transformers' _check_and_enable_flash_attn_2 with
# "FlashAttention2 has been toggled on, but ... flash_attn seems to be not installed".
# Only the eval path escapes, because inference_mode=True forces the flag False in base_llm.py.
# With it off, transformers 4.40 selects SDPA, which is mathematically the same attention; and
# prismatic's sequences are ~290 tokens (256 fused vision + instruction), so the throughput cost
# of not having the fused kernel is small. Set MEMVLA_NO_FLASH=0 if flash_attn ever gets built.
_USE_FLASH_ATTN = os.environ.get("MEMVLA_NO_FLASH", "0") != "1"

# Registry =>> Support LLaMa-2 Models (from HF Transformers)
# fmt: off
LLAMA2_MODELS = {
    # === Pure Meta LLaMa-2 (non-instruct/chat-tuned) Models ===
    "llama2-7b-pure": {
        "llm_family": "llama2", "llm_cls": LlamaForCausalLM, "hf_hub_path": _LLAMA2_7B_REPO
    },

    "llama2-13b-pure": {
        "llm_family": "llama2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-2-13b-hf"
    },

    # === Meta LLaMa-2 Chat Models ===
    "llama2-7b-chat": {
        "llm_family": "llama2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-2-7b-chat-hf"
    },

    "llama2-13b-chat": {
        "llm_family": "llama2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "meta-llama/Llama-2-13b-chat-hf"
    },

    # === Vicuna v1.5 Chat Models ===
    "vicuna-v15-7b": {
        "llm_family": "llama2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "lmsys/vicuna-7b-v1.5"
    },

    "vicuna-v15-13b": {
        "llm_family": "llama2", "llm_cls": LlamaForCausalLM, "hf_hub_path": "lmsys/vicuna-13b-v1.5"
    },
}
# fmt: on


class LLaMa2LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = _USE_FLASH_ATTN,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **LLAMA2_MODELS[llm_backbone_id],
        )

        # [Special Case] LLaMa-2 PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        if self.identifier.startswith("llama2-") and self.identifier.endswith("-pure"):
            return PurePromptBuilder

        elif self.identifier.startswith("llama2-") and self.identifier.endswith("-chat"):
            return LLaMa2ChatPromptBuilder

        elif self.identifier.startswith("vicuna"):
            return VicunaV15ChatPromptBuilder

        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return LlamaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """LLaMa-2 was trained in BF16; see https://huggingface.co/docs/transformers/main/model_doc/llama2."""
        return torch.bfloat16

    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        return (self.llm.model.embed_tokens, self.llm.model.layers[-1], self.llm.lm_head)
