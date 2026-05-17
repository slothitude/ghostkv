"""Model backend abstraction for GhostKV.

Provides a unified interface for loading and running language models.
Currently supports TransformersBackend (HF models with BitsAndBytes).

The key method is forward() which returns (logits, new_kv) — compatible
with DynamicCache for KV-state persistence across steps.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache

logger = logging.getLogger(__name__)


class ModelBackend(ABC):
    """Abstract base class for model backends.

    All backends expose the same interface so the agent loop is model-agnostic.
    """

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        past_kv: Optional[DynamicCache] = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, Optional[DynamicCache]]:
        """Run a single forward pass.

        Args:
            input_ids: (batch, seq_len) token IDs
            past_kv: Previous KV cache for incremental decoding
            use_cache: Whether to return KV cache

        Returns:
            (logits, new_kv) — logits shape (batch, seq_len, vocab_size)
        """
        ...

    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> torch.Tensor:
        """Tokenize text, return input_ids tensor on device."""
        ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        ...

    @property
    @abstractmethod
    def head_dim(self) -> int:
        """Dimension per attention head."""
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        """Device string (e.g. 'cuda', 'cpu')."""
        ...

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """End-of-sequence token ID."""
        ...


class TransformersBackend(ModelBackend):
    """Backend using HuggingFace transformers with optional 4-bit quantization.

    Uses model.forward() directly (not model.generate()) for full KV cache
    control — compatible with DynamicCache serialization.

    Args:
        model_path: Path or HF hub ID for the model
        quantize_4bit: Load with BitsAndBytes NF4 quantization
        device_map: Device mapping strategy (default 'auto')
    """

    def __init__(
        self,
        model_path: str,
        quantize_4bit: bool = True,
        device_map: str = "auto",
    ):
        logger.info(f"Loading model from {model_path}...")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict = {
            "local_files_only": True,
            "torch_dtype": torch.float16,
            "device_map": device_map,
        }
        if quantize_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()

        config = self.model.config
        self._head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self._eos_id = self.tokenizer.eos_token_id

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded: {n_params:,} params, head_dim={self._head_dim}")

        if self._device == "cuda":
            free = torch.cuda.mem_get_info()[0] / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"VRAM: {free:.1f}GB free / {total:.1f}GB total")

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kv: Optional[DynamicCache] = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, Optional[DynamicCache]]:
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                past_key_values=past_kv,
                use_cache=use_cache,
            )
        return out.logits, out.past_key_values

    def tokenize(self, text: str, **kwargs) -> torch.Tensor:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=kwargs.get("max_length", 512),
        )
        return enc["input_ids"].to(self._device)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def device(self) -> str:
        return self._device

    @property
    def eos_token_id(self) -> int:
        return self._eos_id
