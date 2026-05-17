"""Model backend abstraction for GhostKV.

Provides a unified interface for loading and running language models.
Currently supports TransformersBackend (HF models with BitsAndBytes).

The key method is forward() which returns (logits, new_kv) — compatible
with DynamicCache for KV-state persistence across steps.

Symmetric TTQ: monkey-patches Q/K/V projection layers to apply rotate→quantize→derotate
inline during the forward pass. All three projections share the same rotation matrix,
so quantization noise becomes symmetric in the attention dot product Q·K^T and partially
cancels. Uses monkey-patching (not register_forward_hook) because hooks segfault on
BitsAndBytes Linear4bit layers.
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

    # ------------------------------------------------------------------
    # Symmetric TTQ — inline Q/K/V compression during forward pass
    # ------------------------------------------------------------------

    def _find_attn_layers(self) -> list[tuple[str, object]]:
        """Find all attention layers with separate q_proj/k_proj/v_proj."""
        layers = []
        for name, module in self.model.named_modules():
            cls_name = type(module).__name__.lower()
            if ("attention" in cls_name or "attn" in cls_name) and hasattr(module, "q_proj"):
                layers.append((name, module))
        return layers

    def install_symmetric_ttq(
        self,
        rotation: torch.Tensor,
        k_bits: int = 3,
        v_bits: int = 3,
        q_bits: int = 4,
    ) -> bool:
        """Install symmetric TTQ monkey patches on Q/K/V projections.

        All three projections (Q, K, V) get rotate→quantize→derotate using
        the same rotation matrix. Quantization noise becomes symmetric in
        Q·K^T and partially cancels.

        Args:
            rotation: (head_dim, head_dim) orthogonal matrix (from KVSession)
            k_bits: Key quantization bits
            v_bits: Value quantization bits
            q_bits: Query quantization bits

        Returns:
            True if patches were installed, False if no attention layers found.
        """
        from ghostkv.kv import compress_tensor_multihead

        self._ttq_originals = []
        head_dim = self._head_dim
        attn_layers = self._find_attn_layers()

        if not attn_layers:
            logger.warning("No attention layers found for symmetric TTQ")
            return False

        rotation = rotation.to(self._device)

        for _name, module in attn_layers:
            for proj_name, bits in [("q_proj", q_bits), ("k_proj", k_bits), ("v_proj", v_bits)]:
                proj = getattr(module, proj_name)
                orig_fwd = proj.forward
                self._ttq_originals.append((module, proj_name, orig_fwd))

                def make_patched(original_forward, rot, hd, b):
                    def patched_forward(self_mod, *args, **kwargs):
                        out = original_forward(self_mod, *args, **kwargs)
                        return compress_tensor_multihead(out, rot, hd, b)
                    return patched_forward

                proj.forward = make_patched(orig_fwd, rotation, head_dim, bits)

        logger.info(f"Symmetric TTQ installed: {len(attn_layers)} layers, "
                     f"K={k_bits}b V={v_bits}b Q={q_bits}b")
        return True

    def remove_symmetric_ttq(self):
        """Remove symmetric TTQ patches, restore original forward methods."""
        if not hasattr(self, '_ttq_originals') or not self._ttq_originals:
            return
        for module, proj_name, orig_fwd in self._ttq_originals:
            getattr(module, proj_name).forward = orig_fwd
        self._ttq_originals.clear()
        logger.info("Symmetric TTQ removed")
