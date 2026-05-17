"""KV cache management — DynamicCache, TTQ compression, serialization.

Reuses proven patterns from Stage 7c/7d:
- serialize_kv/deserialize_kv: zlib binary format with struct headers
- compress_kv: rotate → quantize → derotate (inlined TTQ)
- DynamicCache API: list(cache) → (K, V, extra), cache.update(k, v, layer_idx)
"""

from __future__ import annotations

import io
import logging
import os
import struct
import zlib
from pathlib import Path
from typing import Optional

import torch
from transformers import DynamicCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TTQ compression (inlined from Stage 7a/7d — no resolution_router dependency)
# ---------------------------------------------------------------------------

def random_orthogonal(dim: int, device: str = "cpu") -> torch.Tensor:
    """Generate a random orthogonal matrix via QR decomposition."""
    Q, _ = torch.linalg.qr(torch.randn(dim, dim, device=device))
    return Q


def compress_tensor(
    x: torch.Tensor,
    rotation: torch.Tensor,
    bits: int = 3,
) -> torch.Tensor:
    """Compress a single tensor: rotate → quantize → derotate.

    Args:
        x: Input tensor with last dim == head_dim
        rotation: (head_dim, head_dim) orthogonal matrix
        bits: Quantization bits (default 3)
    """
    levels = 2 ** (bits - 1) - 1
    x_rot = x.float() @ rotation.T
    x_max = x_rot.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = x_max / levels
    quantized = torch.round(x_rot / scale).clamp(-levels, levels)
    deq = quantized * scale
    return (deq @ rotation).to(x.dtype)


def compress_kv_cache(
    cache: DynamicCache,
    rotation: torch.Tensor,
    bits: int = 3,
) -> DynamicCache:
    """Compress all layers in a DynamicCache using TTQ.

    Args:
        cache: Source DynamicCache
        rotation: (head_dim, head_dim) orthogonal matrix
        bits: Quantization bits

    Returns:
        New DynamicCache with compressed K/V tensors
    """
    layers = list(cache)
    new_cache = DynamicCache()
    for layer_idx, item in enumerate(layers):
        k, v = item[0], item[1]
        if k is not None and k.numel() > 0:
            k = compress_tensor(k, rotation, bits)
            v = compress_tensor(v, rotation, bits)
        new_cache.update(k, v, layer_idx=layer_idx)
    return new_cache


# ---------------------------------------------------------------------------
# Serialization (from Stage 7c — zlib binary format)
# ---------------------------------------------------------------------------

def serialize_kv(cache: DynamicCache) -> bytes:
    """Serialize a DynamicCache to compressed bytes.

    Binary format:
        Header (4 bytes): num_layers (uint16) + reserved (uint16)
        Per layer:
            K shape: 1 byte (ndim) + ndim*4 bytes (int32 per dim)
            V shape: 1 byte (ndim) + ndim*4 bytes (int32 per dim)
            K data: float16 tensor bytes
            V data: float16 tensor bytes
        All layer data zlib-compressed together.
    """
    import numpy as np

    buf = io.BytesIO()
    layers = list(cache)
    num_layers = len(layers)
    buf.write(struct.pack("<HH", num_layers, 0))

    for item in layers:
        k, v = item[0], item[1]
        k_np = k.detach().cpu().to(torch.float16).numpy()
        v_np = v.detach().cpu().to(torch.float16).numpy()

        buf.write(struct.pack("<B", len(k_np.shape)))
        for dim in k_np.shape:
            buf.write(struct.pack("<i", dim))
        buf.write(k_np.tobytes())

        buf.write(struct.pack("<B", len(v_np.shape)))
        for dim in v_np.shape:
            buf.write(struct.pack("<i", dim))
        buf.write(v_np.tobytes())

    raw_bytes = buf.getvalue()
    return zlib.compress(raw_bytes, level=1)


def deserialize_kv(data: bytes, device: str = "cpu") -> DynamicCache:
    """Reconstruct a DynamicCache from compressed bytes."""
    import numpy as np

    raw = zlib.decompress(data)
    buf = io.BytesIO(raw)

    num_layers, _ = struct.unpack("<HH", buf.read(4))
    cache = DynamicCache()

    for layer_idx in range(num_layers):
        k_ndim = struct.unpack("<B", buf.read(1))[0]
        k_shape = tuple(struct.unpack("<i", buf.read(4))[0] for _ in range(k_ndim))
        k_nbytes = int(np.prod(k_shape)) * 2
        k_np = np.frombuffer(buf.read(k_nbytes), dtype=np.float16).reshape(k_shape)
        k_tensor = torch.from_numpy(k_np.copy()).to(device)

        v_ndim = struct.unpack("<B", buf.read(1))[0]
        v_shape = tuple(struct.unpack("<i", buf.read(4))[0] for _ in range(v_ndim))
        v_nbytes = int(np.prod(v_shape)) * 2
        v_np = np.frombuffer(buf.read(v_nbytes), dtype=np.float16).reshape(v_shape)
        v_tensor = torch.from_numpy(v_np.copy()).to(device)

        cache.update(k_tensor, v_tensor, layer_idx=layer_idx)

    return cache


# ---------------------------------------------------------------------------
# Session persistence — KV state + metadata on disk
# ---------------------------------------------------------------------------

class KVSession:
    """Manages KV cache persistence for a session.

    Session layout:
        ~/.ghostkv/sessions/<name>/
            kv_cache.bin        # Serialized DynamicCache
            metadata.json       # {model, created, steps, tokens, head_dim}
            conversation.log    # Full text log
    """

    def __init__(
        self,
        name: str = "default",
        model_name: str = "unknown",
        head_dim: int = 128,
        kv_bits: int = 3,
    ):
        self.name = name
        self.model_name = model_name
        self.head_dim = head_dim
        self.kv_bits = kv_bits
        self.base_dir = Path.home() / ".ghostkv" / "sessions" / name
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Runtime state
        self.cache: Optional[DynamicCache] = None
        self.rotation: Optional[torch.Tensor] = None
        self.steps: int = 0
        self.total_tokens: int = 0
        self.token_costs: list[int] = []
        self._log_lines: list[str] = []

    @property
    def kv_cache_path(self) -> Path:
        return self.base_dir / "kv_cache.bin"

    @property
    def metadata_path(self) -> Path:
        return self.base_dir / "metadata.json"

    @property
    def log_path(self) -> Path:
        return self.base_dir / "conversation.log"

    def ensure_rotation(self, device: str = "cpu"):
        """Initialize rotation matrix if not set."""
        if self.rotation is None:
            self.rotation = random_orthogonal(self.head_dim, device)

    def save(self):
        """Serialize KV cache and metadata to disk."""
        import json

        meta = {
            "model": self.model_name,
            "head_dim": self.head_dim,
            "kv_bits": self.kv_bits,
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "token_costs": self.token_costs,
        }
        self.metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if self.cache is not None:
            data = serialize_kv(self.cache)
            self.kv_cache_path.write_bytes(data)
            logger.info(f"KV saved: {self.kv_seq_length()} tokens, {len(data)} bytes")

        # Append log
        if self._log_lines:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.writelines(self._log_lines)
            self._log_lines.clear()

    def load(self, device: str = "cpu") -> bool:
        """Load KV cache and metadata from disk.

        Returns True if session was loaded, False if empty/new.
        """
        import json

        if not self.metadata_path.exists():
            logger.info(f"New session: {self.name}")
            return False

        meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.model_name = meta.get("model", self.model_name)
        self.head_dim = meta.get("head_dim", self.head_dim)
        self.kv_bits = meta.get("kv_bits", self.kv_bits)
        self.steps = meta.get("steps", 0)
        self.total_tokens = meta.get("total_tokens", 0)
        self.token_costs = meta.get("token_costs", [])

        has_kv = False
        if self.kv_cache_path.exists():
            data = self.kv_cache_path.read_bytes()
            self.cache = deserialize_kv(data, device=device)
            self.ensure_rotation(device)
            has_kv = True
            logger.info(f"KV loaded: {self.kv_seq_length()} tokens")

        return True

    def reset(self):
        """Clear KV state, start fresh."""
        self.cache = None
        self.rotation = None
        self.steps = 0
        self.total_tokens = 0
        self.token_costs = []
        self._log_lines.clear()
        logger.info("Session reset.")

    def log(self, line: str):
        """Append a line to the conversation log."""
        self._log_lines.append(line + "\n")

    def kv_seq_length(self) -> int:
        """Current KV cache sequence length."""
        if self.cache is not None:
            return self.cache.get_seq_length()
        return 0

    def kv_size_bytes(self) -> int:
        """Estimated KV cache size in bytes."""
        if self.cache is None:
            return 0
        size = 0
        for item in list(self.cache):
            k, v = item[0], item[1]
            size += k.numel() * k.element_size()
            size += v.numel() * v.element_size()
        return size

    def stats(self) -> dict:
        """Return session statistics."""
        kv_bytes = self.kv_size_bytes()
        raw_bytes = self.kv_seq_length() * self.head_dim * 2 * 2 * 40  # rough
        compression = (1 - kv_bytes / max(raw_bytes, 1)) * 100 if raw_bytes > 0 else 0

        return {
            "session": self.name,
            "model": self.model_name,
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_step": self.total_tokens / max(1, self.steps),
            "kv_tokens": self.kv_seq_length(),
            "kv_bytes": kv_bytes,
            "kv_compressed_size": len(serialize_kv(self.cache)) if self.cache is not None else 0,
        }
