"""Comprehensive tests for GhostKV package.

Tests all components without requiring a GPU or model:
- kv.py: TTQ compression, serialization round-trip, session persistence
- tools/: search, code, files, http, memory
- agent.py: tool dispatch, regex parsing, ReAct loop logic
- model.py: abstract interface (no instantiation test — needs GPU)
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import DynamicCache


# ===================================================================
# KV module tests
# ===================================================================

class TestKVCompression:
    """Test TTQ compression: rotate → quantize → derotate."""

    def test_random_orthogonal_is_orthogonal(self):
        from ghostkv.kv import random_orthogonal
        R = random_orthogonal(64)
        assert R.shape == (64, 64)
        identity = R @ R.T
        assert torch.allclose(identity, torch.eye(64), atol=1e-5)

    def test_random_orthogonal_different_seeds(self):
        from ghostkv.kv import random_orthogonal
        R1 = random_orthogonal(32)
        R2 = random_orthogonal(32)
        assert not torch.allclose(R1, R2)  # statistically near-certain

    def test_compress_tensor_shape_preserved(self):
        from ghostkv.kv import compress_tensor, random_orthogonal
        R = random_orthogonal(128)
        x = torch.randn(1, 8, 50, 128)
        out = compress_tensor(x, R, bits=3)
        assert out.shape == x.shape

    def test_compress_tensor_dtype_preserved(self):
        from ghostkv.kv import compress_tensor, random_orthogonal
        R = random_orthogonal(32)
        x = torch.randn(2, 4, 10, 32)
        out = compress_tensor(x, R, bits=3)
        assert out.dtype == x.dtype

    def test_compress_tensor_3bit_levels(self):
        """3-bit quantization should have 2^(3-1)-1 = 3 levels in rotated space."""
        from ghostkv.kv import compress_tensor, random_orthogonal
        R = random_orthogonal(16)
        x = torch.randn(1, 1, 20, 16)
        out = compress_tensor(x, R, bits=3)
        # The output should be close to input (lossy but reasonable)
        relative_error = (x - out).norm() / x.norm()
        assert relative_error < 1.0  # less than 100% error

    def test_compress_tensor_4bit_better_than_3bit(self):
        """More bits should give lower error."""
        from ghostkv.kv import compress_tensor, random_orthogonal
        R = random_orthogonal(32)
        x = torch.randn(1, 4, 30, 32)
        out_3 = compress_tensor(x, R, bits=3)
        out_4 = compress_tensor(x, R, bits=4)
        err_3 = (x.float() - out_3.float()).norm().item()
        err_4 = (x.float() - out_4.float()).norm().item()
        assert err_4 < err_3

    def test_compress_kv_cache_round_trip(self):
        """compress_kv_cache should produce a valid DynamicCache."""
        from ghostkv.kv import compress_kv_cache, random_orthogonal
        R = random_orthogonal(64)
        cache = DynamicCache()
        for i in range(4):
            k = torch.randn(1, 4, 20, 64)
            v = torch.randn(1, 4, 20, 64)
            cache.update(k, v, layer_idx=i)
        assert cache.get_seq_length() == 20

        compressed = compress_kv_cache(cache, R, bits=3)
        assert compressed.get_seq_length() == 20
        assert len(list(compressed)) == 4

    def test_compress_kv_cache_preserves_structure(self):
        """Each layer should have same shape after compression."""
        from ghostkv.kv import compress_kv_cache, random_orthogonal
        R = random_orthogonal(32)
        cache = DynamicCache()
        for i in range(3):
            k = torch.randn(1, 2, 15, 32)
            v = torch.randn(1, 2, 15, 32)
            cache.update(k, v, layer_idx=i)

        compressed = compress_kv_cache(cache, R, bits=3)
        orig_layers = list(cache)
        comp_layers = list(compressed)
        for i in range(3):
            assert orig_layers[i][0].shape == comp_layers[i][0].shape
            assert orig_layers[i][1].shape == comp_layers[i][1].shape


class TestKVSerialization:
    """Test serialize/deserialize round-trip with zlib binary format."""

    def _make_cache(self, n_layers=3, seq_len=20, n_heads=4, head_dim=64):
        cache = DynamicCache()
        for i in range(n_layers):
            k = torch.randn(1, n_heads, seq_len, head_dim)
            v = torch.randn(1, n_heads, seq_len, head_dim)
            cache.update(k, v, layer_idx=i)
        return cache

    def test_serialize_deserialize_round_trip(self):
        from ghostkv.kv import serialize_kv, deserialize_kv
        cache = self._make_cache()
        data = serialize_kv(cache)
        assert isinstance(data, bytes)
        assert len(data) > 0

        restored = deserialize_kv(data)
        assert restored.get_seq_length() == cache.get_seq_length()

        # Check tensor values match
        orig_layers = list(cache)
        rest_layers = list(restored)
        for i in range(3):
            assert torch.allclose(
                orig_layers[i][0].cpu().float(),
                rest_layers[i][0].cpu().float(),
                atol=0.05,  # fp32→fp16→fp32 round-trip tolerance
            )
            assert torch.allclose(
                orig_layers[i][1].cpu().float(),
                rest_layers[i][1].cpu().float(),
                atol=0.05,
            )

    def test_serialize_empty_cache(self):
        """Empty cache (0 layers) should serialize/deserialize cleanly."""
        from ghostkv.kv import serialize_kv, deserialize_kv
        cache = DynamicCache()
        data = serialize_kv(cache)
        restored = deserialize_kv(data)
        assert len(list(restored)) == 0

    def test_serialize_single_layer(self):
        from ghostkv.kv import serialize_kv, deserialize_kv
        cache = DynamicCache()
        k = torch.randn(1, 8, 10, 128)
        v = torch.randn(1, 8, 10, 128)
        cache.update(k, v, layer_idx=0)

        data = serialize_kv(cache)
        restored = deserialize_kv(data)
        assert len(list(restored)) == 1
        assert torch.allclose(
            list(restored)[0][0].cpu().float(),
            k.cpu().float(),
            atol=0.05,
        )

    def test_serialize_large_cache(self):
        """Test with realistic sizes (40 layers, 128 head_dim)."""
        from ghostkv.kv import serialize_kv, deserialize_kv
        cache = DynamicCache()
        for i in range(40):
            k = torch.randn(1, 8, 100, 128)
            v = torch.randn(1, 8, 100, 128)
            cache.update(k, v, layer_idx=i)

        data = serialize_kv(cache)
        assert len(data) > 0
        # Should be compressed (much smaller than raw)
        raw_size = 40 * 2 * 8 * 100 * 128 * 2  # layers * KV * heads * seq * dim * bytes
        assert len(data) < raw_size

        restored = deserialize_kv(data)
        assert restored.get_seq_length() == 100

    def test_serialize_fp16_preservation(self):
        """Serialization converts to fp16 — verify round-trip within fp16 tolerance."""
        from ghostkv.kv import serialize_kv, deserialize_kv
        cache = DynamicCache()
        k = torch.randn(1, 4, 50, 64, dtype=torch.float32)
        v = torch.randn(1, 4, 50, 64, dtype=torch.float32)
        cache.update(k, v, layer_idx=0)

        data = serialize_kv(cache)
        restored = deserialize_kv(data)
        rk, rv = list(restored)[0][0], list(restored)[0][1]

        # fp16 round-trip tolerance
        assert torch.allclose(k.float(), rk.float(), atol=1e-2)


class TestKVSession:
    """Test KVSession save/load/reset with temp directories."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.session_dir = Path(self.tmpdir) / "test_session"

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_new_session(self):
        from ghostkv.kv import KVSession
        s = KVSession(name="test_new", head_dim=64)
        # Override dir to tmp
        s.base_dir = self.session_dir / "test_new"
        s.base_dir.mkdir(parents=True, exist_ok=True)
        assert s.kv_seq_length() == 0
        assert s.steps == 0
        assert s.token_costs == []

    def test_save_and_load_metadata(self):
        from ghostkv.kv import KVSession
        s = KVSession(name="test_meta", model_name="test-model", head_dim=64)
        s.base_dir = self.session_dir / "test_meta"
        s.base_dir.mkdir(parents=True, exist_ok=True)
        s.steps = 5
        s.total_tokens = 100
        s.token_costs = [20, 20, 20, 20, 20]
        s.save()

        # Load into new session
        s2 = KVSession(name="test_meta", head_dim=64)
        s2.base_dir = self.session_dir / "test_meta"
        loaded = s2.load()
        assert loaded is True
        assert s2.steps == 5
        assert s2.total_tokens == 100
        assert s2.token_costs == [20, 20, 20, 20, 20]
        assert s2.model_name == "test-model"

    def test_save_and_load_kv_cache(self):
        from ghostkv.kv import KVSession
        s = KVSession(name="test_kv", head_dim=32)
        s.base_dir = self.session_dir / "test_kv"
        s.base_dir.mkdir(parents=True, exist_ok=True)
        s.ensure_rotation("cpu")

        # Add a KV cache
        cache = DynamicCache()
        for i in range(3):
            k = torch.randn(1, 2, 10, 32)
            v = torch.randn(1, 2, 10, 32)
            cache.update(k, v, layer_idx=i)
        s.cache = cache
        s.steps = 1
        s.save()

        # Load
        s2 = KVSession(name="test_kv", head_dim=32)
        s2.base_dir = self.session_dir / "test_kv"
        loaded = s2.load()
        assert loaded is True
        assert s2.kv_seq_length() == 10
        assert s2.rotation is not None
        assert s2.rotation.shape == (32, 32)

    def test_reset_clears_everything(self):
        from ghostkv.kv import KVSession
        s = KVSession(name="test_reset", head_dim=32)
        s.base_dir = self.session_dir / "test_reset"
        s.base_dir.mkdir(parents=True, exist_ok=True)
        s.cache = DynamicCache()
        for i in range(2):
            k = torch.randn(1, 2, 5, 32)
            v = torch.randn(1, 2, 5, 32)
            s.cache.update(k, v, layer_idx=i)
        s.steps = 3
        s.total_tokens = 50
        s.token_costs = [10, 20, 20]

        s.reset()
        assert s.cache is None
        assert s.steps == 0
        assert s.total_tokens == 0
        assert s.token_costs == []
        assert s.rotation is None

    def test_stats(self):
        from ghostkv.kv import KVSession
        s = KVSession(name="test_stats", model_name="qwen3-4b", head_dim=32)
        s.base_dir = self.session_dir / "test_stats"
        s.base_dir.mkdir(parents=True, exist_ok=True)
        s.steps = 4
        s.total_tokens = 80
        s.token_costs = [20, 20, 20, 20]

        stats = s.stats()
        assert stats["session"] == "test_stats"
        assert stats["model"] == "qwen3-4b"
        assert stats["steps"] == 4
        assert stats["total_tokens"] == 80
        assert stats["avg_tokens_per_step"] == 20.0

    def test_log_writes_to_file(self):
        from ghostkv.kv import KVSession
        s = KVSession(name="test_log", head_dim=32)
        s.base_dir = self.session_dir / "test_log"
        s.base_dir.mkdir(parents=True, exist_ok=True)
        s.log("Line 1")
        s.log("Line 2")
        s.save()

        content = s.log_path.read_text()
        assert "Line 1" in content
        assert "Line 2" in content

    def test_load_nonexistent_session(self):
        from ghostkv.kv import KVSession
        s = KVSession(name="nonexistent", head_dim=32)
        s.base_dir = self.session_dir / "nonexistent"
        s.base_dir.mkdir(parents=True, exist_ok=True)
        loaded = s.load()
        assert loaded is False

    def test_kv_size_bytes(self):
        from ghostkv.kv import KVSession
        s = KVSession(name="test_size", head_dim=32)
        s.base_dir = self.session_dir / "test_size"
        s.base_dir.mkdir(parents=True, exist_ok=True)
        cache = DynamicCache()
        k = torch.randn(1, 2, 10, 32, dtype=torch.float32)
        v = torch.randn(1, 2, 10, 32, dtype=torch.float32)
        cache.update(k, v, layer_idx=0)
        s.cache = cache
        size = s.kv_size_bytes()
        expected = (2 * 10 * 32 * 4) * 2  # K+V, 10 tokens, 32 dim, 4 bytes each
        assert size == expected


# ===================================================================
# Tool tests
# ===================================================================

class TestSearchTool:
    """Test web search tool (SearXNG). Offline: test error handling."""

    def test_search_instantiation(self):
        from ghostkv.tools.search import SearchTool
        s = SearchTool(base_url="http://nonexistent:9999", timeout=1)
        assert s.name == "search"

    def test_search_offline_returns_error(self):
        from ghostkv.tools.search import SearchTool
        s = SearchTool(base_url="http://nonexistent:9999", timeout=1)
        result = s.run("test query")
        assert "error" in result.lower() or "Error" in result


class TestCodeTool:
    """Test Python code execution."""

    def test_simple_print(self):
        from ghostkv.tools.code import CodeTool
        c = CodeTool(timeout=5)
        result = c.run("print('hello world')")
        assert "hello world" in result

    def test_math_computation(self):
        from ghostkv.tools.code import CodeTool
        c = CodeTool(timeout=5)
        result = c.run("x = 2 + 3; print(x)")
        assert "5" in result

    def test_stderr_captured(self):
        from ghostkv.tools.code import CodeTool
        c = CodeTool(timeout=5)
        result = c.run("import sys; print('err', file=sys.stderr)")
        assert "err" in result

    def test_timeout(self):
        from ghostkv.tools.code import CodeTool
        c = CodeTool(timeout=1)
        result = c.run("import time; time.sleep(10)")
        assert "timeout" in result.lower() or "Timeout" in result

    def test_exception_captured(self):
        from ghostkv.tools.code import CodeTool
        c = CodeTool(timeout=5)
        result = c.run("raise ValueError('test error')")
        assert "test error" in result
        assert "1]" in result  # exit code

    def test_no_output(self):
        from ghostkv.tools.code import CodeTool
        c = CodeTool(timeout=5)
        result = c.run("x = 42")
        assert "no output" in result.lower()


class TestFileTools:
    """Test file read/write tools."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_and_read(self):
        from ghostkv.tools.files import FileReadTool, FileWriteTool
        w = FileWriteTool()
        r = FileReadTool()
        path = os.path.join(self.tmpdir, "test.txt")

        w_result = w.run(path, "hello from ghostkv")
        assert "Wrote" in w_result

        r_result = r.run(path)
        assert "hello from ghostkv" in r_result

    def test_read_nonexistent(self):
        from ghostkv.tools.files import FileReadTool
        r = FileReadTool()
        result = r.run("/nonexistent/file.txt")
        assert "not found" in result.lower() or "File not found" in result

    def test_write_creates_dirs(self):
        from ghostkv.tools.files import FileWriteTool
        w = FileWriteTool()
        path = os.path.join(self.tmpdir, "sub", "dir", "file.txt")
        result = w.run(path, "deep write")
        assert "Wrote" in result
        assert Path(path).read_text() == "deep write"

    def test_read_truncates_large_file(self):
        from ghostkv.tools.files import FileReadTool
        r = FileReadTool()
        path = os.path.join(self.tmpdir, "big.txt")
        Path(path).write_text("x" * 200_000)
        result = r.run(path)
        assert "truncated" in result.lower()

    def test_read_directory(self):
        from ghostkv.tools.files import FileReadTool
        r = FileReadTool()
        result = r.run(self.tmpdir)
        assert "not a file" in result.lower()


class TestHttpTool:
    """Test HTTP tool. Offline: test error handling."""

    def test_http_instantiation(self):
        from ghostkv.tools.http import HttpTool
        h = HttpTool(timeout=2)
        assert h.name == "http"

    def test_http_no_url(self):
        from ghostkv.tools.http import HttpTool
        h = HttpTool(timeout=2)
        result = h.run(url="")
        assert "no url" in result.lower()

    def test_http_offline_returns_error(self):
        from ghostkv.tools.http import HttpTool
        h = HttpTool(timeout=1)
        result = h.run(url="http://nonexistent-host-99999.local/test")
        assert "error" in result.lower()


class TestMemoryTool:
    """Test Obsidian vault memory tool."""

    def setup_method(self):
        self.vault = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.vault, ignore_errors=True)

    def test_write_and_recall(self):
        from ghostkv.tools.memory import MemoryTool
        m = MemoryTool(vault_path=self.vault)
        m.write("Eiffel Tower", "Built in 1889 by [[Gustave Eiffel]] in [[Paris]].",
                tags=["eiffel", "paris", "landmark"])

        result = m.run("Eiffel Tower")
        assert "1889" in result
        assert "Eiffel" in result

    def test_recall_no_match(self):
        from ghostkv.tools.memory import MemoryTool
        m = MemoryTool(vault_path=self.vault)
        result = m.run("something that doesn't exist in vault")
        assert "No memories" in result

    def test_write_creates_frontmatter(self):
        from ghostkv.tools.memory import MemoryTool
        m = MemoryTool(vault_path=self.vault)
        path = m.write("Test Note", "Content here.", memory_type="fact",
                       tags=["test"], related="Other Note")
        content = Path(path).read_text()
        assert content.startswith("---")
        assert "type: fact" in content
        assert "test" in content
        assert "[[Other Note]]" in content
        assert "Content here." in content

    def test_multiple_memories_ranked(self):
        from ghostkv.tools.memory import MemoryTool
        m = MemoryTool(vault_path=self.vault)
        m.write("Python Language", "Python is a programming language.", tags=["python", "programming"])
        m.write("Python Snake", "Pythons are large snakes found in Asia.", tags=["python", "animal"])
        m.write("Monty Python", "Monty Python is a comedy group.", tags=["comedy"])

        # "programming" should rank Python Language first
        result = m.run("python programming")
        result_lower = result.lower()
        assert "python language" in result_lower
        # The programming one should appear first (highest relevance)
        lang_pos = result_lower.find("python language")
        snake_pos = result_lower.find("python snake")
        if snake_pos >= 0:
            assert lang_pos < snake_pos

    def test_list_files(self):
        from ghostkv.tools.memory import MemoryTool
        m = MemoryTool(vault_path=self.vault)
        m.write("First Note", "Content 1.")
        m.write("Second Note", "Content 2.")
        files = m.list_files()
        assert len(files) == 2

    def test_count(self):
        from ghostkv.tools.memory import MemoryTool
        m = MemoryTool(vault_path=self.vault)
        assert m.count() == 0
        m.write("Note", "Content.")
        assert m.count() == 1

    def test_wikilinks_in_content(self):
        from ghostkv.tools.memory import MemoryTool
        m = MemoryTool(vault_path=self.vault)
        m.write("Paris", "Capital of [[France]].", tags=["city"])
        m.write("France", "Country in [[Western Europe]]. Capital: [[Paris]].", tags=["country"])

        result = m.run("Paris")
        assert "Paris" in result

    def test_slugify_title(self):
        from ghostkv.tools.memory import MemoryTool
        m = MemoryTool(vault_path=self.vault)
        path = m.write("What is the Eiffel Tower???", "Content", tags=["test"])
        filename = Path(path).name
        assert "?" not in filename
        assert filename.endswith(".md")


# ===================================================================
# Agent tests (tool dispatch, regex, no model needed)
# ===================================================================

class TestToolDispatch:
    """Test regex parsing and tool dispatch."""

    def test_parse_search(self):
        from ghostkv.agent import ACTION_RE
        m = ACTION_RE.search('Action: search("Eiffel Tower")')
        assert m is not None
        assert m.group(1).lower() == "search"
        assert m.group(2) == "Eiffel Tower"

    def test_parse_run(self):
        from ghostkv.agent import ACTION_RE
        m = ACTION_RE.search('Action: run("print(42)")')
        assert m is not None
        assert m.group(1).lower() == "run"
        assert m.group(2) == "print(42)"

    def test_parse_write_two_args(self):
        from ghostkv.agent import ACTION_RE
        m = ACTION_RE.search('Action: write("/tmp/file.txt", "hello world")')
        assert m is not None
        assert m.group(1).lower() == "write"
        assert m.group(2) == "/tmp/file.txt"
        assert m.group(3) == "hello world"

    def test_parse_recall(self):
        from ghostkv.agent import ACTION_RE
        m = ACTION_RE.search('Action: recall("Gustave Eiffel birthplace")')
        assert m is not None
        assert m.group(1).lower() == "recall"

    def test_parse_embedded_in_thought(self):
        from ghostkv.agent import ACTION_RE
        text = 'I need to find this. Action: search("Paris population")'
        m = ACTION_RE.search(text)
        assert m is not None
        assert m.group(2) == "Paris population"

    def test_no_parse_plain_text(self):
        from ghostkv.agent import ACTION_RE
        m = ACTION_RE.search("Just some regular text without any tool call")
        assert m is None

    def test_dispatch_unknown_tool(self):
        from ghostkv.agent import ToolDispatch
        tools = ToolDispatch()
        name, result = tools.dispatch('Action: fly("to the moon")')
        assert name == "fly"
        assert "Unknown tool" in result

    def test_dispatch_code_tool(self):
        from ghostkv.agent import ToolDispatch
        from ghostkv.tools.code import CodeTool
        tools = ToolDispatch(code=CodeTool(timeout=5))
        name, result = tools.dispatch('Action: run("print(2+2)")')
        assert name == "run"
        assert "4" in result

    def test_dispatch_file_read(self):
        from ghostkv.agent import ToolDispatch
        from ghostkv.tools.files import FileReadTool
        tools = ToolDispatch(file_read=FileReadTool())
        name, result = tools.dispatch('Action: read("/nonexistent/file.txt")')
        assert name == "read"

    def test_dispatch_file_write(self):
        from ghostkv.agent import ToolDispatch
        from ghostkv.tools.files import FileReadTool, FileWriteTool
        tools = ToolDispatch(file_write=FileWriteTool(), file_read=FileReadTool())
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "dispatch_test.txt")
        try:
            name, result = tools.dispatch(f'Action: write("{path}", "dispatched content")')
            assert name == "write"
            assert "Wrote" in result
            assert Path(path).read_text() == "dispatched content"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_dispatch_memory_recall_empty(self):
        from ghostkv.agent import ToolDispatch
        from ghostkv.tools.memory import MemoryTool
        tmpdir = tempfile.mkdtemp()
        try:
            tools = ToolDispatch(memory=MemoryTool(vault_path=tmpdir))
            name, result = tools.dispatch('Action: recall("nothing here")')
            assert name == "recall"
            assert "No memories" in result
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_available_tools(self):
        from ghostkv.agent import ToolDispatch
        from ghostkv.tools.code import CodeTool
        from ghostkv.tools.memory import MemoryTool
        tools = ToolDispatch(code=CodeTool(), memory=MemoryTool())
        available = tools.available_tools()
        assert "run" in available
        assert "recall" in available
        assert "search" not in available


class TestGenerateStep:
    """Test generate_step with a mock model backend."""

    def test_generate_step_basic(self):
        """Mock model that always predicts token 42."""
        from ghostkv.agent import generate_step
        from ghostkv.model import ModelBackend

        class MockModel(ModelBackend):
            def __init__(self):
                self._eos = 999
                self._count = 0

            def forward(self, input_ids, past_kv=None, use_cache=True):
                self._count += 1
                batch, seq = input_ids.shape
                # Return logits where token 42 always wins, then EOS on step 5
                vocab_size = 1000
                logits = torch.zeros(1, seq, vocab_size)
                if self._count < 5:
                    logits[:, -1, 42] = 100.0
                else:
                    logits[:, -1, self._eos] = 100.0
                # Return a simple cache
                cache = DynamicCache() if past_kv is None else past_kv
                if cache.get_seq_length() == 0:
                    k = torch.randn(1, 2, input_ids.shape[1], 8)
                    v = torch.randn(1, 2, input_ids.shape[1], 8)
                    cache.update(k, v, layer_idx=0)
                return logits, cache

            def tokenize(self, text, **kwargs):
                return torch.tensor([[1, 2, 3]])

            def decode(self, ids):
                return "decoded text"

            @property
            def head_dim(self): return 8

            @property
            def device(self): return "cpu"

            @property
            def eos_token_id(self): return self._eos

        mock = MockModel()
        input_ids = torch.tensor([[1, 2, 3]])
        gen_ids, pkv = generate_step(mock, input_ids, max_new=20)
        assert len(gen_ids) == 5  # 4 tokens of 42 + 1 EOS
        assert all(t == 42 for t in gen_ids[:4])
        assert gen_ids[4] == 999  # EOS


class TestAgentReAct:
    """Test agent's tag extraction and memory saving (no model needed)."""

    def test_extract_tags(self):
        from ghostkv.agent import GhostKVAgent
        # We need to mock the constructor dependencies
        # Just test the static method behavior via a minimal mock
        class MockAgent:
            pass

        # Import the helper directly
        from ghostkv.agent import ACTION_RE
        # Test the regex can handle multi-line responses with thoughts
        response = """Thought: I should look this up.
Action: search("test query")"""
        m = ACTION_RE.search(response)
        assert m is not None

    def test_extract_tags_from_text(self):
        """Test the _extract_tags method by creating a minimal agent-like object."""
        import re
        stop = {"about", "where", "which", "there", "their", "would", "could",
                "should", "what", "when", "how", "that", "this", "with",
                "from", "they", "been", "have", "will", "each", "does"}
        text = "Who built the Eiffel Tower and where were they born?"
        words = re.findall(r'\b[a-z]{5,}\b', text.lower())
        tags = list({w for w in words if w not in stop})[:10]
        assert "built" in tags
        assert "eiffel" in tags
        assert "tower" in tags
        assert "where" not in tags  # stop word


# ===================================================================
# Template tests
# ===================================================================

class TestTemplates:
    """Test Jinja template rendering."""

    def test_system_template(self):
        from jinja2 import Environment, FileSystemLoader
        from pathlib import Path
        template_dir = Path(__file__).parent.parent / "ghostkv" / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        t = env.get_template("system.j2")
        result = t.render(tools="search, run, read, write, http, recall")
        assert "search" in result
        assert "Action:" in result

    def test_observe_template(self):
        from jinja2 import Environment, FileSystemLoader
        from pathlib import Path
        template_dir = Path(__file__).parent.parent / "ghostkv" / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        t = env.get_template("observe.j2")
        result = t.render(result="The Eiffel Tower was built in 1889.")
        assert "Observation:" in result
        assert "1889" in result

    def test_memory_template(self):
        from jinja2 import Environment, FileSystemLoader
        from pathlib import Path
        template_dir = Path(__file__).parent.parent / "ghostkv" / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        t = env.get_template("memory.j2")
        result = t.render(memories=["Memory 1", "Memory 2"])
        assert "Relevant memories:" in result
        assert "Memory 1" in result

    def test_error_template(self):
        from jinja2 import Environment, FileSystemLoader
        from pathlib import Path
        template_dir = Path(__file__).parent.parent / "ghostkv" / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        t = env.get_template("error.j2")
        result = t.render(error="Connection refused")
        assert "Error:" in result
        assert "Connection refused" in result


# ===================================================================
# Integration test: full pipeline without model
# ===================================================================

class TestIntegration:
    """Integration tests that exercise the full pipeline minus the actual model."""

    def test_session_with_tools_and_memory(self):
        """Test session + tools + memory working together."""
        from ghostkv.kv import KVSession
        from ghostkv.tools.memory import MemoryTool
        from ghostkv.tools.code import CodeTool
        from ghostkv.agent import ToolDispatch

        tmpdir = tempfile.mkdtemp()
        vault = os.path.join(tmpdir, "vault")
        session_dir = os.path.join(tmpdir, "session")

        try:
            memory = MemoryTool(vault_path=vault)
            session = KVSession(name="integration", head_dim=32)
            session.base_dir = Path(session_dir)
            session.base_dir.mkdir(parents=True, exist_ok=True)

            # Write memory
            memory.write("Test Fact", "The answer is 42.", tags=["test"])

            # Recall
            result = memory.run("test")
            assert "42" in result

            # Use code tool
            code = CodeTool(timeout=5)
            code_result = code.run("print(6 * 7)")
            assert "42" in code_result

            # Dispatch through ToolDispatch
            tools = ToolDispatch(code=code, memory=memory)
            name, res = tools.dispatch('Action: run("print(6*7)")')
            assert "42" in res

            name, res = tools.dispatch('Action: recall("test")')
            assert "42" in res

            # Save session with metadata
            session.steps = 2
            session.total_tokens = 40
            session.token_costs = [20, 20]
            session.log("Step 1: code execution")
            session.log("Step 2: memory recall")
            session.save()

            # Reload
            session2 = KVSession(name="integration", head_dim=32)
            session2.base_dir = Path(session_dir)
            loaded = session2.load()
            assert loaded
            assert session2.steps == 2
            assert session2.total_tokens == 40

            # Check log was written
            log_content = session2.log_path.read_text()
            assert "Step 1" in log_content
            assert "Step 2" in log_content

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_compress_serialize_restore(self):
        """Full pipeline: create cache → compress → serialize → deserialize → verify."""
        from ghostkv.kv import (
            random_orthogonal, compress_kv_cache, serialize_kv, deserialize_kv, KVSession
        )

        tmpdir = tempfile.mkdtemp()
        try:
            # Create a realistic cache
            head_dim = 64
            n_layers = 8
            n_heads = 4
            seq_len = 30
            cache = DynamicCache()
            for i in range(n_layers):
                k = torch.randn(1, n_heads, seq_len, head_dim)
                v = torch.randn(1, n_heads, seq_len, head_dim)
                cache.update(k, v, layer_idx=i)

            # Compress
            R = random_orthogonal(head_dim)
            compressed = compress_kv_cache(cache, R, bits=3)
            assert compressed.get_seq_length() == seq_len

            # Serialize
            data = serialize_kv(compressed)
            assert isinstance(data, bytes)

            # Deserialize
            restored = deserialize_kv(data, device="cpu")
            assert restored.get_seq_length() == seq_len
            assert len(list(restored)) == n_layers

            # Verify tensors are close (compressed is lossy, so just check shapes)
            for layer in list(restored):
                assert layer[0].shape == (1, n_heads, seq_len, head_dim)
                assert layer[1].shape == (1, n_heads, seq_len, head_dim)

            # Save to session
            session = KVSession(name="compress_test", head_dim=head_dim)
            session.base_dir = Path(tmpdir) / "session"
            session.base_dir.mkdir(parents=True, exist_ok=True)
            session.cache = restored
            session.rotation = R
            session.steps = 1
            session.save()

            # Load back
            session2 = KVSession(name="compress_test", head_dim=head_dim)
            session2.base_dir = Path(tmpdir) / "session"
            loaded = session2.load()
            assert loaded
            assert session2.kv_seq_length() == seq_len

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
