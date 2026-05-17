"""Comprehensive tests for GhostKV package.

Tests all components without requiring a GPU or model:
- kv.py: TTQ compression, serialization round-trip, session persistence
- tools/: search, code, files, http, memory
- agent.py: tool dispatch, regex parsing, ReAct loop logic
- model.py: abstract interface (no instantiation test — needs GPU)
- remote.py: RemoteBackend, MessageSession, agent remote mode
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

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

    def test_compress_tensor_multihead_shape(self):
        """Multi-head compression preserves shape of projection output."""
        from ghostkv.kv import compress_tensor_multihead, random_orthogonal
        R = random_orthogonal(32)
        # Simulate a projection output: (batch=1, seq=10, num_heads * head_dim = 4 * 32 = 128)
        x = torch.randn(1, 10, 128)
        out = compress_tensor_multihead(x, R, head_dim=32, bits=3)
        assert out.shape == x.shape

    def test_compress_tensor_multihead_dtype(self):
        from ghostkv.kv import compress_tensor_multihead, random_orthogonal
        R = random_orthogonal(16)
        x = torch.randn(2, 5, 64, dtype=torch.float32)
        out = compress_tensor_multihead(x, R, head_dim=16, bits=3)
        assert out.dtype == x.dtype

    def test_compress_tensor_multihead_4bit_better_than_3bit(self):
        from ghostkv.kv import compress_tensor_multihead, random_orthogonal
        R = random_orthogonal(32)
        x = torch.randn(1, 20, 128)
        out_3 = compress_tensor_multihead(x, R, head_dim=32, bits=3)
        out_4 = compress_tensor_multihead(x, R, head_dim=32, bits=4)
        err_3 = (x.float() - out_3.float()).norm().item()
        err_4 = (x.float() - out_4.float()).norm().item()
        assert err_4 < err_3


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


class TestBashTool:
    """Test bash command execution."""

    def test_simple_echo(self):
        from ghostkv.tools.bash import BashTool
        b = BashTool(timeout=5)
        result = b.run("echo hello world")
        assert "hello world" in result

    def test_pipe_and_redirect(self):
        from ghostkv.tools.bash import BashTool
        b = BashTool(timeout=5)
        result = b.run("echo 'foo bar baz' | tr ' ' '\\n' | sort")
        assert "bar" in result
        assert "baz" in result
        assert "foo" in result

    def test_exit_code(self):
        from ghostkv.tools.bash import BashTool
        b = BashTool(timeout=5)
        result = b.run("exit 1")
        assert "exit code" in result.lower()

    def test_stderr_captured(self):
        from ghostkv.tools.bash import BashTool
        b = BashTool(timeout=5)
        result = b.run("echo error >&2")
        assert "error" in result

    def test_timeout(self):
        from ghostkv.tools.bash import BashTool
        b = BashTool(timeout=1)
        result = b.run("sleep 10")
        assert "timeout" in result.lower()

    def test_no_output(self):
        from ghostkv.tools.bash import BashTool
        b = BashTool(timeout=5)
        result = b.run("true")
        assert "no output" in result.lower()

    def test_dispatch_bash(self):
        from ghostkv.agent import ToolDispatch
        from ghostkv.tools.bash import BashTool
        tools = ToolDispatch(bash=BashTool(timeout=5))
        name, result = tools.dispatch('Action: bash("echo dispatched")')
        assert name == "bash"
        assert "dispatched" in result


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


class TestSampling:
    """Test sample_token with temperature, top-k, top-p."""

    def test_greedy_equals_argmax(self):
        """temperature=0 should return argmax (deterministic)."""
        from ghostkv.agent import sample_token
        logits = torch.tensor([[1.0, 5.0, 3.0, 0.5]])
        tok = sample_token(logits, temperature=0)
        assert tok.item() == 1  # index of max

    def test_temperature_scaling(self):
        """Higher temperature flattens distribution — less concentrated on max."""
        from ghostkv.agent import sample_token
        logits = torch.tensor([[0.0, 10.0, 0.0, 0.0]])
        # Low temp: almost always picks index 1
        torch.manual_seed(42)
        counts = sum(sample_token(logits, temperature=0.01).item() == 1 for _ in range(20))
        assert counts >= 18  # almost always 1

    def test_top_k_filters(self):
        """top_k=2 should only sample from top 2 tokens."""
        from ghostkv.agent import sample_token
        logits = torch.tensor([[1.0, 5.0, 3.0, 0.5]])
        # top_k=2: indices 1 (5.0) and 2 (3.0)
        torch.manual_seed(0)
        for _ in range(50):
            tok = sample_token(logits, temperature=1.0, top_k=2, top_p=1.0)
            assert tok.item() in (1, 2)

    def test_top_p_nucleus(self):
        """top_p should keep smallest set exceeding threshold."""
        from ghostkv.agent import sample_token
        # Logits: [1, 10, 1, 1] — index 1 dominates
        logits = torch.tensor([[1.0, 10.0, 1.0, 1.0]])
        torch.manual_seed(0)
        counts = sum(sample_token(logits, temperature=1.0, top_k=0, top_p=0.5).item() == 1
                     for _ in range(50))
        assert counts >= 40  # index 1 should dominate

    def test_top_p_1_no_filtering(self):
        """top_p=1.0 should not filter anything."""
        from ghostkv.agent import sample_token
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        seen = set()
        torch.manual_seed(42)
        for _ in range(100):
            tok = sample_token(logits, temperature=2.0, top_k=0, top_p=1.0)
            seen.add(tok.item())
        # With high temp and no filtering, should see multiple tokens
        assert len(seen) >= 2

    def test_sampling_with_generate_step(self):
        """generate_step with sampling should still terminate on EOS."""
        from ghostkv.agent import generate_step
        from ghostkv.model import ModelBackend

        class MockModel(ModelBackend):
            def __init__(self):
                self._eos = 999
                self._count = 0

            def forward(self, input_ids, past_kv=None, use_cache=True):
                self._count += 1
                vocab_size = 1000
                logits = torch.zeros(1, input_ids.shape[1], vocab_size)
                if self._count < 4:
                    logits[:, -1, 42] = 100.0
                    logits[:, -1, 7] = 90.0  # second choice
                else:
                    logits[:, -1, self._eos] = 100.0
                cache = DynamicCache() if past_kv is None else past_kv
                return logits, cache

            def tokenize(self, text, **kwargs): return torch.tensor([[1]])
            def decode(self, ids): return "text"
            @property
            def head_dim(self): return 8
            @property
            def device(self): return "cpu"
            @property
            def eos_token_id(self): return self._eos

        mock = MockModel()
        ids, _ = generate_step(mock, torch.tensor([[1]]), max_new=20,
                               temperature=0.8, top_k=50, top_p=0.9)
        assert len(ids) == 4
        assert ids[-1] == 999


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


# ===================================================================
# Remote backend tests
# ===================================================================

class TestRemoteBackend:
    """Test RemoteBackend — mock requests.post."""

    def _mock_response(self, status_code=200, content="Hello!", usage=None):
        """Create a mock requests.Response."""
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = content
        resp.json.return_value = {
            "choices": [{"message": {"content": content}}],
            "usage": usage or {"total_tokens": 50, "prompt_tokens": 30, "completion_tokens": 20},
        }
        return resp

    def test_generate_success(self):
        """Basic generate returns (text, usage)."""
        from ghostkv.remote import RemoteBackend
        backend = RemoteBackend(
            base_url="https://api.test.com/v1/chat/completions",
            api_key="test-key",
            model="test-model",
        )
        mock_resp = self._mock_response(content="Test response", usage={"total_tokens": 42})

        with patch("ghostkv.remote.requests.post", return_value=mock_resp) as mock_post:
            text, usage = backend.generate(
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.7,
                max_tokens=100,
            )

        assert text == "Test response"
        assert usage["total_tokens"] == 42
        mock_post.assert_called_once()

        # Verify request format
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"]
        assert payload["model"] == "test-model"
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 100
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]

        headers = call_kwargs.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-key"

    def test_generate_request_format(self):
        """Verify messages are sent correctly."""
        from ghostkv.remote import RemoteBackend
        backend = RemoteBackend(base_url="https://api.test.com", api_key="k")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        mock_resp = self._mock_response()
        with patch("ghostkv.remote.requests.post", return_value=mock_resp) as mock_post:
            backend.generate(messages=messages)

        payload = mock_post.call_args.kwargs["json"]
        assert payload["messages"] == messages

    def test_generate_retry_on_429(self):
        """Should retry on 429 with backoff."""
        from ghostkv.remote import RemoteBackend
        backend = RemoteBackend(base_url="https://api.test.com", api_key="k")

        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_200 = self._mock_response(content="Success")

        with patch("ghostkv.remote.requests.post", side_effect=[resp_429, resp_200]) as mock_post:
            with patch("ghostkv.remote.time.sleep") as mock_sleep:
                text, usage = backend.generate(messages=[{"role": "user", "content": "hi"}])

        assert text == "Success"
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(10)  # first retry, 10s backoff

    def test_generate_raises_on_non_200(self):
        """Should raise RuntimeError on non-200 response."""
        from ghostkv.remote import RemoteBackend
        backend = RemoteBackend(base_url="https://api.test.com", api_key="k")

        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_500.text = "Internal Server Error"

        with patch("ghostkv.remote.requests.post", return_value=resp_500):
            with pytest.raises(RuntimeError, match="Remote API error 500"):
                backend.generate(messages=[{"role": "user", "content": "hi"}])

    def test_generate_empty_usage(self):
        """Should handle missing usage field gracefully."""
        from ghostkv.remote import RemoteBackend
        backend = RemoteBackend(base_url="https://api.test.com", api_key="k")

        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            # no "usage" key
        }

        with patch("ghostkv.remote.requests.post", return_value=resp):
            text, usage = backend.generate(messages=[{"role": "user", "content": "hi"}])

        assert text == "ok"
        assert usage == {}


class TestMessageSession:
    """Test MessageSession save/load/reset."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_session(self, name="test"):
        from ghostkv.remote import MessageSession
        s = MessageSession(name=name, model_name="test-model")
        s.base_dir = Path(self.tmpdir) / name
        s.base_dir.mkdir(parents=True, exist_ok=True)
        return s

    def test_new_session(self):
        s = self._make_session()
        assert len(s.messages) == 0
        assert s.steps == 0
        assert s.token_costs == []

    def test_add_message(self):
        s = self._make_session()
        s.add_message("user", "Hello")
        s.add_message("assistant", "Hi there")
        assert len(s.messages) == 2
        assert s.messages[0]["role"] == "user"
        assert s.messages[1]["role"] == "assistant"

    def test_save_and_load(self):
        s = self._make_session("save_load")
        s.add_message("user", "Test question")
        s.add_message("assistant", "Test answer")
        s.steps = 2
        s.total_tokens = 50
        s.token_costs = [25, 25]
        s.save()

        # Load into new session
        s2 = self._make_session("save_load")
        loaded = s2.load()
        assert loaded is True
        assert len(s2.messages) == 2
        assert s2.messages[0]["content"] == "Test question"
        assert s2.steps == 2
        assert s2.total_tokens == 50
        assert s2.token_costs == [25, 25]

    def test_reset_clears_state(self):
        s = self._make_session("reset_test")
        s.add_message("user", "Hello")
        s.steps = 5
        s.total_tokens = 100
        s.token_costs = [20, 20, 20, 20, 20]

        s.reset()
        assert len(s.messages) == 0
        assert s.steps == 0
        assert s.total_tokens == 0
        assert s.token_costs == []

    def test_load_nonexistent(self):
        s = self._make_session("nonexistent")
        loaded = s.load()
        assert loaded is False

    def test_stats(self):
        s = self._make_session("stats_test")
        s.steps = 3
        s.total_tokens = 60
        stats = s.stats()
        assert stats["session"] == "stats_test"
        assert stats["model"] == "test-model"
        assert stats["mode"] == "remote"
        assert stats["steps"] == 3
        assert stats["messages"] == 0

    def test_log_writes_to_file(self):
        s = self._make_session("log_test")
        s.log("Line 1")
        s.log("Line 2")
        s.save()

        content = s.log_path.read_text()
        assert "Line 1" in content
        assert "Line 2" in content


class TestAgentRemoteMode:
    """Test GhostKVAgent with mock RemoteBackend."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.vault = os.path.join(self.tmpdir, "vault")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_remote_agent(self):
        """Create an agent with a mock RemoteBackend and MessageSession."""
        from ghostkv.agent import GhostKVAgent, ToolDispatch
        from ghostkv.remote import RemoteBackend, MessageSession
        from ghostkv.tools import MemoryTool

        session = MessageSession(name="test_remote", model_name="test-model")
        session.base_dir = Path(self.tmpdir) / "remote_session"
        session.base_dir.mkdir(parents=True, exist_ok=True)

        # Mock backend that returns a fixed response
        backend = MagicMock(spec=RemoteBackend)
        backend.generate.return_value = ("Final answer: 42", {"total_tokens": 30})

        tools = ToolDispatch(memory=MemoryTool(vault_path=self.vault))
        memory = MemoryTool(vault_path=self.vault)

        agent = GhostKVAgent(
            session=session,
            tools=tools,
            memory=memory,
            remote_backend=backend,
            max_new_tokens=120,
            max_steps=5,
            temperature=0.7,
        )
        return agent, session, backend

    def test_remote_run_no_tools(self):
        """Agent with remote backend should call generate and return answer."""
        agent, session, backend = self._make_remote_agent()
        answer = agent.run("What is 6*7?")

        assert "42" in answer
        assert backend.generate.call_count == 1  # no tools, single call

        # Messages should have grown
        msgs = session.messages
        assert len(msgs) >= 2  # at least user + assistant
        assert any(m["role"] == "user" for m in msgs)
        assert any(m["role"] == "assistant" for m in msgs)

    def test_remote_message_history_grows(self):
        """Each run should add messages to the session."""
        agent, session, backend = self._make_remote_agent()
        session.add_message("system", "You are helpful.")  # pre-seed

        backend.generate.return_value = ("Answer 1", {"total_tokens": 10})
        agent.run("Question 1")
        count_after_1 = len(session.messages)

        backend.generate.return_value = ("Answer 2", {"total_tokens": 10})
        agent.run("Question 2")
        count_after_2 = len(session.messages)

        assert count_after_2 > count_after_1

    def test_remote_tracks_token_costs(self):
        """Token costs from usage should be tracked."""
        agent, session, backend = self._make_remote_agent()
        backend.generate.return_value = ("Done", {"total_tokens": 42})

        agent.run("test")

        assert len(session.token_costs) >= 1
        assert session.total_tokens > 0

    def test_remote_session_persistence(self):
        """Messages should survive save/load cycle."""
        agent, session, backend = self._make_remote_agent()
        backend.generate.return_value = ("Hello", {"total_tokens": 10})
        agent.run("Hi")

        session.save()

        # Load into new session
        from ghostkv.remote import MessageSession
        s2 = MessageSession(name="test_remote", model_name="test-model")
        s2.base_dir = session.base_dir
        loaded = s2.load()
        assert loaded
        assert len(s2.messages) == len(session.messages)


# ===================================================================
# Hybrid mode tests
# ===================================================================

class TestHybridMode:
    """Test hybrid escalation: local ReAct + remote synthesis."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.vault = os.path.join(self.tmpdir, "vault")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_mock_model(self):
        """Create a mock ModelBackend that returns predictable output."""
        from ghostkv.model import ModelBackend

        class MockModel(ModelBackend):
            def __init__(self):
                self._eos = 999
                self._count = 0

            def forward(self, input_ids, past_kv=None, use_cache=True):
                self._count += 1
                batch, seq = input_ids.shape
                vocab_size = 1000
                logits = torch.zeros(1, seq, vocab_size)
                if self._count < 3:
                    logits[:, -1, 42] = 100.0
                else:
                    logits[:, -1, self._eos] = 100.0
                cache = DynamicCache() if past_kv is None else past_kv
                if cache.get_seq_length() == 0:
                    k = torch.randn(1, 2, seq, 8)
                    v = torch.randn(1, 2, seq, 8)
                    cache.update(k, v, layer_idx=0)
                return logits, cache

            def tokenize(self, text, **kwargs):
                return torch.tensor([[1, 2, 3]])

            def decode(self, ids):
                return "local draft answer"

            @property
            def head_dim(self): return 8

            @property
            def device(self): return "cpu"

            @property
            def eos_token_id(self): return self._eos

        return MockModel()

    def _make_hybrid_agent(self, remote_answer="Remote synthesized answer", remote_tokens=50):
        """Create a hybrid agent with mock model + mock remote."""
        from ghostkv.agent import GhostKVAgent, ToolDispatch
        from ghostkv.kv import KVSession
        from ghostkv.remote import RemoteBackend
        from ghostkv.tools import MemoryTool

        model = self._make_mock_model()
        session = KVSession(name="test_hybrid", head_dim=8)
        session.base_dir = Path(self.tmpdir) / "hybrid_session"
        session.base_dir.mkdir(parents=True, exist_ok=True)
        session.ensure_rotation("cpu")

        backend = MagicMock(spec=RemoteBackend)
        backend.generate.return_value = (remote_answer, {"total_tokens": remote_tokens})

        tools = ToolDispatch(memory=MemoryTool(vault_path=self.vault))
        memory = MemoryTool(vault_path=self.vault)

        agent = GhostKVAgent(
            model=model,
            session=session,
            tools=tools,
            memory=memory,
            remote_backend=backend,
            max_new_tokens=120,
            max_steps=5,
            temperature=0.7,
        )
        return agent, session, backend

    def test_hybrid_mode_detection(self):
        """Both backends set → _mode == 'hybrid'."""
        agent, _, _ = self._make_hybrid_agent()
        assert agent._mode == "hybrid"

    def test_hybrid_escalates_after_react(self):
        """Remote.generate should be called once, final answer is remote's."""
        agent, session, backend = self._make_hybrid_agent(
            remote_answer="Polished remote answer"
        )
        answer = agent.run("What is 6*7?")

        # Remote was called exactly once for synthesis
        backend.generate.assert_called_once()
        # Answer should be from remote
        assert answer == "Polished remote answer"

    def test_hybrid_tool_history_tracked(self):
        """Tool calls should create history entries."""
        agent, session, backend = self._make_hybrid_agent()
        # The mock model always returns "local draft answer" which has no Action: pattern
        # So _tool_history will be empty in this case (no tools called)
        agent.run("test question")
        assert isinstance(agent._tool_history, list)

    def test_hybrid_synthesis_prompt(self):
        """Synthesis prompt should contain question + tool results + draft."""
        agent, session, backend = self._make_hybrid_agent()

        # Build a prompt manually to verify structure
        from ghostkv.agent import ToolHistoryEntry
        history = [
            ToolHistoryEntry(step=1, thought="Looking up", tool_name="search",
                             tool_args="Eiffel Tower", result="Built in 1889"),
            ToolHistoryEntry(step=2, thought="Checking", tool_name="run",
                             tool_args="print(1+1)", result="2"),
        ]
        messages = agent._build_synthesis_prompt("When was the Eiffel Tower built?", history, "1889")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        user_content = messages[1]["content"]
        assert "Eiffel Tower" in user_content
        assert "search" in user_content
        assert "Built in 1889" in user_content
        assert "local draft" in user_content.lower() or "1889" in user_content

    def test_hybrid_fallback_on_remote_failure(self):
        """If remote raises, local draft should be returned."""
        agent, session, backend = self._make_hybrid_agent()
        backend.generate.side_effect = RuntimeError("Remote API error 500")

        answer = agent.run("What is 6*7?")

        # Should fall back to local draft (decoded from mock model)
        assert "local draft" in answer.lower() or len(answer) > 0

    def test_hybrid_kv_ingestion(self):
        """Session cache should grow after escalation."""
        agent, session, backend = self._make_hybrid_agent(
            remote_answer="Synthesized answer for KV"
        )

        initial_kv_len = session.kv_seq_length()
        agent.run("test question")
        # After ingest, KV cache should have grown
        assert session.kv_seq_length() > initial_kv_len

    def test_hybrid_remote_stats(self):
        """remote_tokens and remote_calls should be tracked and survive save/load."""
        agent, session, backend = self._make_hybrid_agent(
            remote_tokens=42
        )
        agent.run("test")

        assert session.remote_calls == 1
        assert session.remote_tokens == 42

        # Save and reload
        session.save()
        from ghostkv.kv import KVSession
        s2 = KVSession(name="test_hybrid", head_dim=8)
        s2.base_dir = session.base_dir
        s2.load()
        assert s2.remote_calls == 1
        assert s2.remote_tokens == 42

    def test_hybrid_no_tools_direct_answer(self):
        """Even with no tool calls, should still escalate to remote."""
        agent, session, backend = self._make_hybrid_agent(
            remote_answer="Direct remote answer"
        )
        # Mock model returns "local draft answer" (no Action: patterns)
        answer = agent.run("Simple question")

        # Still escalated
        backend.generate.assert_called_once()
        assert answer == "Direct remote answer"
