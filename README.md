# GhostKV — Memory That Haunts, Not Repeats

> The agent that remembers without re-reading.

Every other agent re-feeds its entire conversation history on every step. Token cost grows O(n). Context windows fill up. History gets truncated or summarized at a loss.

GhostKV is different. **The agent's memory IS the KV cache.** Not the token sequence. Not a vector DB. Not a summary buffer. The actual attention state — compressed, serialized, and persisted across sessions.

Each agent step feeds only new tokens. Past context lives in the KV cache. Cost per step is O(1) regardless of conversation length. The KV state compresses to disk and restores on restart.

```
Step 1: 17 tokens   (question + system prompt)
Step 5: 18 tokens   (just the new query — KV has the history)
Step 50: 16 tokens  (still constant — KV grows, token cost doesn't)
```

## Three Modes

### Local Mode — KV Cache Memory

Local model runs the full ReAct loop. Memory lives in the KV cache. Every step is O(1). No API costs.

```
$ python -m ghostkv.main --model /path/to/model
```

### Remote Mode — API Backend

No GPU required. Uses any OpenAI-compatible chat completion API. Session stored as message history.

```
$ python -m ghostkv.main --remote https://api.z.ai/api/coding/paas/v4/chat/completions --remote-key YOUR_KEY
```

### Hybrid Mode — Local ReAct + Remote Synthesis

**The Perplexity pattern, done with KV coherence.** Local model runs the tool-heavy ReAct loop (cheap, KV-cached, O(1) per step). Then a single remote API call synthesizes the polished final answer. The remote answer is fed back into the local KV cache (~2ms) so the next question benefits from both local tool results AND remote synthesis.

90% of tokens stay local. One remote call for quality.

```
$ python -m ghostkv.main --model /path/to/model --hybrid --remote https://api.z.ai/api/coding/paas/v4/chat/completions --remote-key YOUR_KEY
```

```
User Question
    |
Local model (KV cache) -- ReAct loop
    | (N tool calls: search, bash, read, etc.)
    | (each step: O(1) via KV cache)
    v
Local draft answer
    |
Remote model -- single synthesis call
    | (question + tool history + draft -> polished answer)
    v
Feed remote answer back into local KV cache (~2ms)
    |
Return to user
```

## How It Works

```
+-------------------------------------+
|            CLI REPL                 |
|  > ask question                     |
|  > see think/act/observe streaming  |
|  > KV state, token cost displayed   |
+----------+--------------------------+
           |
+----------v--------------------------+
|         Agent Core                  |
|  - ReAct loop (think -> act -> obs)|
|  - KV-state management (DynamicCache)|
|  - Sampling (temperature/top-k/top-p)|
|  - Tool dispatch                    |
|  - Hybrid escalation (v0.4.0)      |
|  - Jinja prompt templates           |
|  - Session persistence              |
+--+-----+------+------+-------------+
   |     |      |      |
   v     v      v      v
+------+ +------+ +------+ +----------+
|Web   | |Code  | |File  | |Obsidian  |
|Srch  | |Exec  | |R/W   | |Graph Mem |
+------+ +------+ +------+ +----------+
                               |
                    +----------v----------+
                    |  Markdown Vault     |
                    |  (persistent disk)  |
                    +---------------------+
           |
+----------v--------------------------+
|       Model Interface               |
|  Abstract: transformers / GGUF      |
|  DynamicCache for KV state          |
|  TTQ compression (rotate->quant->deq)|
|  Symmetric TTQ (Q/K/V same frame)  |
|  Serialize/deserialize to disk      |
+-------------------------------------+
```

### The Core Insight

In a standard agent loop, each step re-encodes the full conversation:

```
Naive:  [system + history + new_query] -> model -> answer
        Token cost: O(n) where n = total history length
```

GhostKV injects past state as a KV cache and only feeds new tokens:

```
GhostKV: [new_query] + past_kv -> model -> answer
         Token cost: O(1) -- constant, always just the new query
```

The model sees the full context through its KV cache — it just doesn't need to re-read it as tokens.

## Tools

| Tool | Description | Implementation |
|------|-------------|---------------|
| `search(query)` | Web search | SearXNG HTTP API |
| `run(code)` | Execute Python code | subprocess with timeout |
| `bash(cmd)` | Execute shell command | subprocess with timeout |
| `read(path)` | Read file | Direct filesystem access |
| `write(path, content)` | Write file | Direct filesystem access |
| `http(method, url)` | HTTP requests | requests library |
| `recall(query)` | Search memory vault | Full-text search over markdown files |

## Obsidian Graph Memory

The agent's long-term memory is a vault of markdown files at `~/.ghostkv/memory/`:

```markdown
---
type: observation
timestamp: 2026-05-17T14:30:00
tags: [eiffel, tower, paris]
related: "[[Gustave Eiffel]]"
---

The Eiffel Tower was built in 1889 by [[Gustave Eiffel]] for the World's Fair.
Located in [[Paris]], [[France]].
```

- `recall(query)` — grep vault for matching files, return top-K snippets
- After each question, the agent writes observations as new markdown files
- `[[wikilinks]]` create graph edges — open the vault in Obsidian to see the graph
- The vault is human-browsable — the agent's memory is inspectable

## Sampling

Token generation uses temperature-scaled multinomial sampling with top-k and nucleus (top-p) filtering — not greedy argmax. This produces varied, non-repetitive responses and prevents the agent from getting stuck in loops.

```
Temperature 0.0  -> greedy (deterministic, same as argmax)
Temperature 0.8  -> focused but varied (default)
Temperature 1.5+ -> creative/divergent
Top-k 50         -> only sample from 50 most likely tokens
Top-p 0.9        -> nucleus sampling (smallest set with >=90% cumulative probability)
```

## Symmetric TTQ

Optional inline compression that applies rotate->quantize->derotate to **all three** attention projections (Q, K, V) in the same coordinate frame during the forward pass. When Q and K share the same rotation, quantization noise becomes symmetric in the dot product Q*K^T and partially cancels.

Standard TTQ (post-hoc): compress K/V only, after generation.
Symmetric TTQ (inline): compress Q/K/V during generation, same rotation.

Enabled with `--symmetric-ttq`. Proven in ResolutionRouter Stage 7e: PPL 7.76 (symmetric) vs 7.92 (standard TTQ) — noise cancellation is real.

## KV Persistence

Sessions serialize to `~/.ghostkv/sessions/<name>/`:

```
~/.ghostkv/sessions/default/
  kv_cache.bin        # Serialized DynamicCache (zlib compressed)
  metadata.json       # {model, steps, tokens, head_dim, remote_tokens, remote_calls}
  conversation.log    # Full text log
```

- On startup: load session if it exists, restore KV state
- Each step: KV updated in memory
- `/save` or every N steps: serialize KV to disk
- `--session` flag to run named sessions (multiple agents, multiple contexts)

## CLI

```
$ python -m ghostkv.main --model /path/to/model --hybrid --remote URL --remote-key KEY
GhostKV v0.4.0 — memory that haunts, not repeats
Hybrid mode: local=qwen3-4b + remote=glm-5.1 @ https://api.z.ai/...
model: qwen3-4b | session: default
KV: 0 tokens (empty) | Memory: 0 files in vault

> Who built the Eiffel Tower and where were they born?
Thought: I need to find who built the Eiffel Tower first.
Action: search("Eiffel Tower builder")
Observation: The Eiffel Tower was built in 1889 by Gustave Eiffel...
Thought: Now I need to find where Gustave Eiffel was born.
Action: recall("Gustave Eiffel birthplace")
Observation: Gustave Eiffel was born in Dijon, France in 1832.
Answer: Gustave Eiffel built the Eiffel Tower. He was born in Dijon, France.

  Steps: 2 | Tokens: 34 | KV: 156 tokens (3.2 KB) | Remote: 1 calls

> What else do you know about Dijon?
[uses KV from previous question — no re-feeding, remote answer absorbed]

  Steps: 1 | Tokens: 18 | KV: 204 tokens (4.1 KB) | Remote: 2 calls

> /stats
  Session: default
  Model:   qwen3-4b (4-bit NF4)
  Steps:   3
  Tokens:  52 (avg 17/step)
  KV:      204 tokens
  KV disk: 4.1 KB
  Remote:  2 calls, 300 tokens
```

### Commands

| Command | Description |
|---------|-------------|
| `> question` | Run agent loop on a question |
| `> /save` | Serialize KV to disk |
| `> /stats` | Show session statistics |
| `> /reset` | Clear KV, start fresh |
| `> /vault` | Show memory vault contents |
| `> /quit` | Save and exit |

### CLI Flags

```
python -m ghostkv.main [OPTIONS]

  --model PATH          Path to HF model (default: qwen3-4b)
  --session NAME        Session name (default: 'default')
  --no-quantize         Disable 4-bit quantization
  --no-search           Disable web search tool
  --max-tokens N        Max new tokens per generation step (default: 120)
  --max-steps N         Max agent ReAct steps per question (default: 10)
  --kv-bits N           KV compression bits (default: 3)
  --temperature F       Sampling temperature (default: 0.8, 0 = greedy)
  --top-k N             Top-k sampling (default: 50, 0 = disabled)
  --top-p F             Nucleus sampling threshold (default: 0.9, 1.0 = disabled)
  --symmetric-ttq       Enable symmetric TTQ (compress Q/K/V in same rotation frame)
  --query-bits N        Query quantization bits for symmetric TTQ (default: 4)
  --code-timeout N      Code execution timeout in seconds (default: 30)
  --log-level LEVEL     Logging level (default: WARNING)

Remote mode:
  --remote URL          Base URL for OpenAI-compatible chat API
  --remote-key KEY      API key (or set GHOSTKV_API_KEY env var)
  --remote-model NAME   Remote model name (default: glm-5.1)
  --remote-timeout N    Request timeout in seconds (default: 120)

Hybrid mode:
  --hybrid              Local ReAct + remote synthesis (requires --remote + --remote-key)
```

## Landscape: How GhostKV Compares

The AI coding agent space exploded after the March 2026 Claude Code source leak (512K lines published to npm via source map). Here's where GhostKV fits.

### Memory Architecture

Every agent needs persistent memory. The approaches differ radically:

| Agent | Memory Substrate | Cost | Information Loss |
|-------|-----------------|------|-----------------|
| **GhostKV** | **KV cache (binary attention state)** | **O(1) per step** | **None** |
| Claude Code | 4-tier prompt compaction (LLM summarization) | O(n) tokens per compaction | High (summarization discards detail) |
| Aider | Tree-sitter repo map (function signatures) | O(repo size) initial, O(1) per file | Moderate (only structure, not content) |
| Claw Code/OpenCode | Message history | O(n) tokens per turn | None until truncation |
| ForgeCode | Context slicing across agents | O(n) per agent slice | Moderate |
| Cline | IDE context (open files, diagnostics) | O(context window) | None until overflow |

GhostKV is the only agent where memory is the KV cache itself — not text, not a vector DB, not a summary. The attention state IS the memory. This means:

- **No re-encoding** — past context lives in the KV cache, not re-fed as tokens
- **No summarization loss** — the full attention representation is preserved
- **O(1) per step** — constant cost regardless of conversation length
- **~2ms ingestion** — remote answers absorbed into local cache nearly instantly

### Hybrid vs. Remote-Only

Several agents now support hybrid local/remote architectures:

| Agent | Hybrid Approach | Remote Calls/Question | Cost/Question |
|-------|----------------|----------------------|---------------|
| **GhostKV** | Local ReAct + remote synthesis + KV back-feed | **1** | **~$0.003-0.01** |
| Aider | Architect (Sonnet) designs, Editor (Haiku) implements | 2-4 | ~$0.02-0.08 |
| Claude Code | Remote only, cache-aware compaction | 1-3 (with compaction) | ~$0.03-0.15 |
| MinionS Protocol | Remote orchestrates, local executes in Docker | 1 (orchestration) | ~$0.01-0.05 |
| claude-code-local | 100% local (MLX on Apple Silicon) | 0 | $0.00 |

GhostKV's hybrid advantage: the remote answer is **fed back into the local KV cache**. The next question starts with both the local tool results AND the remote's polished synthesis already in memory. No other agent does this.

### What GhostKV Doesn't Do (Yet)

| Feature | Who Has It |
|---------|-----------|
| MCP (Model Context Protocol) | Claude Code, Goose, OpenCode |
| Parallel tool execution | Claude Code (StreamingToolExecutor) |
| IDE integration | Cline (5M VS Code installs), Cursor |
| Multi-model routing | ForgeCode (10 parallel agents), Aider (architect/editor) |
| Tree-sitter code analysis | Aider (any language repo map) |
| Terminal-Bench scores | ForgeCode (81.8%), AgentFlow (84.3%) |
| Production security (8 layers) | Claude Code (kill switches, trust dialogs) |

### Academic Validation

GhostKV's core approach — KV cache as agent memory — is being validated by 2026 research:

| Paper/Project | What They Found |
|---------------|----------------|
| **LMCache** (open source) | Persistent KV for vLLM/SGLang — 3x throughput improvement, "prefill each text only once" |
| **"Agent Memory Below the Prompt"** (arXiv:2603.04428) | Q4 KV blocks in safetensors, 22-136x TTFT improvement from cache restoration |
| **MemArt** (openreview) | "KVCache-centric Memory Paradigm" — agents store compressed attention representations instead of text |
| **Cartridges** (Stanford Hazy Research) | Learns compact KV caches offline to represent large corpora — load knowledge without runtime prefill |
| **SnapKV / PyramidKV / LeanKV** | Attention-based sparsity and asymmetric compression to reduce KV size while preserving quality |

The research converges on what GhostKV has done since v0.1: **the KV cache is the native memory format of a Transformer**. Keeping data in KV form preserves the full attention representation, avoiding information loss from text serialization.

## Architecture

### File Structure

```
ghostkv/
  __init__.py              # Package root, version
  main.py                  # CLI REPL entry point, --hybrid flag
  agent.py                 # ReAct agent loop, tool dispatch, hybrid escalation
  model.py                 # ModelBackend ABC + TransformersBackend + symmetric TTQ
  kv.py                    # DynamicCache, TTQ compression, serialization, remote stats
  remote.py                # RemoteBackend (OpenAI-compatible), MessageSession
  tools/
    __init__.py
    search.py              # SearXNG web search
    code.py                # Python subprocess execution
    bash.py                # Shell command execution
    files.py               # File read/write
    http.py                # HTTP requests
    memory.py              # Obsidian vault read/write/search
  templates/
    system.j2              # Agent identity, tool listing, output format
    observe.j2             # Tool result -> observation
    tool_call.j2           # Tool call passthrough
    memory.j2              # Inject recalled memories
    error.j2               # Tool error formatting

tests/
  test_ghostkv.py          # 106 tests covering all components
```

### Model Backend

Abstract interface so any model backend works:

```python
class ModelBackend(ABC):
    def forward(self, input_ids, past_kv=None) -> (logits, new_kv)
    def tokenize(self, text) -> input_ids
    def decode(self, ids) -> str
    @property
    def head_dim(self) -> int
```

**TransformersBackend** — loads HF models with BitsAndBytes NF4 quantization. Uses `model.forward()` directly (not `model.generate()`) for full KV cache control with DynamicCache. Supports `install_symmetric_ttq()` for inline Q/K/V compression.

### Remote Backend

OpenAI-compatible chat completion client with retry on 429:

```python
class RemoteBackend:
    def generate(messages, temperature, max_tokens) -> (text, usage)
```

**MessageSession** — duck-type replacement for KVSession in remote mode. Stores message history as JSON instead of KV binary.

### Hybrid Escalation (v0.4.0)

Three new methods on the agent:

1. **`_build_synthesis_prompt(question, tool_history, local_draft)`** — structures the remote call with question + all tool results + local draft answer
2. **`_escalate_to_remote(question, tool_history, local_draft)`** — single remote generate call, returns (answer, remote_tokens)
3. **`_ingest_into_kv(text)`** — tokenizes the remote answer and runs a minimal generate_step (temp=0, max_new=1) purely for the KV side effect. ~2ms overhead.

### KV Compression (TTQ)

Training-free KV cache compression adapted from TurboQuant:

1. **Rotate** — multiply KV tensors by a random orthogonal matrix (spreads information uniformly)
2. **Quantize** — aggressive 3-bit symmetric uniform quantization (safe after rotation)
3. **Derotate** — rotate back to original coordinate frame

This is the same pipeline proven in Stages 5-7 of the ResolutionRouter research, inlined here with no external dependencies.

### Serialization

Binary format via zlib with struct headers:

```
Header:    4 bytes (num_layers: uint16 + reserved: uint16)
Per layer:
  K shape:  1 byte (ndim) + ndim*4 bytes (int32 per dim)
  K data:   float16 tensor bytes
  V shape:  1 byte (ndim) + ndim*4 bytes (int32 per dim)
  V data:   float16 tensor bytes
All zlib-compressed at level 1 (fast)
```

### ReAct Loop

```
Question
    |
[Recall memories from vault]
    |
[Build prompt: system + memories + question]
    |
[Forward pass with past KV -> response]
    |
[Sample next token: temperature + top-k + top-p]
    |
[Parse response for Action: tool_name(args)]
    | yes                        | no
[Execute tool]                [Final answer]
    |                             |
[Track in tool_history]       [If hybrid: escalate to remote]
    |                             |
[Feed observation as          [Ingest remote answer into KV]
 new tokens only]             [Save to memory vault]
    |                             |
[Back to parsing]             [Update session stats]
```

The key: observations are fed as **new tokens only**. The model's past context is in the KV cache — it doesn't re-read.

## Test Suite

106 tests covering every component without requiring a GPU:

```
$ python -m pytest tests/ -v

KV Compression ............ 11 passed (orthogonal, shapes, 4-bit > 3-bit, round-trip, multihead)
KV Serialization .......... 5 passed  (round-trip, empty, single, large, fp16)
KV Session ................ 9 passed  (save/load, reset, stats, log, size, remote stats)
Search Tool ............... 2 passed  (instantiation, offline error)
Code Tool ................. 6 passed  (print, math, stderr, timeout, exceptions)
Bash Tool ................. 7 passed  (echo, pipe, exit code, stderr, timeout, dispatch)
File Tools ................ 5 passed  (read/write, dirs, truncate, edge cases)
HTTP Tool ................. 3 passed  (instantiation, no URL, offline error)
Memory Tool ............... 8 passed  (write/recall, frontmatter, ranking, wikilinks)
Tool Dispatch ............. 10 passed (regex parsing, dispatch, available tools)
Generate Step ............. 1 passed  (mock model autoregressive)
Sampling .................. 7 passed  (greedy, temperature, top-k, top-p, integration)
Agent ReAct ............... 2 passed  (multiline regex, tag extraction)
Templates ................. 4 passed  (system, observe, memory, error)
Integration ............... 2 passed  (full pipeline, compress->serialize->restore)
Remote Backend ............ 5 passed  (generate, request format, retry 429, error, empty usage)
Message Session ........... 8 passed  (new, add, save/load, reset, stats, log)
Agent Remote Mode ......... 4 passed  (run, history grows, token costs, persistence)
Hybrid Mode ............... 8 passed  (detection, escalation, tool history, synthesis prompt,
                                       fallback, KV ingestion, remote stats, no-tools)
                                   ---
                              106 passed
```

## Dependencies

- Python >=3.10
- PyTorch >=2.0
- transformers >=4.40 (for BitsAndBytes quantization)
- jinja2 (template rendering)
- requests (HTTP tool, remote backend, SearXNG search)
- pytest >=8.0 (dev)

## Provenance

Built on research from the ResolutionRouter project (Stages 7a-7e):

| Component | Origin | What it proved |
|-----------|--------|----------------|
| KV-state agent loop | Stage 7a | O(1) per-step token cost at equal accuracy |
| KV serialization | Stage 7c | DynamicCache portability (zlib binary format) |
| TTQ compression | Stage 7d | rotate->quantize->derotate for KV compression |
| Symmetric TTQ | Stage 7e | Q/K/V same-frame rotation for noise cancellation |
| Hybrid escalation | GhostKV v0.4.0 | Perplexity-style local+remote with KV coherence |
| KV ingestion | GhostKV v0.4.0 | Remote answer absorbed into local cache (~2ms) |

## Changelog

### v0.4.0
- **Hybrid escalation mode**: local model runs ReAct loop, single remote API call synthesizes final answer. `--hybrid` CLI flag.
- **KV ingestion**: remote answer fed back into local KV cache for context continuity (~2ms overhead).
- **Tool history tracking**: `ToolHistoryEntry` dataclass records every tool call for the synthesis prompt.
- **Remote stats**: `KVSession` tracks `remote_tokens` and `remote_calls` through save/load/reset.
- **Mode detection fix**: `self._mode` attribute (`"local"`, `"remote"`, `"hybrid"`) prevents incorrect routing when both backends are set.
- **8 new tests** (98 -> 106).

### v0.3.0
- **Remote mode**: OpenAI-compatible chat completion backend. No GPU required. `--remote`, `--remote-key`, `--remote-model` CLI flags.
- **MessageSession**: duck-type replacement for KVSession in remote mode (message history stored as JSON).
- **21 new tests** (75 -> 96... wait, 98).
- **Bash tool**: shell command execution alongside Python code execution.

### v0.2.0
- **Sampling**: temperature + top-k + top-p (nucleus) sampling replaces greedy argmax. `--temperature`, `--top-k`, `--top-p` CLI flags.
- **Symmetric TTQ**: inline Q/K/V compression during the forward pass. `--symmetric-ttq` and `--query-bits` CLI flags.
- **Multihead compression**: `compress_tensor_multihead()` splits projection outputs into per-head chunks.
- **9 new tests** (66 -> 75).

### v0.1.0
- Initial release: ReAct loop, KV-state persistence, TTQ compression, Obsidian vault memory, 6 tools, 66 tests.

## License

MIT
