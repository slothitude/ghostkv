# GhostKV — Memory That Haunts, Not Repeats

> The agent that remembers without re-reading.

Every other agent re-feeds its entire conversation history on every step. Token cost grows O(n). Context windows fill up. History gets truncated.

GhostKV is different. **The agent's memory IS the KV cache.** Not the token sequence. Not a vector DB. Not a summary buffer. The actual attention state — compressed, serialized, and persisted across sessions.

Each agent step feeds only new tokens. Past context lives in the KV cache. Cost per step is O(1) regardless of conversation length. The KV state compresses to disk and restores on restart.

```
Step 1: 17 tokens   (question + system prompt)
Step 5: 18 tokens   (just the new query — KV has the history)
Step 50: 16 tokens  (still constant — KV grows, token cost doesn't)
```

## How It Works

```
┌─────────────────────────────────────┐
│            CLI REPL                 │
│  > ask question                     │
│  > see think/act/observe streaming  │
│  > KV state, token cost displayed   │
└──────────┬──────────────────────────┘
           │
┌──────────▼──────────────────────────┐
│         Agent Core                  │
│  - ReAct loop (think → act → obs)  │
│  - KV-state management (DynamicCache) │
│  - Sampling (temperature/top-k/top-p) │
│  - Tool dispatch                    │
│  - Jinja prompt templates           │
│  - Session persistence              │
└──┬─────┬──────┬──────┬─────────────┘
   │     │      │      │
   ▼     ▼      ▼      ▼
┌─────┐┌─────┐┌─────┐┌──────────┐
│Web  ││Code ││File ││Obsidian  │
│Srch ││Exec ││R/W  ││Graph Mem │
└─────┘└─────┘└─────┘└──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Markdown Vault     │
                    │  (persistent disk)  │
                    └─────────────────────┘
           │
┌──────────▼──────────────────────────┐
│       Model Interface               │
│  Abstract: transformers / GGUF      │
│  DynamicCache for KV state          │
│  TTQ compression (rotate→quant→deq) │
│  Symmetric TTQ (Q/K/V same frame)  │
│  Serialize/deserialize to disk      │
└─────────────────────────────────────┘
```

### The Core Insight

In a standard agent loop, each step re-encodes the full conversation:

```
Naive:  [system + history + new_query] → model → answer
        Token cost: O(n) where n = total history length
```

GhostKV injects past state as a KV cache and only feeds new tokens:

```
GhostKV: [new_query] + past_kv → model → answer
         Token cost: O(1) — constant, always just the new query
```

The model sees the full context through its KV cache — it just doesn't need to re-read it as tokens.

## Tools

| Tool | Description | Implementation |
|------|-------------|---------------|
| `search(query)` | Web search | SearXNG HTTP API |
| `run(code)` | Execute Python code | subprocess with timeout |
| `read(path)` | Read file | Direct filesystem access |
| `write(path, content)` | Write file | Direct filesystem access |
| `http(method, url, body?)` | HTTP requests | requests library |
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
Temperature 0.0  → greedy (deterministic, same as argmax)
Temperature 0.8  → focused but varied (default)
Temperature 1.5+ → creative/divergent
Top-k 50         → only sample from 50 most likely tokens
Top-p 0.9        → nucleus sampling (smallest set with ≥90% cumulative probability)
```

## Symmetric TTQ

Optional inline compression that applies rotate→quantize→derotate to **all three** attention projections (Q, K, V) in the same coordinate frame during the forward pass. When Q and K share the same rotation, quantization noise becomes symmetric in the dot product Q·K^T and partially cancels.

Standard TTQ (post-hoc): compress K/V only, after generation.
Symmetric TTQ (inline): compress Q/K/V during generation, same rotation.

Enabled with `--symmetric-ttq`. Proven in ResolutionRouter Stage 7e: PPL 7.76 (symmetric) vs 7.92 (standard TTQ) — noise cancellation is real.

## KV Persistence

Sessions serialize to `~/.ghostkv/sessions/<name>/`:

```
~/.ghostkv/sessions/default/
  kv_cache.bin        # Serialized DynamicCache (zlib compressed)
  metadata.json       # {model, steps, tokens, head_dim}
  conversation.log    # Full text log
```

- On startup: load session if it exists, restore KV state
- Each step: KV updated in memory
- `/save` or every N steps: serialize KV to disk
- `--session` flag to run named sessions (multiple agents, multiple contexts)

## CLI

```
$ python -m ghostkv.main --model /path/to/model
GhostKV v0.2 — memory that haunts, not repeats
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

  Steps: 2 | Tokens: 34 | KV: 156 tokens (3.2 KB compressed)
  Memory written: eiffel_tower_builder.md

> What else do you know about Dijon?
[uses KV from previous question — no re-feeding]

  Steps: 1 | Tokens: 18 | KV: 204 tokens (4.1 KB compressed)

> /save
Session saved. KV: 204 tokens | Vault: 2 files

> /stats
  Session: default
  Model:   qwen3-4b (4-bit NF4)
  Steps:   3
  Tokens:  52 (avg 17/step)
  KV:      204 tokens
  KV disk: 4.1 KB

> /quit
Saved. KV: 204 tokens | Vault: 2 files
Goodbye.
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

  --model PATH          Path to HF model (default: C:/Users/aaron/qwen3_exp/model)
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
```

## Architecture

### File Structure

```
ghostkv/
  __init__.py              # Package root, version
  main.py                  # CLI REPL entry point
  agent.py                 # ReAct agent loop, sampling, tool dispatch
  model.py                 # ModelBackend ABC + TransformersBackend + symmetric TTQ
  kv.py                    # DynamicCache, TTQ compression, multihead, serialization
  tools/
    __init__.py
    search.py              # SearXNG web search
    code.py                # Python subprocess execution
    files.py               # File read/write
    http.py                # HTTP requests
    memory.py              # Obsidian vault read/write/search
  templates/
    system.j2              # Agent identity, tool listing, output format
    observe.j2             # Tool result → observation
    tool_call.j2           # Tool call passthrough
    memory.j2              # Inject recalled memories
    error.j2               # Tool error formatting

tests/
  test_ghostkv.py          # 75 tests covering all components
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

### KV Compression (TTQ)

Training-free KV cache compression adapted from TurboQuant:

1. **Rotate** — multiply KV tensors by a random orthogonal matrix (spreads information uniformly)
2. **Quantize** — aggressive 3-bit symmetric uniform quantization (safe after rotation)
3. **Derotate** — rotate back to original coordinate frame

This is the same pipeline proven in Stages 5-7 of the ResolutionRouter research, inlined here with no external dependencies.

### Symmetric TTQ

Extends TTQ to compress Q alongside K/V during the forward pass:

1. Monkey-patches Q/K/V projection layers on all attention heads
2. Each projection output: split into per-head chunks → rotate → quantize → derotate
3. All three use the same rotation matrix — quantization noise cancels in Q·K^T
4. Uses monkey-patching (not `register_forward_hook`) — hooks segfault on BitsAndBytes layers

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
    ↓
[Recall memories from vault]
    ↓
[Build prompt: system + memories + question]
    ↓
[Forward pass with past KV → response]
    ↓
[Sample next token: temperature + top-k + top-p]
    ↓
[Parse response for Action: tool_name(args)]
    ↓ yes                      ↓ no
[Execute tool]              [Final answer]
    ↓                           ↓
[Feed observation as      [Save to memory vault]
 new tokens only]          [Update session stats]
    ↓
[Back to parsing]
```

The key: observations are fed as **new tokens only**. The model's past context is in the KV cache — it doesn't re-read.

## Test Suite

75 tests covering every component without requiring a GPU:

```
$ python -m pytest tests/ -v

KV Compression ............ 11 passed (orthogonal, shapes, 4-bit > 3-bit, round-trip, multihead)
KV Serialization .......... 5 passed  (round-trip, empty, single, large, fp16)
KV Session ................ 9 passed  (save/load, reset, stats, log, size)
Search Tool ............... 2 passed  (instantiation, offline error)
Code Tool ................. 6 passed  (print, math, stderr, timeout, exceptions)
File Tools ................ 5 passed  (read/write, dirs, truncate, edge cases)
HTTP Tool ................. 3 passed  (instantiation, no URL, offline error)
Memory Tool ............... 8 passed  (write/recall, frontmatter, ranking, wikilinks)
Tool Dispatch ............. 10 passed (regex parsing, dispatch, available tools)
Generate Step ............. 1 passed  (mock model autoregressive)
Sampling .................. 7 passed  (greedy, temperature, top-k, top-p, integration)
Agent ReAct ............... 2 passed  (multiline regex, tag extraction)
Templates ................. 4 passed  (system, observe, memory, error)
Integration ............... 2 passed  (full pipeline, compress→serialize→restore)
                                   ---
                              75 passed
```

## Dependencies

- Python >=3.10
- PyTorch >=2.0
- transformers >=4.40 (for BitsAndBytes quantization)
- jinja2 (template rendering)
- requests (HTTP tool, SearXNG search)
- pytest >=8.0 (dev)

## Provenance

Built on research from the ResolutionRouter project (Stages 7a-7e):

| Component | Origin | What it proved |
|-----------|--------|----------------|
| KV-state agent loop | Stage 7a | O(1) per-step token cost at equal accuracy |
| KV serialization | Stage 7c | DynamicCache portability (zlib binary format) |
| TTQ compression | Stage 7d | rotate→quantize→derotate for KV compression |
| Symmetric TTQ | Stage 7e | Q/K/V same-frame rotation for noise cancellation |
| head_dim fix | Stage 7e | `getattr(config, "head_dim", hidden_size // heads)` |
| DynamicCache API | Stage 7c | `list(cache) → (K,V,extra)`, `cache.update(k,v,layer_idx)` |

## Changelog

### v0.2.0
- **Sampling**: temperature + top-k + top-p (nucleus) sampling replaces greedy argmax. Varied, non-repetitive output. `--temperature`, `--top-k`, `--top-p` CLI flags.
- **Symmetric TTQ**: inline Q/K/V compression during the forward pass. All three projections share the same rotation matrix for symmetric noise cancellation in Q·K^T. `--symmetric-ttq` and `--query-bits` CLI flags.
- **Multihead compression**: `compress_tensor_multihead()` splits projection outputs into per-head chunks for correct TTQ on multi-head attention.
- **9 new tests** (66 → 75).

### v0.1.0
- Initial release: ReAct loop, KV-state persistence, TTQ compression, Obsidian vault memory, 6 tools, 66 tests.

## License

MIT
