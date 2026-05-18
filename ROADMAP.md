# GhostKV Roadmap

## v0.5.0 — MCP Support (Highest Impact)

Every major competitor (Claude Code, Goose, OpenCode) supports the Model Context Protocol. Without it, GhostKV can't connect to external tool servers.

- [x] Add MCP client to `ghostkv/mcp.py` — discover tools from MCP servers via stdio/SSE transport
- [x] Merge MCP tools into `ToolDispatch` alongside built-in tools
- [x] `--mcp` CLI flag to specify MCP server URLs (repeatable)
- [x] MCP tool schemas auto-converted to `Action: tool_name(args)` format for local model
- [x] Tests with mock MCP server

## v0.6.0 — Tree-sitter Repo Map (Aider Pattern)

Aider's killer feature is the Tree-sitter repo map — the LLM sees function signatures/types without the full codebase. This is how you do code-aware agents.

- [ ] `ghostkv/tools/repo.py` — Tree-sitter parser that builds a structural map of the codebase
- [ ] Auto-inject repo map into system prompt when `--repo` flag is set
- [ ] `Action: map()` tool to refresh/query the repo map
- [ ] Language support: Python, JS/TS, Rust, Go (Tree-sitter grammars)
- [ ] Tests with sample repos

## v0.7.0 — Typed Tool Dispatch

Regex `Action: tool_name("args")` parsing is fragile. Claude Code uses typed tool registries with structured args.

- [ ] `ToolSpec` dataclass — name, description, parameters (typed), handler
- [ ] JSON-arg parsing: `Action: search({"query": "test", "limit": 5})`
- [ ] Tool schema injection into system prompt (OpenAI function-calling style)
- [ ] Backward-compatible with regex format
- [ ] Tests

## v0.8.0 — Parallel Tool Execution

Claude Code's `StreamingToolExecutor` runs concurrency-safe tools in parallel during generation. GhostKV runs tools sequentially.

- [ ] `isConcurrencySafe` flag on each tool
- [ ] When model output contains multiple `Action:` lines, dispatch safe tools concurrently
- [ ] Collect results, feed all observations back at once
- [ ] Thread pool executor with configurable max concurrency
- [ ] Tests

## v0.9.0 — Streaming Output

GhostKV generates tokens silently then shows the result. Every competitor streams.

- [ ] `generate_step` yields tokens via generator (Python `yield`)
- [ ] REPL prints tokens as they arrive (typing effect)
- [ ] Hybrid mode: local streams, remote streams, synthesis streams
- [ ] `--no-stream` flag to disable
- [ ] Tests

## v1.0.0 — Production Hardening

- [ ] Dangerous action confirmation (file writes outside CWD, bash with `rm`, etc.)
- [ ] Token budget — stop loop if cumulative tokens exceed `--max-budget`
- [ ] Graceful error recovery — retry failed tool calls with backoff
- [ ] Session encryption option for API keys in metadata
- [ ] Windows/Linux/macOS CI via GitHub Actions
- [ ] PyPI package (`pip install ghostkv`)

## Research Explorations

- [ ] **KV cache editing** — LLMSteer/PASTA pattern: modify cached KV representations to inject personality or context without reprocessing
- [ ] **Multi-agent KV sharing** — Block pool pattern from "Agent Memory Below the Prompt": shared KV blocks across multiple agent instances
- [ ] **Adaptive routing** — classify question complexity locally, skip remote call for easy questions (pure local), escalate for hard ones
- [ ] **Cartridge pre-training** — learn compressed KV representations offline from documentation, load as "knowledge" at runtime
- [ ] **Context window overflow** — when KV exceeds model's max position embeddings, compress older layers more aggressively (PyramidKV pattern)

## v1.1.0 — Godot 4.6 Desktop GUI (GhostKV Desktop)

A native desktop application built with Godot 4.6 — the Claude Desktop of local KV-state agents. Full chat interface, real-time tool visualization, KV state inspector, and MCP server management.

- [ ] Chat UI — message bubbles with streaming output, markdown rendering, code syntax highlighting
- [ ] Tool call visualization — expandable cards showing Action → tool args → result for each ReAct step
- [ ] KV state display — real-time visualization of KV cache size, compression ratio, token count, session history
- [ ] MCP server management panel — add/remove MCP servers (stdio/SSE), see connected tools, test tool calls
- [ ] Session manager — switch between sessions, see session stats, create/delete sessions
- [ ] Memory vault browser — tree view of vault files, search, preview markdown
- [ ] Settings panel — model path, temperature/top-k/top-p sliders, remote API config, KV compression bits
- [ ] System tray integration — minimize to tray, global hotkey to show/hide
- [ ] Platform: Windows (primary), Linux, macOS via Godot export templates
- [ ] Communication: GDScript → Python subprocess (ghostkv CLI in `--json` mode) or embedded Python via GDExtension
