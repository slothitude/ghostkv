"""GhostKV CLI — interactive REPL for the KV-state agent.

Usage:
    ghostkv                          # Start with default session
    ghostkv --session work           # Named session
    ghostkv --model /path/to/model   # Specific model
    ghostkv --no-quantize            # Full precision (no 4-bit)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ghostkv import __version__
from ghostkv.agent import GhostKVAgent, ToolDispatch
from ghostkv.kv import KVSession
from ghostkv.model import TransformersBackend
from ghostkv.tools import (
    SearchTool,
    CodeTool,
    FileReadTool,
    FileWriteTool,
    HttpTool,
    MemoryTool,
)

logger = logging.getLogger("ghostkv")

BANNER = f"""\
GhostKV v{__version__} — memory that haunts, not repeats
"""


def build_agent(args: argparse.Namespace) -> GhostKVAgent:
    """Construct the full agent from CLI arguments."""
    # Load model backend
    print(f"Loading model from {args.model}...")
    model = TransformersBackend(
        model_path=args.model,
        quantize_4bit=not args.no_quantize,
    )
    print(f"Model ready. head_dim={model.head_dim}, device={model.device}")

    # Create or load session
    session = KVSession(
        name=args.session,
        model_name=Path(args.model).name,
        head_dim=model.head_dim,
        kv_bits=args.kv_bits,
    )
    loaded = session.load(device=model.device)
    if loaded:
        print(f"Session '{args.session}' restored: {session.kv_seq_length()} KV tokens, "
              f"{session.steps} steps")
    else:
        print(f"New session: {args.session}")

    # Install symmetric TTQ if requested
    if args.symmetric_ttq:
        session.ensure_rotation(model.device)
        installed = model.install_symmetric_ttq(
            rotation=session.rotation,
            k_bits=args.kv_bits,
            v_bits=args.kv_bits,
            q_bits=args.query_bits,
        )
        if installed:
            print(f"Symmetric TTQ active: K={args.kv_bits}b V={args.kv_bits}b Q={args.query_bits}b")
        else:
            print("Warning: symmetric TTQ requested but no attention layers found")

    # Initialize tools
    tools = ToolDispatch(
        search=SearchTool() if not args.no_search else None,
        code=CodeTool(timeout=args.code_timeout),
        file_read=FileReadTool(),
        file_write=FileWriteTool(),
        http=HttpTool(),
        memory=MemoryTool(),
    )

    # Build agent
    agent = GhostKVAgent(
        model=model,
        session=session,
        tools=tools,
        memory=MemoryTool(),
        max_new_tokens=args.max_tokens,
        max_steps=args.max_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    return agent


def cmd_stats(session: KVSession):
    """Print session statistics."""
    stats = session.stats()
    print(f"\n  Session: {stats['session']}")
    print(f"  Model:   {stats['model']}")
    print(f"  Steps:   {stats['steps']}")
    print(f"  Tokens:  {stats['total_tokens']} (avg {stats['avg_tokens_per_step']:.0f}/step)")
    print(f"  KV:      {stats['kv_tokens']} tokens")
    if stats['kv_tokens'] > 0:
        compressed = stats['kv_compressed_size']
        print(f"  KV disk: {compressed:,} bytes ({compressed/1024:.1f} KB)")
    print()


def cmd_vault(memory: MemoryTool):
    """Open vault info."""
    files = memory.list_files()
    print(f"\n  Vault: {memory.vault_path}")
    print(f"  Files: {len(files)}")
    for f in files[:20]:
        print(f"    - {f}")
    if len(files) > 20:
        print(f"    ... and {len(files) - 20} more")
    print()


def repl(agent: GhostKVAgent):
    """Run the interactive REPL."""
    session = agent.session
    memory = agent.memory

    print(BANNER)
    print(f"  model: {session.model_name} | session: {session.name}")
    print(f"  KV: {session.kv_seq_length()} tokens | Memory: {memory.count()} files in vault")
    print()
    print("  Type a question to ask, or use /commands:")
    print("    /save   — Save KV state to disk")
    print("    /stats  — Show session statistics")
    print("    /reset  — Clear KV, start fresh")
    print("    /vault  — Show memory vault contents")
    print("    /help   — Show commands")
    print("    /quit   — Save and exit")
    print()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        # Handle commands
        if line.startswith("/"):
            cmd = line.lower().split()[0]

            if cmd in ("/quit", "/exit", "/q"):
                print("Saving session...")
                session.save()
                print(f"Saved. KV: {session.kv_seq_length()} tokens | Vault: {memory.count()} files")
                print("Goodbye.")
                break

            elif cmd == "/save":
                session.save()
                print(f"Session saved. KV: {session.kv_seq_length()} tokens | "
                      f"Vault: {memory.count()} files")

            elif cmd == "/stats":
                cmd_stats(session)

            elif cmd == "/reset":
                session.reset()
                print("Session reset. KV cleared.")

            elif cmd == "/vault":
                cmd_vault(memory)

            elif cmd == "/help":
                print("\n  Commands:")
                print("    /save   — Save KV state to disk")
                print("    /stats  — Show session statistics")
                print("    /reset  — Clear KV, start fresh")
                print("    /vault  — Show memory vault contents")
                print("    /quit   — Save and exit")
                print()

            else:
                print(f"Unknown command: {cmd}. Type /help for available commands.")

        else:
            # Run agent on the question
            try:
                answer = agent.run(line)

                print()
                print(answer)
                print()

                kv_tokens = session.kv_seq_length()
                step_tokens = session.token_costs[-1] if session.token_costs else 0
                compressed_size = session.kv_size_bytes()
                print(f"  Steps: {session.steps} | Tokens: {step_tokens} | "
                      f"KV: {kv_tokens} tokens ({compressed_size/1024:.1f} KB)")
                print()

            except Exception as e:
                logger.exception("Agent error")
                print(f"\n  Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="GhostKV — memory that haunts, not repeats",
    )
    parser.add_argument("--model", default="C:/Users/aaron/qwen3_exp/model",
                        help="Path to model (default: Qwen3-4B)")
    parser.add_argument("--session", default="default",
                        help="Session name (default: 'default')")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Disable 4-bit quantization")
    parser.add_argument("--no-search", action="store_true",
                        help="Disable web search tool")
    parser.add_argument("--max-tokens", type=int, default=120,
                        help="Max new tokens per generation step")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Max agent ReAct steps per question")
    parser.add_argument("--kv-bits", type=int, default=3,
                        help="KV compression bits (default: 3)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8, 0=greedy)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling (default: 50, 0=disabled)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling threshold (default: 0.9, 1.0=disabled)")
    parser.add_argument("--symmetric-ttq", action="store_true",
                        help="Enable symmetric TTQ (compress Q/K/V in same rotation frame)")
    parser.add_argument("--query-bits", type=int, default=4,
                        help="Query quantization bits for symmetric TTQ (default: 4)")
    parser.add_argument("--code-timeout", type=int, default=30,
                        help="Code execution timeout in seconds")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(name)s: %(message)s")

    agent = build_agent(args)
    repl(agent)


if __name__ == "__main__":
    main()
