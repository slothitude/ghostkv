"""GhostKV Agent — ReAct loop with KV-state management.

Core loop: Think → Act (tool call) → Observe → Repeat
The agent's memory IS the KV cache. History lives in KV state, not re-fed tokens.

Tool dispatch parses model output for Action: tool_name(args) patterns.
After each step, observations are written to the Obsidian vault.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.nn import functional as F
from jinja2 import Environment, FileSystemLoader

from ghostkv.kv import KVSession
from ghostkv.model import ModelBackend
from ghostkv.remote import RemoteBackend, MessageSession
from ghostkv.tools.memory import MemoryTool
from ghostkv.tools.search import SearchTool
from ghostkv.tools.code import CodeTool
from ghostkv.tools.bash import BashTool
from ghostkv.tools.files import FileReadTool, FileWriteTool
from ghostkv.tools.http import HttpTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool history tracking (for hybrid escalation)
# ---------------------------------------------------------------------------

@dataclass
class ToolHistoryEntry:
    """Record of a single tool call during a ReAct loop."""
    step: int
    thought: str
    tool_name: str
    tool_args: str
    result: str


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

# Regex for parsing tool calls from model output
# Matches: Action: tool_name("arg1") or Action: tool_name("arg1", "arg2")
ACTION_RE = re.compile(
    r'Action:\s*(\w+)\(\s*"((?:[^"\\]|\\.)*)"\s*(?:,\s*"((?:[^"\\]|\\.)*)")?\s*\)',
    re.IGNORECASE,
)


class ToolDispatch:
    """Dispatches tool calls from the agent to the right tool implementation."""

    def __init__(
        self,
        search: SearchTool | None = None,
        code: CodeTool | None = None,
        bash: BashTool | None = None,
        file_read: FileReadTool | None = None,
        file_write: FileWriteTool | None = None,
        http: HttpTool | None = None,
        memory: MemoryTool | None = None,
    ):
        self._tools: dict[str, object] = {}
        if search:
            self._tools["search"] = search
        if code:
            self._tools["run"] = code
        if bash:
            self._tools["bash"] = bash
        if file_read:
            self._tools["read"] = file_read
        if file_write:
            self._tools["write"] = file_write
        if http:
            self._tools["http"] = http
        if memory:
            self._tools["recall"] = memory

    def available_tools(self) -> list[str]:
        return list(self._tools.keys())

    def dispatch(self, action_str: str) -> tuple[str, Optional[str]]:
        """Parse an Action: line and execute the tool.

        Args:
            action_str: Raw action string like 'Action: search("Eiffel Tower")'

        Returns:
            (tool_name, result_string) or (tool_name, None) if not found
        """
        m = ACTION_RE.search(action_str)
        if not m:
            return ("unknown", f"Invalid action format: {action_str}")

        tool_name = m.group(1).lower()
        arg1 = m.group(2).replace('\\"', '"').replace("\\n", "\n") if m.group(2) else ""
        arg2 = m.group(3).replace('\\"', '"').replace("\\n", "\n") if m.group(3) else None

        tool = self._tools.get(tool_name)
        if tool is None:
            return (tool_name, f"Unknown tool: {tool_name}")

        try:
            if tool_name == "search":
                result = tool.run(arg1)
            elif tool_name == "run":
                result = tool.run(arg1)
            elif tool_name == "bash":
                result = tool.run(arg1)
            elif tool_name == "read":
                result = tool.run(arg1)
            elif tool_name == "write":
                result = tool.run(arg1, arg2 or "")
            elif tool_name == "http":
                method = arg1.upper() if arg1 else "GET"
                url = arg2 or ""
                result = tool.run(method=method, url=arg1)
            elif tool_name == "recall":
                result = tool.run(arg1)
            else:
                result = f"Tool '{tool_name}' not implemented"
        except Exception as e:
            result = f"Tool error ({tool_name}): {e}"

        return (tool_name, result)


# ---------------------------------------------------------------------------
# Sampling — temperature, top-k, top-p (nucleus)
# ---------------------------------------------------------------------------

def sample_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Sample next token with temperature, top-k, and nucleus (top-p) filtering.

    Args:
        logits: (batch, vocab_size) raw logits for next position
        temperature: Sampling temperature (>0). Lower = more deterministic.
        top_k: Keep only top-k highest probability tokens. 0 = disabled.
        top_p: Keep smallest set of tokens with cumulative prob >= top_p. 1.0 = disabled.

    Returns:
        (batch, 1) sampled token IDs
    """
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, k, dim=-1)
        threshold = topk_vals[:, -1:]
        logits = logits.masked_fill(logits < threshold, float('-inf'))

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Mask tokens above cumulative threshold (keep first that exceeds)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        # Scatter back to original ordering
        mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(mask, float('-inf'))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# Manual autoregressive generation (from Stage 7a)
# ---------------------------------------------------------------------------

def generate_step(
    model: ModelBackend,
    input_ids: torch.Tensor,
    past_kv=None,
    max_new: int = 120,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> tuple[list[int], object]:
    """Generate tokens one at a time using model.forward().

    Returns: (generated_token_ids, past_key_values)
    """
    generated = []
    current_ids = input_ids
    pkv = past_kv

    for _ in range(max_new):
        logits, pkv = model.forward(current_ids, past_kv=pkv, use_cache=True)
        next_logits = logits[:, -1, :]
        next_token = sample_token(next_logits, temperature, top_k, top_p)
        tid = next_token.item()
        generated.append(tid)
        current_ids = next_token
        if tid == model.eos_token_id:
            break

    return generated, pkv


# ---------------------------------------------------------------------------
# GhostKV Agent
# ---------------------------------------------------------------------------

class GhostKVAgent:
    """ReAct agent with KV-state persistence.

    The agent loop:
    1. Build prompt from system template + recalled memories + user question
    2. Generate response using model.forward() with past KV cache
    3. Parse response for Action: tool calls
    4. Execute tool, format observation
    5. Feed observation back to model (only new tokens — KV has the history)
    6. Repeat until no more tool calls (final answer)
    7. Write observations to memory vault
    8. Update session stats
    """

    def __init__(
        self,
        model: ModelBackend | None = None,
        session: KVSession | MessageSession | None = None,
        tools: ToolDispatch | None = None,
        memory: MemoryTool | None = None,
        remote_backend: RemoteBackend | None = None,
        max_new_tokens: int = 120,
        max_steps: int = 10,
        auto_save_steps: int = 5,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ):
        self.model = model
        self.session = session
        self.tools = tools
        self.memory = memory
        self.remote_backend = remote_backend
        self.max_new_tokens = max_new_tokens
        self._tool_history: list[ToolHistoryEntry] = []

        # Determine operating mode
        if model is not None and remote_backend is not None:
            self._mode = "hybrid"
        elif remote_backend is not None:
            self._mode = "remote"
        else:
            self._mode = "local"
        self.max_steps = max_steps
        self.auto_save_steps = auto_save_steps
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # Load templates
        template_dir = Path(__file__).parent / "templates"
        self.jinja = Environment(loader=FileSystemLoader(str(template_dir)))
        self.system_template = self.jinja.get_template("system.j2")
        self.observe_template = self.jinja.get_template("observe.j2")
        self.error_template = self.jinja.get_template("error.j2")
        self.memory_template = self.jinja.get_template("memory.j2")

        # Ensure session rotation is initialized (local mode only)
        if self.model is not None and hasattr(self.session, 'ensure_rotation'):
            self.session.ensure_rotation(self.model.device)

    def _build_initial_prompt(self, question: str) -> str:
        """Build the first prompt with system instructions + question."""
        system = self.system_template.render(
            tools=", ".join(self.tools.available_tools())
        )
        return f"{system}\n\nQuestion: {question}\n"

    def _generate_response(self, prompt_text: str, is_observation: bool = False) -> tuple[str, int]:
        """Generate a response, branching on local vs remote backend.

        Args:
            prompt_text: Text to feed to the model (new tokens for local, message for remote).
            is_observation: If True, prompt_text is an observation (not the initial question).

        Returns:
            (response_text, n_tokens_used)
        """
        if self._mode == "remote":
            # Remote path — build message history
            if is_observation:
                self.session.add_message("assistant", "Using tool...")
                self.session.add_message("user", prompt_text)
            else:
                # Initial question — add system + user messages
                self.session.add_message("user", prompt_text)

            text, usage = self.remote_backend.generate(
                messages=self.session.messages,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
            n_tokens = usage.get("total_tokens", len(text.split()))

            # Track assistant response in message history
            self.session.add_message("assistant", text)
            self.session.token_costs.append(n_tokens)
            self.session.total_tokens += n_tokens
            self.session.steps += 1

            return text, n_tokens
        else:
            # Local path — tokenize + generate_step + decode
            input_ids = self.model.tokenize(prompt_text)
            n_tokens = input_ids.shape[1]
            self.session.token_costs.append(n_tokens)
            self.session.total_tokens += n_tokens

            past_kv = self.session.cache if hasattr(self.session, 'cache') else None
            gen_ids, new_kv = generate_step(
                self.model, input_ids,
                past_kv=past_kv,
                max_new=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            response = self.model.decode(gen_ids)

            if hasattr(self.session, 'cache'):
                self.session.cache = new_kv
            self.session.steps += 1

            return response, n_tokens

    def run(self, question: str) -> str:
        """Run the full agent loop on a question.

        Returns the final answer text.
        """
        # Step 0: Recall relevant memories
        memories_raw = self._recall_memories(question)

        # Build initial prompt
        if memories_raw:
            memory_block = self.memory_template.render(memories=memories_raw)
            prompt = self._build_initial_prompt(question)
            prompt = memory_block + "\n" + prompt
        else:
            prompt = self._build_initial_prompt(question)

        self.session.log(f"\n{'='*60}")
        self.session.log(f"Q: {question}")
        self.session.log(f"{'='*60}")

        # Generate first response
        response, n_tokens = self._generate_response(prompt, is_observation=False)

        self.session.log(f"Step 1 ({n_tokens} tokens): {response[:200]}")

        # Check for tool calls and loop
        answer = self._react_loop(response, max_steps=self.max_steps - 1)

        # Hybrid escalation: synthesize final answer via remote
        if self._mode == "hybrid":
            try:
                remote_answer, remote_tokens = self._escalate_to_remote(
                    question, self._tool_history, answer
                )
                if hasattr(self.session, 'remote_tokens'):
                    self.session.remote_tokens += remote_tokens
                    self.session.remote_calls += 1
                self._ingest_into_kv(remote_answer)
                answer = remote_answer
            except Exception:
                logger.warning("Remote escalation failed, using local draft", exc_info=True)

        # Write observations to memory
        self._save_to_memory(question, answer)

        # Auto-save if needed
        if self.session.steps % self.auto_save_steps == 0:
            self.session.save()

        return answer

    def _react_loop(self, initial_response: str, max_steps: int) -> str:
        """Continue the ReAct loop from an initial response.

        Parses tool calls, executes them, feeds observations back.
        """
        self._tool_history.clear()
        current_response = initial_response

        for step in range(max_steps):
            # Check for tool call
            action_match = ACTION_RE.search(current_response)
            if not action_match:
                # No tool call — this is the final answer
                # Strip any "Thought:" prefixes for clean output
                answer = re.sub(r'^Thought:\s*', '', current_response, flags=re.MULTILINE)
                answer = answer.strip()
                self.session.log(f"Answer: {answer[:200]}")
                return answer

            # Execute tool
            tool_name, result = self.tools.dispatch(current_response)

            if result is None:
                answer = current_response.strip()
                self.session.log(f"Answer: {answer[:200]}")
                return answer

            # Track in tool history
            tool_args = action_match.group(2) if action_match.group(2) else ""
            self._tool_history.append(ToolHistoryEntry(
                step=step + 1,
                thought=current_response[:200],
                tool_name=tool_name,
                tool_args=tool_args,
                result=(result or "")[:200],
            ))

            self.session.log(f"  Tool: {tool_name}")
            self.session.log(f"  Result: {result[:200]}")

            # Format observation
            observation = self.observe_template.render(result=result[:2000])

            # Generate next response from observation
            current_response, n_tokens = self._generate_response(
                observation, is_observation=True
            )
            self.session.log(f"Step {self.session.steps} ({n_tokens} tokens): {current_response[:200]}")

        return current_response.strip()

    def _build_synthesis_prompt(
        self,
        question: str,
        tool_history: list[ToolHistoryEntry],
        local_draft: str,
    ) -> list[dict]:
        """Build message list for remote synthesis call."""
        research_steps = []
        for entry in tool_history:
            research_steps.append(
                f"Step {entry.step}: Used {entry.tool_name}({entry.tool_args})\n"
                f"Result: {entry.result[:1000]}"
            )
        research_block = "\n\n".join(research_steps) if research_steps else "No tools used."

        user_content = (
            f"## Original Question\n{question}\n\n"
            f"## Research Steps\n{research_block}\n\n"
            f"## Local Draft Answer\n{local_draft}\n\n"
            f"## Instructions\n"
            f"Synthesize a polished, accurate final answer from the research above. "
            f"Use all available information from the research steps. "
            f"Be concise and direct."
        )
        return [
            {"role": "system", "content": "You are a helpful research assistant. Synthesize a final answer from the provided research."},
            {"role": "user", "content": user_content},
        ]

    def _escalate_to_remote(
        self,
        question: str,
        tool_history: list[ToolHistoryEntry],
        local_draft: str,
    ) -> tuple[str, int]:
        """Send synthesis request to remote backend.

        Returns (answer, remote_tokens). Falls back to local_draft on failure.
        """
        messages = self._build_synthesis_prompt(question, tool_history, local_draft)
        text, usage = self.remote_backend.generate(
            messages=messages,
            temperature=0.5,
            max_tokens=self.max_new_tokens * 2,
        )
        remote_tokens = usage.get("total_tokens", len(text.split()))
        return text, remote_tokens

    def _ingest_into_kv(self, text: str) -> None:
        """Feed remote answer into local KV cache for context continuity.

        Runs a minimal generate_step (temp=0, max_new=1) purely for the
        KV side effect — the model absorbs the text into cache.
        """
        if self.model is None or not hasattr(self.session, 'cache'):
            return
        ingest_text = f"Synthesized Answer: {text}"
        input_ids = self.model.tokenize(ingest_text)
        n_tokens = input_ids.shape[1]
        self.session.token_costs.append(n_tokens)
        self.session.total_tokens += n_tokens

        past_kv = self.session.cache
        _, new_kv = generate_step(
            self.model, input_ids,
            past_kv=past_kv,
            max_new=1,
            temperature=0,
        )
        self.session.cache = new_kv

    def _recall_memories(self, question: str) -> list[str]:
        """Search the vault for memories relevant to the question."""
        try:
            result = self.memory.run(question, top_k=3)
            if "No memories found" in result:
                return []
            return [result]
        except Exception:
            return []

    def _save_to_memory(self, question: str, answer: str):
        """Save the Q&A as a memory observation."""
        try:
            # Extract key entities for tags
            tags = self._extract_tags(question + " " + answer)
            content = f"**Q:** {question}\n\n**A:** {answer}"
            self.memory.write(
                title=question[:60],
                content=content,
                memory_type="observation",
                tags=tags,
            )
        except Exception as e:
            logger.warning(f"Failed to save memory: {e}")

    def _extract_tags(self, text: str) -> list[str]:
        """Extract meaningful tags from text (simple keyword extraction)."""
        # Remove common stop words, keep words > 4 chars
        stop = {"about", "where", "which", "there", "their", "would", "could",
                "should", "what", "when", "how", "that", "this", "with",
                "from", "they", "been", "have", "will", "each", "does"}
        words = re.findall(r'\b[a-z]{5,}\b', text.lower())
        tags = list({w for w in words if w not in stop})[:10]
        return tags
