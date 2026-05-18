"""GhostKV MCP Client — Model Context Protocol tool discovery and execution.

Supports stdio (subprocess) and SSE (HTTP) transports for connecting to
any MCP-compatible tool server. Pure synchronous — no asyncio needed.

Protocol: JSON-RPC 2.0
- Handshake: initialize → initialized
- Discovery: tools/list → array of {name, description, inputSchema}
- Execution: tools/call with {name, arguments} → {content, isError}
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from typing import Any

import requests

logger = logging.getLogger(__name__)

# JSON-RPC 2.0 helpers

def _make_request(method: str, params: dict | None = None, req_id: int | None = None) -> dict:
    """Build a JSON-RPC 2.0 request dict."""
    msg: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        msg["params"] = params
    if req_id is not None:
        msg["id"] = req_id
    return msg


def _make_notification(method: str, params: dict | None = None) -> dict:
    """Build a JSON-RPC 2.0 notification (no id field)."""
    msg: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        msg["params"] = params
    return msg


# ---------------------------------------------------------------------------
# MCPTool — wraps a single MCP tool as a callable
# ---------------------------------------------------------------------------

class MCPTool:
    """Wraps a single MCP-discovered tool with a run() interface."""

    def __init__(self, name: str, description: str, input_schema: dict, client: MCPClient):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self._client = client

    @property
    def params(self) -> str:
        """Return a human-readable parameter signature string."""
        props = self.input_schema.get("properties", {})
        if not props:
            return ""
        parts = []
        for pname, pdef in props.items():
            ptype = pdef.get("type", "any")
            parts.append(f"{pname}: {ptype}")
        return ", ".join(parts)

    def run(self, **kwargs) -> str:
        """Call the MCP tool and return result as string."""
        result = self._client.call_tool(self.name, kwargs)
        if result.get("isError"):
            return f"MCP tool error: {self._format_content(result['content'])}"
        return self._format_content(result["content"])

    @staticmethod
    def _format_content(content: list[dict]) -> str:
        """Concatenate text blocks from MCP content array."""
        texts = [c["text"] for c in content if c.get("type") == "text"]
        return "\n".join(texts) if texts else "No output"


# ---------------------------------------------------------------------------
# MCPClient — manages connection to a single MCP server
# ---------------------------------------------------------------------------

class MCPClient:
    """Manages a connection to a single MCP server (stdio or SSE transport).

    Usage:
        client = MCPClient(transport="stdio", command="python", args=["server.py"])
        client.connect()
        tools = client.discover_tools()
        result = client.call_tool("search", {"query": "hello"})
        client.disconnect()
    """

    def __init__(
        self,
        transport: str,
        url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
    ):
        self._transport = transport  # "stdio" or "sse"
        self._url = url
        self._command = command
        self._args = args or []
        self._tools: dict[str, MCPTool] = {}
        self._request_id = 0

        # stdio state
        self._process: subprocess.Popen | None = None

        # SSE state
        self._endpoint_url: str | None = None
        self._sse_thread: threading.Thread | None = None
        self._sse_responses: dict[int, dict] = {}
        self._sse_stop = threading.Event()

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    # -- stdio transport --

    def _stdio_send(self, msg: dict) -> None:
        """Write JSON-RPC message to subprocess stdin."""
        assert self._process is not None and self._process.stdin is not None
        data = json.dumps(msg) + "\n"
        self._process.stdin.write(data.encode("utf-8"))
        self._process.stdin.flush()

    def _stdio_recv(self) -> dict:
        """Read a single JSON-RPC response from subprocess stdout."""
        assert self._process is not None and self._process.stdout is not None
        line = self._process.stdout.readline()
        if not line:
            raise ConnectionError("MCP server closed stdout")
        return json.loads(line.decode("utf-8").strip())

    # -- SSE transport --

    def _sse_connect(self) -> None:
        """Connect to SSE endpoint and start background reader thread."""
        sse_url = self._url
        if not sse_url:
            raise ValueError("SSE transport requires a URL")

        resp = requests.get(sse_url, stream=True, timeout=30,
                            headers={"Accept": "text/event-stream"})
        resp.raise_for_status()

        self._sse_thread = threading.Thread(
            target=self._sse_reader, args=(resp,), daemon=True
        )
        self._sse_thread.start()

        # Wait for endpoint URL (usually first event)
        import time
        deadline = time.time() + 10
        while self._endpoint_url is None and time.time() < deadline:
            time.sleep(0.05)
        if self._endpoint_url is None:
            raise ConnectionError("SSE server did not send endpoint URL")

    def _sse_reader(self, resp: requests.Response) -> None:
        """Background thread: read SSE events and match responses."""
        event_type = None
        data_buf = ""

        for raw_line in resp.iter_lines(decode_unicode=True):
            if self._sse_stop.is_set():
                break
            if raw_line is None:
                continue
            line = raw_line.strip()
            if not line:
                # Empty line = end of event
                if event_type == "endpoint" and data_buf:
                    self._endpoint_url = data_buf.strip()
                elif event_type == "message" and data_buf:
                    try:
                        msg = json.loads(data_buf)
                        if "id" in msg:
                            self._sse_responses[msg["id"]] = msg
                    except json.JSONDecodeError:
                        pass
                event_type = None
                data_buf = ""
                continue
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_buf = line[5:].strip()

    def _sse_send(self, msg: dict) -> dict:
        """Send JSON-RPC request via HTTP POST and wait for response."""
        if not self._endpoint_url:
            raise ConnectionError("SSE endpoint not established")

        # POST the request
        resp = requests.post(
            self._endpoint_url,
            json=msg,
            timeout=60,
        )
        # Some SSE servers respond directly in the POST response
        if resp.status_code == 200 and resp.content:
            try:
                return resp.json()
            except json.JSONDecodeError:
                pass

        # Otherwise wait for SSE response matching our request ID
        import time
        req_id = msg.get("id")
        if req_id is None:
            return {}

        deadline = time.time() + 30
        while time.time() < deadline:
            if req_id in self._sse_responses:
                return self._sse_responses.pop(req_id)
            time.sleep(0.05)

        raise TimeoutError(f"SSE response timeout for request {req_id}")

    # -- Common interface --

    def connect(self) -> None:
        """Start transport and perform MCP initialize handshake."""
        if self._transport == "stdio":
            cmd = [self._command] + self._args
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            # Initialize handshake
            init_req = _make_request(
                "initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "ghostkv", "version": "0.5.0"},
                },
                req_id=self._next_id(),
            )
            self._stdio_send(init_req)
            resp = self._stdio_recv()
            if "error" in resp:
                raise ConnectionError(f"MCP initialize failed: {resp['error']}")

            # Send initialized notification
            self._stdio_send(_make_notification("notifications/initialized"))

        elif self._transport == "sse":
            self._sse_connect()
            init_req = _make_request(
                "initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "ghostkv", "version": "0.5.0"},
                },
                req_id=self._next_id(),
            )
            resp = self._sse_send(init_req)
            if "error" in resp:
                raise ConnectionError(f"MCP initialize failed: {resp['error']}")

            # Send initialized notification
            self._sse_send(_make_notification("notifications/initialized"))

        else:
            raise ValueError(f"Unknown transport: {self._transport}")

        logger.info(f"MCP connected ({self._transport}): "
                    f"{self._command or self._url}")

    def discover_tools(self) -> list[dict]:
        """Call tools/list, create MCPTool instances, return tool info dicts."""
        req = _make_request("tools/list", req_id=self._next_id())

        if self._transport == "stdio":
            self._stdio_send(req)
            resp = self._stdio_recv()
        else:
            resp = self._sse_send(req)

        if "error" in resp:
            raise RuntimeError(f"tools/list failed: {resp['error']}")

        tool_list = resp.get("result", {}).get("tools", [])
        results = []
        for t in tool_list:
            name = t["name"]
            description = t.get("description", "")
            schema = t.get("inputSchema", {})
            tool = MCPTool(name, description, schema, self)
            self._tools[name] = tool
            results.append({
                "name": name,
                "description": description,
                "params": tool.params,
                "tool": tool,
            })

        logger.info(f"MCP discovered {len(results)} tools from "
                    f"{self._command or self._url}")
        return results

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Call tools/call and return the result dict."""
        req = _make_request(
            "tools/call",
            params={"name": name, "arguments": arguments},
            req_id=self._next_id(),
        )

        if self._transport == "stdio":
            self._stdio_send(req)
            resp = self._stdio_recv()
        else:
            resp = self._sse_send(req)

        if "error" in resp:
            return {
                "content": [{"type": "text", "text": str(resp["error"])}],
                "isError": True,
            }

        return resp.get("result", {
            "content": [{"type": "text", "text": "Empty response"}],
            "isError": False,
        })

    def disconnect(self) -> None:
        """Clean shutdown of transport."""
        if self._transport == "stdio" and self._process is not None:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None

        elif self._transport == "sse":
            self._sse_stop.set()

        logger.info(f"MCP disconnected ({self._transport})")


# ---------------------------------------------------------------------------
# MCPClientManager — manages multiple MCP server connections
# ---------------------------------------------------------------------------

class MCPClientManager:
    """Manages connections to multiple MCP servers.

    Usage:
        mgr = MCPClientManager()
        mgr.connect("stdio:python", args=["server.py"])
        mgr.connect("sse:http://localhost:8003/sse")
        tools = mgr.discover_all()
        # ... use tools ...
        mgr.disconnect_all()
    """

    def __init__(self):
        self._clients: list[MCPClient] = []

    @staticmethod
    def parse_spec(spec: str, args: list[str] | None = None) -> MCPClient:
        """Parse a server spec string into an MCPClient.

        Formats:
            "stdio:command"        → stdio transport with command
            "sse:url"              → SSE transport with URL
            "command"              → defaults to stdio
        """
        if spec.startswith("sse:"):
            url = spec[4:]
            if not url.startswith("http"):
                url = "http://" + url
            return MCPClient(transport="sse", url=url)
        elif spec.startswith("stdio:"):
            command = spec[6:]
            return MCPClient(transport="stdio", command=command, args=args)
        else:
            # Default to stdio with the spec as command
            return MCPClient(transport="stdio", command=spec, args=args)

    def connect(self, spec: str, args: list[str] | None = None) -> None:
        """Connect to an MCP server by spec string."""
        client = self.parse_spec(spec, args)
        client.connect()
        self._clients.append(client)

    def discover_all(self) -> list[dict]:
        """Discover tools from all connected servers. Returns list of tool info dicts."""
        all_tools = []
        for client in self._clients:
            tools = client.discover_tools()
            all_tools.extend(tools)
        return all_tools

    def disconnect_all(self) -> None:
        """Disconnect all MCP servers."""
        for client in self._clients:
            client.disconnect()
        self._clients.clear()
