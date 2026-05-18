"""Tests for GhostKV MCP client support.

Tests MCPTool, MCPClient (stdio), and integration with ToolDispatch.
All tests use mocks — no real MCP servers or subprocesses needed.
"""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ===================================================================
# MCPTool tests
# ===================================================================

class TestMCPTool:
    """Test MCPTool.run() and content formatting."""

    def test_mcp_tool_run_returns_text(self):
        """Tool.run() concatenates text content blocks."""
        from ghostkv.mcp import MCPTool, MCPClient

        client = MagicMock(spec=MCPClient)
        client.call_tool.return_value = {
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
            "isError": False,
        }

        tool = MCPTool("test_tool", "A test tool", {}, client)
        result = tool.run(query="test")
        assert result == "Hello\nWorld"

    def test_mcp_tool_error_response(self):
        """Tool.run() returns error string when isError=True."""
        from ghostkv.mcp import MCPTool, MCPClient

        client = MagicMock(spec=MCPClient)
        client.call_tool.return_value = {
            "content": [{"type": "text", "text": "Something went wrong"}],
            "isError": True,
        }

        tool = MCPTool("failing_tool", "Fails", {}, client)
        result = tool.run()
        assert "MCP tool error" in result
        assert "Something went wrong" in result

    def test_mcp_tool_empty_content(self):
        """Tool.run() returns 'No output' when no text blocks present."""
        from ghostkv.mcp import MCPTool, MCPClient

        client = MagicMock(spec=MCPClient)
        client.call_tool.return_value = {
            "content": [{"type": "image", "data": "..."}],
            "isError": False,
        }

        tool = MCPTool("img_tool", "Returns images", {}, client)
        result = tool.run()
        assert result == "No output"

    def test_mcp_tool_params_property(self):
        """Tool.params returns formatted parameter signature."""
        from ghostkv.mcp import MCPTool, MCPClient

        client = MagicMock(spec=MCPClient)
        schema = {
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            }
        }
        tool = MCPTool("search", "Search stuff", schema, client)
        assert tool.params == "query: string, limit: integer"

    def test_mcp_tool_params_empty_schema(self):
        """Tool.params returns empty string for no parameters."""
        from ghostkv.mcp import MCPTool, MCPClient

        client = MagicMock(spec=MCPClient)
        tool = MCPTool("noop", "Does nothing", {}, client)
        assert tool.params == ""


# ===================================================================
# MCPClient tests (stdio transport)
# ===================================================================

class TestMCPClient:
    """Test MCPClient stdio transport with mocked subprocess."""

    def test_stdio_connect_handshake(self):
        """Connect sends initialize request and initialized notification."""
        from ghostkv.mcp import MCPClient

        # Mock subprocess with stdin/stdout pipes
        mock_proc = MagicMock()
        init_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-server", "version": "1.0"},
            },
        }).encode() + b"\n"

        mock_proc.stdout.readline.return_value = init_response
        mock_proc.stderr = MagicMock()

        with patch("ghostkv.mcp.subprocess.Popen", return_value=mock_proc):
            client = MCPClient(transport="stdio", command="python", args=["server.py"])
            client.connect()

        # Verify initialize was written to stdin
        calls = mock_proc.stdin.write.call_args_list
        assert len(calls) == 2  # initialize request + initialized notification

        init_msg = json.loads(calls[0][0][0].decode())
        assert init_msg["method"] == "initialize"
        assert init_msg["params"]["protocolVersion"] == "2024-11-05"

        notif_msg = json.loads(calls[1][0][0].decode())
        assert notif_msg["method"] == "notifications/initialized"
        assert "id" not in notif_msg  # notifications have no id

    def test_discover_tools(self):
        """discover_tools() calls tools/list and creates MCPTool instances."""
        from ghostkv.mcp import MCPClient

        mock_proc = MagicMock()

        # First readline: initialize response
        init_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}).encode() + b"\n"
        # Second readline: tools/list response
        tools_resp = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "tools": [
                    {
                        "name": "search",
                        "description": "Search the web",
                        "inputSchema": {
                            "properties": {"query": {"type": "string"}},
                        },
                    },
                    {
                        "name": "read_file",
                        "description": "Read a file",
                        "inputSchema": {},
                    },
                ]
            },
        }).encode() + b"\n"

        mock_proc.stdout.readline.side_effect = [init_resp, tools_resp]
        mock_proc.stderr = MagicMock()

        with patch("ghostkv.mcp.subprocess.Popen", return_value=mock_proc):
            client = MCPClient(transport="stdio", command="python", args=["server.py"])
            client.connect()
            tools = client.discover_tools()

        assert len(tools) == 2
        assert tools[0]["name"] == "search"
        assert tools[0]["description"] == "Search the web"
        assert tools[0]["params"] == "query: string"
        assert tools[1]["name"] == "read_file"

        # Verify MCPTool instances stored in client
        assert "search" in client._tools
        assert "read_file" in client._tools

    def test_call_tool(self):
        """call_tool() sends tools/call and returns result."""
        from ghostkv.mcp import MCPClient

        mock_proc = MagicMock()

        init_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}).encode() + b"\n"
        call_resp = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [{"type": "text", "text": "Search results here"}],
                "isError": False,
            },
        }).encode() + b"\n"

        mock_proc.stdout.readline.side_effect = [init_resp, call_resp]
        mock_proc.stderr = MagicMock()

        with patch("ghostkv.mcp.subprocess.Popen", return_value=mock_proc):
            client = MCPClient(transport="stdio", command="python", args=["server.py"])
            client.connect()
            result = client.call_tool("search", {"query": "hello"})

        assert result["isError"] is False
        assert result["content"][0]["text"] == "Search results here"

    def test_disconnect_cleans_up(self):
        """disconnect() terminates the subprocess."""
        from ghostkv.mcp import MCPClient

        mock_proc = MagicMock()
        init_resp = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}).encode() + b"\n"
        mock_proc.stdout.readline.return_value = init_resp
        mock_proc.stderr = MagicMock()

        with patch("ghostkv.mcp.subprocess.Popen", return_value=mock_proc):
            client = MCPClient(transport="stdio", command="python")
            client.connect()
            client.disconnect()

        mock_proc.stdin.close.assert_called_once()
        mock_proc.terminate.assert_called_once()


# ===================================================================
# MCPClientManager tests
# ===================================================================

class TestMCPClientManager:
    """Test MCPClientManager multi-server management."""

    def test_parse_spec_sse(self):
        """parse_spec handles sse: URLs."""
        from ghostkv.mcp import MCPClientManager

        client = MCPClientManager.parse_spec("sse:http://localhost:8003/sse")
        assert client._transport == "sse"
        assert client._url == "http://localhost:8003/sse"

    def test_parse_spec_stdio(self):
        """parse_spec handles stdio: commands."""
        from ghostkv.mcp import MCPClientManager

        client = MCPClientManager.parse_spec("stdio:python", args=["server.py"])
        assert client._transport == "stdio"
        assert client._command == "python"
        assert client._args == ["server.py"]

    def test_parse_spec_bare_defaults_to_stdio(self):
        """parse_spec defaults bare spec to stdio."""
        from ghostkv.mcp import MCPClientManager

        client = MCPClientManager.parse_spec("python")
        assert client._transport == "stdio"
        assert client._command == "python"


# ===================================================================
# Integration tests
# ===================================================================

class TestMCPIntegration:
    """Test MCP tools integrated into ToolDispatch."""

    def test_mcp_tools_registered_in_dispatch(self):
        """MCPClientManager tools register in ToolDispatch."""
        from ghostkv.agent import ToolDispatch
        from ghostkv.mcp import MCPTool, MCPClient

        tools = ToolDispatch()
        client = MagicMock(spec=MCPClient)

        mcp_tool = MCPTool("web_search", "Search the web", {}, client)
        tools.register_tool("web_search", mcp_tool)

        assert "web_search" in tools.available_tools()

    def test_mcp_tool_dispatch_via_action(self):
        """Action: web_search("test") dispatched to MCPTool."""
        from ghostkv.agent import ToolDispatch
        from ghostkv.mcp import MCPTool, MCPClient

        client = MagicMock(spec=MCPClient)
        client.call_tool.return_value = {
            "content": [{"type": "text", "text": "Found: test results"}],
            "isError": False,
        }

        mcp_tool = MCPTool("web_search", "Search the web", {}, client)
        tools = ToolDispatch()
        tools.register_tool("web_search", mcp_tool)

        name, result = tools.dispatch('Action: web_search("hello world")')
        assert name == "web_search"
        assert "Found: test results" in result

    def test_mcp_tools_in_system_prompt(self):
        """MCP tools appear in rendered system template."""
        from ghostkv.agent import ToolDispatch, GhostKVAgent
        from ghostkv.mcp import MCPTool, MCPClient
        from ghostkv.kv import KVSession
        from ghostkv.remote import MessageSession

        client = MagicMock(spec=MCPClient)
        mcp_tool = MCPTool(
            "web_reader",
            "Read web pages",
            {"properties": {"url": {"type": "string"}}},
            client,
        )

        tools = ToolDispatch()
        tools.register_tool("web_reader", mcp_tool)

        session = MessageSession(name="test", model_name="test")
        agent = GhostKVAgent(session=session, tools=tools, memory=MagicMock())

        prompt = agent._build_initial_prompt("test question")
        assert "web_reader" in prompt
        assert "url: string" in prompt
        assert "Additional Tools (MCP)" in prompt
