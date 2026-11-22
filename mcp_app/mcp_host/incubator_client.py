"""Host-side MCP client for the incubator MCP server.

Each tool call opens a fresh stdio MCP session. The server keeps one shared
hardware connection inside its process.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, ContentBlock


@dataclass(frozen=True)
class IncubatorServerConfig:
    """Settings for spawning the incubator MCP server subprocess.

    Attributes:
        python_executable: Python binary used to run the server module.
        server_module: Import path of the MCP server (default incubator server).
    """

    python_executable: str
    server_module: str = "mcp_servers.incubator_mcp_server"


def _text_from_content(content: list[ContentBlock]) -> str:
    """Join MCP text content blocks into one string.

    Args:
        content: Content blocks from CallToolResult.content.

    Returns:
        Combined text, or empty string if no text blocks.
    """
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


def call_tool_result_to_dict(result: CallToolResult) -> dict[str, Any]:
    """Turn an MCP CallToolResult into a plain dict the model can read.

    Args:
        result: Raw MCP response from session.call_tool.

    Returns:
        Dict with ok flag; failures include error when possible.
    """
    text = _text_from_content(result.content)
    out: dict[str, Any]

    if result.structuredContent is not None:
        out = dict(result.structuredContent)
    elif text:
        try:
            parsed = json.loads(text)
            out = dict(parsed) if isinstance(parsed, dict) else {"message": text}
        except json.JSONDecodeError:
            out = {"message": text}
    else:
        out = {}

    if result.isError:
        out["ok"] = False
        if "error" not in out:
            out["error"] = text or "MCP tool call failed (no details from server)."
    elif "ok" not in out:
        out["ok"] = True

    return out


class IncubatorMcpClient:
    """Async client that lists and calls tools on the incubator MCP server."""

    def __init__(self, config: IncubatorServerConfig) -> None:
        """Store config used to spawn the MCP server on each call.

        Args:
            config: Python executable and server module path.
        """
        self._config = config

    def _server_params(self) -> StdioServerParameters:
        """Build stdio launch parameters for the MCP server module.

        Returns:
            StdioServerParameters passed to stdio_client.
        """
        return StdioServerParameters(
            command=self._config.python_executable,
            args=["-m", self._config.server_module],
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one MCP tool by name.

        Args:
            name: Tool name registered on the incubator server.
            arguments: JSON-serializable tool arguments.

        Returns:
            Plain dict result (ok, error, and tool-specific fields).
        """
        server_params = self._server_params()
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                raw = await session.call_tool(name, arguments)
                return call_tool_result_to_dict(raw)

    async def list_tools(self) -> list[dict[str, Any]]:
        """List tools and input schemas from the incubator MCP server.

        Returns:
            List of dicts with name, description, and input_schema keys.
        """
        server_params = self._server_params()
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                response = await session.list_tools()
                return [
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.inputSchema,
                    }
                    for tool in response.tools
                ]
