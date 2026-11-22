"""Host-side MCP client for the voice gateway MCP server.

Connects the agent tool loop and UI to voice_mcp_server over stdio JSON-RPC.
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
try:
    from mcp_app.mcp_host.incubator_client import call_tool_result_to_dict
except ModuleNotFoundError:
    from mcp_host.incubator_client import call_tool_result_to_dict


@dataclass(frozen=True)
class VoiceServerConfig:
    """Settings for spawning the voice gateway MCP server subprocess.

    Attributes:
        python_executable: Python binary used to run the server module.
        server_module: Import path of the voice server (default: mcp_app.mcp_servers.voice_mcp_server).
    """

    python_executable: str = sys.executable
    server_module: str = "mcp_app.mcp_servers.voice_mcp_server"


class VoiceMcpClient:
    """Client for invoking voice tools on voice_mcp_server via FastMCP stdio."""

    def __init__(self, config: VoiceServerConfig | None = None) -> None:
        """Initialize the client with configuration.

        Args:
            config: Server spawning parameters.
        """
        self.config = config or VoiceServerConfig()
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "VoiceMcpClient":
        """Start long-lived stdio session to the voice gateway server."""
        self._exit_stack = AsyncExitStack()
        params = StdioServerParameters(
            command=self.config.python_executable,
            args=["-m", self.config.server_module],
            env=None,
        )
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._session = session
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close stdio session and terminate server process."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call a tool on the voice gateway server.

        Args:
            tool_name: Registered tool name (e.g. voice_run_e2e_pipeline).
            arguments: Tool arguments mapping.

        Returns:
            Normalized dictionary payload with 'ok' and result fields.
        """
        args = arguments or {}

        # If long-lived session active, reuse it
        if self._session is not None:
            raw_result = await self._session.call_tool(tool_name, arguments=args)
            return call_tool_result_to_dict(raw_result)

        # Otherwise, spawn per-call session
        params = StdioServerParameters(
            command=self.config.python_executable,
            args=["-m", self.config.server_module],
            env=None,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                raw_result = await session.call_tool(tool_name, arguments=args)
                return call_tool_result_to_dict(raw_result)
