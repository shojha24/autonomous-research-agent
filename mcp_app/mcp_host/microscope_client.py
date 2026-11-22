"""Host-side MCP client for the microscope MCP server.

Unlike a spawn-per-call client, this keeps one stdio MCP server process open
for the duration of an agent run. That matters because microscope hardware
(especially the XY START handshake) is slow to reconnect.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_host.incubator_client import call_tool_result_to_dict


@dataclass(frozen=True)
class MicroscopeServerConfig:
    """Settings for spawning the microscope MCP server subprocess.

    Attributes:
        python_executable: Python binary used to run the server module.
        server_module: Import path of the MCP server.
    """

    python_executable: str
    server_module: str = "mcp_servers.microscope_mcp_server"


class MicroscopeMcpClient:
    """Async client with a long-lived stdio session for microscope tools."""

    def __init__(self, config: MicroscopeServerConfig) -> None:
        """Store config; call connect() before list_tools / call_tool.

        Args:
            config: Python executable and server module path.
        """
        self._config = config
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    def _server_params(self) -> StdioServerParameters:
        """Build stdio launch parameters for the MCP server module.

        Returns:
            StdioServerParameters passed to stdio_client.
        """
        return StdioServerParameters(
            command=self._config.python_executable,
            args=["-m", self._config.server_module],
        )

    async def connect(self) -> None:
        """Start the MCP server subprocess and initialize one session.

        Returns:
            None.

        Raises:
            RuntimeError: If already connected.
        """
        if self._session is not None:
            raise RuntimeError("MicroscopeMcpClient is already connected.")
        stack = AsyncExitStack()
        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(self._server_params())
        )
        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        self._stack = stack
        self._session = session

    async def close(self) -> None:
        """Close the MCP session and terminate the server subprocess.

        Returns:
            None.
        """
        self._session = None
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None

    def _require_session(self) -> ClientSession:
        """Return the open session or raise if connect() was not called.

        Returns:
            Active ClientSession.

        Raises:
            RuntimeError: If the client is not connected.
        """
        if self._session is None:
            raise RuntimeError(
                "MicroscopeMcpClient is not connected; call connect() first."
            )
        return self._session

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one MCP tool by name on the open microscope session.

        Args:
            name: Tool name registered on the microscope server.
            arguments: JSON-serializable tool arguments.

        Returns:
            Plain dict result (ok, error, and tool-specific fields).
        """
        session = self._require_session()
        raw = await session.call_tool(name, arguments)
        return call_tool_result_to_dict(raw)

    async def list_tools(self) -> list[dict[str, Any]]:
        """List tools and input schemas from the open microscope session.

        Returns:
            List of dicts with name, description, and input_schema keys.
        """
        session = self._require_session()
        response = await session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]
