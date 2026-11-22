"""CLI entry point for the Ollama-driven MCP tool loop.

Prints trace lines to stdout. Use --mock-tools when no incubator/microscope
hardware is connected.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from mcp_host.mcp_tool_loop import ToolLoopParams, run_tool_loop


async def run(prompt: str, max_steps: int, port: str, mock_tools: bool) -> None:
    """Run one tool loop and print progress to stdout.

    Args:
        prompt: User objective for the model.
        max_steps: Max model turns before stopping.
        port: Serial port passed to incubator tools.
        mock_tools: If True, use fake tool responses (no hardware).

    Returns:
        None. Exits with code 1 if the loop ends with status error.
    """
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
    ollama_command = os.getenv("OLLAMA_COMMAND", "ollama")

    params = ToolLoopParams(
        prompt=prompt,
        max_steps=max_steps,
        port=port,
        mock_tools=mock_tools,
        model_name=model_name,
        ollama_command=ollama_command,
        microscope_xy_port=os.getenv("MICROSCOPE_XY_PORT", "COM3"),
        microscope_kcube_serial=os.getenv("MICROSCOPE_KCUBE_SERIAL", "26004971"),
        microscope_camera_serial=os.getenv("MICROSCOPE_CAMERA_SERIAL", "25267"),
    )

    def trace(msg: str) -> None:
        """Print one harness log line to stdout.

        Args:
            msg: Log line text.

        Returns:
            None.
        """
        print(msg, flush=True)

    result = await run_tool_loop(
        params,
        trace=trace,
        python_executable=sys.executable,
        abort_event=None,
        confirm_tool=None,
    )
    if result.status == "error" and result.error:
        raise SystemExit(1)


def main() -> None:
    """Parse CLI arguments and run the MCP harness.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Test Ollama-driven MCP tool loop (incubator + microscope)."
    )
    parser.add_argument("--prompt", required=True, help="User objective for the model.")
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool-loop steps.")
    parser.add_argument(
        "--port",
        default=os.getenv("INCUBATOR_SERIAL_PORT", "COM1"),
        help="Serial port for incubator tools.",
    )
    parser.add_argument(
        "--mock-tools",
        action="store_true",
        help="Simulate incubator and microscope tools (no real hardware).",
    )
    args = parser.parse_args()
    asyncio.run(
        run(
            prompt=args.prompt,
            max_steps=args.max_steps,
            port=args.port,
            mock_tools=args.mock_tools,
        )
    )


if __name__ == "__main__":
    main()
