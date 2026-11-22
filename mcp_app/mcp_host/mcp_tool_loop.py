"""Shared Ollama + MCP tool loop used by the CLI harness and GUI.

Each step: model picks JSON action -> validate -> run tool or return final answer.
Bad JSON gets fed back so the model can retry.

Supports two MCP backends at once:
  - incubator tools (unprefixed names like initialize, import_plate)
  - microscope tools (prefixed names like microscope_snap)
plus the harness-only read_repo_file tool.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from mcp.types import CallToolResult

from mcp_host.agent_actions import (
    FinalAction,
    ToolCallAction,
    ValidationError,
    parse_agent_action,
)
from mcp_host.harness_tools import (
    default_file_index_path,
    load_allowlisted_paths_from_index,
    load_file_index_excerpt,
    merge_harness_tools,
)
from mcp_host.incubator_client import IncubatorMcpClient, IncubatorServerConfig
from mcp_host.microscope_client import MicroscopeMcpClient, MicroscopeServerConfig
from mcp_host.ollama_client import OllamaClient, OllamaClientError, OllamaConfig
from mcp_host.repo_read import read_repo_file, resolve_repo_root

# Real incubator MCP tool names (used to auto-fill port argument).
INCUBATOR_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "initialize",
        "is_ready",
        "import_plate",
        "export_plate",
        "swap_transfer",
        "open_gate",
        "close_gate",
    }
)

# Real microscope MCP tool names (all prefixed to avoid collisions).
MICROSCOPE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "microscope_initialize",
        "microscope_get_status",
        "microscope_goto_xy",
        "microscope_move_x",
        "microscope_move_y",
        "microscope_move_z",
        "microscope_move_z_relative",
        "microscope_home_x",
        "microscope_home_y",
        "microscope_home_z",
        "microscope_home_all",
        "microscope_snap",
        "microscope_autofocus",
        "microscope_close",
    }
)

# Fake incubator tool list when mock mode is on.
INCUBATOR_MOCK_TOOLS: list[dict[str, Any]] = [
    {
        "name": "initialize",
        "description": "Initialize incubator connection.",
        "input_schema": {
            "type": "object",
            "properties": {"port": {"type": "string"}},
            "required": ["port"],
        },
    },
    {
        "name": "is_ready",
        "description": "Return incubator readiness status.",
        "input_schema": {
            "type": "object",
            "properties": {"port": {"type": "string"}},
            "required": ["port"],
        },
    },
    {
        "name": "import_plate",
        "description": "Import a plate into an incubator rack/slot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "incu_rack": {"type": "integer"},
                "incu_slot": {"type": "integer"},
                "port": {"type": "string"},
            },
            "required": ["incu_rack", "incu_slot", "port"],
        },
    },
    {
        "name": "export_plate",
        "description": "Export a plate from an incubator rack/slot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "incu_rack": {"type": "integer"},
                "incu_slot": {"type": "integer"},
                "port": {"type": "string"},
            },
            "required": ["incu_rack", "incu_slot", "port"],
        },
    },
    {
        "name": "swap_transfer",
        "description": "Rotate the transfer station between gate and handler positions.",
        "input_schema": {
            "type": "object",
            "properties": {"port": {"type": "string"}},
            "required": ["port"],
        },
    },
    {
        "name": "open_gate",
        "description": "Open the incubator gate.",
        "input_schema": {
            "type": "object",
            "properties": {"port": {"type": "string"}},
            "required": ["port"],
        },
    },
    {
        "name": "close_gate",
        "description": "Close the incubator gate.",
        "input_schema": {
            "type": "object",
            "properties": {"port": {"type": "string"}},
            "required": ["port"],
        },
    },
]

# Fake microscope tool list when mock mode is on.
MICROSCOPE_MOCK_TOOLS: list[dict[str, Any]] = [
    {
        "name": "microscope_initialize",
        "description": "Connect to the microscope (Z-stage, camera, XY stage).",
        "input_schema": {
            "type": "object",
            "properties": {
                "kcube_serial": {"type": "string"},
                "camera_serial": {"type": "string"},
                "xy_port": {"type": "string"},
                "xy_baudrate": {"type": "integer"},
            },
            "required": [],
        },
    },
    {
        "name": "microscope_get_status",
        "description": "Return current X/Y/Z positions.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "microscope_goto_xy",
        "description": "Move XY stage to an absolute position (step units).",
        "input_schema": {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
        },
    },
    {
        "name": "microscope_move_x",
        "description": "Relative X move (+right / -left).",
        "input_schema": {
            "type": "object",
            "properties": {"steps": {"type": "integer"}},
            "required": ["steps"],
        },
    },
    {
        "name": "microscope_move_y",
        "description": "Relative Y move (+up / -down).",
        "input_schema": {
            "type": "object",
            "properties": {"steps": {"type": "integer"}},
            "required": ["steps"],
        },
    },
    {
        "name": "microscope_move_z",
        "description": "Move Z to an absolute position in meters.",
        "input_schema": {
            "type": "object",
            "properties": {"meters": {"type": "number"}},
            "required": ["meters"],
        },
    },
    {
        "name": "microscope_move_z_relative",
        "description": "Move Z by a relative delta in meters.",
        "input_schema": {
            "type": "object",
            "properties": {"delta_meters": {"type": "number"}},
            "required": ["delta_meters"],
        },
    },
    {
        "name": "microscope_home_x",
        "description": "Home the X axis.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "microscope_home_y",
        "description": "Home the Y axis.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "microscope_home_z",
        "description": "Home the Z axis.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "microscope_home_all",
        "description": "Home X, Y, and Z.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "microscope_snap",
        "description": "Capture and save one image at the current position.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "microscope_autofocus",
        "description": "Run Tenengrad autofocus and move to optimal Z.",
        "input_schema": {
            "type": "object",
            "properties": {
                "z_start": {"type": "number"},
                "z_end": {"type": "number"},
                "num_steps": {"type": "integer"},
                "exposure": {"type": "number"},
            },
            "required": [],
        },
    },
    {
        "name": "microscope_close",
        "description": "Close microscope hardware connections.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

# Combined mock catalog shown to the model in mock mode.
MOCK_TOOLS: list[dict[str, Any]] = INCUBATOR_MOCK_TOOLS + MICROSCOPE_MOCK_TOOLS

# In-memory XY/Z state for microscope mock responses across one agent run.
_mock_scope_state: dict[str, float | int] = {"x": 0, "y": 0, "z": 0.0}

# Message shown when the model sends invalid JSON.
_RETRY_JSON_HINT = (
    "Your last response was not a valid action object. "
    'Reply with exactly one JSON object: either '
    '{"action":"final","answer":"..."} or '
    '{"action":"tool_call","tool_name":"...","arguments":{...}}.'
)


def build_system_prompt(tools: list[dict[str, Any]], file_index_markdown: str) -> str:
    """Build the system prompt with JSON rules, tool list, and FILE_INDEX.

    Args:
        tools: MCP plus harness tool descriptors shown to the model.
        file_index_markdown: Contents of prompts/FILE_INDEX.md (possibly truncated).

    Returns:
        Full system prompt string for Ollama chat_messages.
    """
    tool_blob = json.dumps(tools, indent=2)
    return (
        "You are a lab automation assistant that can use MCP tools for an "
        "incubator and a microscope when designing or running experiments.\n"
        "You must reply with ONLY one JSON object and nothing else.\n"
        "Choose exactly one action each turn:\n"
        '1) {"action":"tool_call","tool_name":"<name>","arguments":{...}}\n'
        '2) {"action":"final","answer":"<final answer>"}\n'
        "Only use tool names from the provided tool list.\n"
        "Incubator tools are unprefixed (e.g. initialize, import_plate).\n"
        "Microscope tools are prefixed with microscope_ "
        "(e.g. microscope_initialize, microscope_snap).\n"
        "Call microscope_initialize before other microscope tools.\n"
        "Return valid JSON with double quotes.\n\n"
        f"Available tools:\n{tool_blob}\n\n"
        "Repository index (paths you may pass to read_repo_file; only "
        "allowlisted paths work):\n"
        f"{file_index_markdown}\n"
    )


def result_for_history(result: Any) -> Any:
    """Normalize a tool result before adding it to chat history.

    Args:
        result: Raw tool return value (dict, CallToolResult, or other).

    Returns:
        JSON-serializable value for the history message.
    """
    if isinstance(result, CallToolResult):
        return result.structuredContent or {}
    if isinstance(result, dict):
        return result
    return str(result)


def result_for_stdout_log(tool_name: str, result: Any) -> Any:
    """Shorten huge read_repo_file content for console trace logs.

    Args:
        tool_name: Name of the tool that was invoked.
        result: Raw tool return value.

    Returns:
        Same structure as result but with long file content truncated.
    """
    normalized = result_for_history(result)
    if tool_name != "read_repo_file" or not isinstance(normalized, dict):
        return normalized
    logged = dict(normalized)
    content = logged.get("content")
    if logged.get("ok") and isinstance(content, str) and len(content) > 240:
        logged["content"] = (
            f"{content[:240]}... ({len(content)} chars; full text in history message)"
        )
    return logged


def run_mock_incubator_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Return fake incubator tool results when mock_tools is enabled.

    Args:
        tool_name: Incubator tool name the model requested.
        arguments: Tool arguments from the model JSON.

    Returns:
        Dict with ok flag and message or error, matching real tool shape.
    """
    if tool_name == "initialize":
        return {"ok": True, "message": f"[mock] Connected on {arguments.get('port', 'COM1')}"}
    if tool_name == "is_ready":
        return {"ok": True, "ready": True, "message": "[mock] Incubator ready."}
    if tool_name in {"import_plate", "export_plate"}:
        rack = arguments.get("incu_rack")
        slot = arguments.get("incu_slot")
        if not isinstance(rack, int) or not isinstance(slot, int):
            return {"ok": False, "error": "[mock] incu_rack and incu_slot must be integers"}
        if rack not in {1, 2} or not (1 <= slot <= 21):
            return {"ok": False, "error": "[mock] Invalid rack/slot bounds"}
        verb = "Imported" if tool_name == "import_plate" else "Exported"
        return {"ok": True, "message": f"[mock] {verb} plate at rack {rack}, slot {slot}"}
    if tool_name == "swap_transfer":
        return {"ok": True, "message": "[mock] Transfer station swapped."}
    if tool_name == "open_gate":
        return {"ok": True, "message": "[mock] Incubator gate open command sent."}
    if tool_name == "close_gate":
        return {"ok": True, "message": "[mock] Incubator gate close command sent."}
    return {"ok": False, "error": f"[mock] Unknown incubator tool '{tool_name}'"}


def run_mock_microscope_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Return fake microscope tool results when mock_tools is enabled.

    Args:
        tool_name: Microscope tool name the model requested.
        arguments: Tool arguments from the model JSON.

    Returns:
        Dict with ok flag and message or error, matching real tool shape.
    """
    global _mock_scope_state

    if tool_name == "microscope_initialize":
        _mock_scope_state = {"x": 0, "y": 0, "z": 0.0}
        return {
            "ok": True,
            "message": (
                "[mock] Connected microscope "
                f"(kcube={arguments.get('kcube_serial', '26004971')}, "
                f"camera={arguments.get('camera_serial', '25267')}, "
                f"xy_port={arguments.get('xy_port', 'COM3')})."
            ),
        }
    if tool_name == "microscope_get_status":
        return {"ok": True, **_mock_scope_state}
    if tool_name == "microscope_goto_xy":
        x = arguments.get("x")
        y = arguments.get("y")
        if not isinstance(x, int) or not isinstance(y, int):
            return {"ok": False, "error": "[mock] x and y must be integers"}
        _mock_scope_state["x"] = x
        _mock_scope_state["y"] = y
        return {"ok": True, "message": f"[mock] Moved XY to ({x}, {y}).", **_mock_scope_state}
    if tool_name == "microscope_move_x":
        steps = arguments.get("steps")
        if not isinstance(steps, int):
            return {"ok": False, "error": "[mock] steps must be an integer"}
        _mock_scope_state["x"] = int(_mock_scope_state["x"]) + steps
        return {"ok": True, "message": f"[mock] Moved X by {steps}.", **_mock_scope_state}
    if tool_name == "microscope_move_y":
        steps = arguments.get("steps")
        if not isinstance(steps, int):
            return {"ok": False, "error": "[mock] steps must be an integer"}
        _mock_scope_state["y"] = int(_mock_scope_state["y"]) + steps
        return {"ok": True, "message": f"[mock] Moved Y by {steps}.", **_mock_scope_state}
    if tool_name == "microscope_move_z":
        meters = arguments.get("meters")
        if not isinstance(meters, (int, float)):
            return {"ok": False, "error": "[mock] meters must be a number"}
        _mock_scope_state["z"] = float(meters)
        return {"ok": True, "message": f"[mock] Moved Z to {meters}.", **_mock_scope_state}
    if tool_name == "microscope_move_z_relative":
        delta = arguments.get("delta_meters")
        if not isinstance(delta, (int, float)):
            return {"ok": False, "error": "[mock] delta_meters must be a number"}
        _mock_scope_state["z"] = float(_mock_scope_state["z"]) + float(delta)
        return {"ok": True, "message": f"[mock] Moved Z by {delta}.", **_mock_scope_state}
    if tool_name == "microscope_home_x":
        _mock_scope_state["x"] = 0
        return {"ok": True, "message": "[mock] Homed X.", **_mock_scope_state}
    if tool_name == "microscope_home_y":
        _mock_scope_state["y"] = 0
        return {"ok": True, "message": "[mock] Homed Y.", **_mock_scope_state}
    if tool_name == "microscope_home_z":
        _mock_scope_state["z"] = 0.0
        return {"ok": True, "message": "[mock] Homed Z.", **_mock_scope_state}
    if tool_name == "microscope_home_all":
        _mock_scope_state = {"x": 0, "y": 0, "z": 0.0}
        return {"ok": True, "message": "[mock] Homed X/Y/Z.", **_mock_scope_state}
    if tool_name == "microscope_snap":
        return {
            "ok": True,
            "message": "[mock] Image captured.",
            "image_path": "[mock]/mcp_servers/microscope_results/test_imgs/mock.jpg",
            "z": _mock_scope_state["z"],
        }
    if tool_name == "microscope_autofocus":
        optimal = 0.0015
        _mock_scope_state["z"] = optimal
        return {
            "ok": True,
            "message": "[mock] Autofocus complete.",
            "optimal_z": optimal,
            **_mock_scope_state,
        }
    if tool_name == "microscope_close":
        return {"ok": True, "message": "[mock] Microscope connections closed."}
    return {"ok": False, "error": f"[mock] Unknown microscope tool '{tool_name}'"}


def run_mock_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a mock tool call to incubator or microscope mock handlers.

    Args:
        tool_name: Tool name the model requested.
        arguments: Tool arguments from the model JSON.

    Returns:
        Dict with ok flag and message or error.
    """
    if tool_name in MICROSCOPE_TOOL_NAMES:
        return run_mock_microscope_tool(tool_name, arguments)
    if tool_name in INCUBATOR_TOOL_NAMES:
        return run_mock_incubator_tool(tool_name, arguments)
    return {"ok": False, "error": f"[mock] Unknown tool '{tool_name}'"}


def format_confirm_message(tool_name: str, arguments: dict[str, Any]) -> str:
    """Build human-readable text for the GUI tool confirmation dialog.

    Args:
        tool_name: Tool the model requested.
        arguments: Tool arguments dict (may be truncated in the message).

    Returns:
        Multi-line string suitable for messagebox.askyesno.
    """
    if tool_name == "read_repo_file":
        path = arguments.get("path", "?")
        return (
            "The model wants to read a repository file.\n\n"
            f"Path: {path}\n\n"
            "Allow this read? (Only allowlisted paths succeed.)"
        )
    args_preview = json.dumps(arguments, indent=2, ensure_ascii=True)
    if len(args_preview) > 800:
        args_preview = args_preview[:800] + "\n… (truncated)"
    device = "microscope" if tool_name in MICROSCOPE_TOOL_NAMES else "hardware / MCP"
    return (
        f"The model wants to run a {device} tool.\n\n"
        f"Tool: {tool_name}\n\n"
        f"Arguments:\n{args_preview}\n\n"
        "Allow this tool call?"
    )


@dataclass(frozen=True)
class ToolLoopParams:
    """Inputs for one MCP agent run.

    Attributes:
        prompt: User objective passed to the model.
        max_steps: Maximum model turns before forced stop.
        port: Serial port injected into incubator tool args when missing.
        mock_tools: If True, skip real MCP and use MOCK_TOOLS responses.
        model_name: Ollama model tag.
        ollama_command: Ollama CLI name (kept for config parity).
        ollama_host: Ollama HTTP base URL.
        timeout_seconds: HTTP timeout for Ollama requests.
        microscope_xy_port: Default XY serial port for microscope_initialize.
        microscope_kcube_serial: Default K-Cube serial for microscope_initialize.
        microscope_camera_serial: Default camera serial for microscope_initialize.
    """

    prompt: str
    max_steps: int
    port: str
    mock_tools: bool
    model_name: str
    ollama_command: str
    ollama_host: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 300.0
    microscope_xy_port: str = "COM3"
    microscope_kcube_serial: str = "26004971"
    microscope_camera_serial: str = "25267"


@dataclass
class ToolLoopResult:
    """Outcome returned by run_tool_loop.

    Attributes:
        status: completed, max_steps, aborted, or error.
        answer: Final model answer when status is completed.
        error: Error message when status is error.
    """

    status: Literal["completed", "max_steps", "aborted", "error"]
    answer: str = ""
    error: str = ""


def _inject_microscope_defaults(
    tool_name: str,
    arguments: dict[str, Any],
    params: ToolLoopParams,
) -> dict[str, Any]:
    """Fill missing microscope_initialize connection args from params/env defaults.

    Args:
        tool_name: Tool being called.
        arguments: Mutable copy of model arguments.
        params: Loop params holding microscope defaults.

    Returns:
        Arguments dict (same object, possibly updated).
    """
    if tool_name != "microscope_initialize":
        return arguments
    arguments.setdefault("xy_port", params.microscope_xy_port)
    arguments.setdefault("kcube_serial", params.microscope_kcube_serial)
    arguments.setdefault("camera_serial", params.microscope_camera_serial)
    return arguments


async def _execute_tool(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    params: ToolLoopParams,
    incubator_client: IncubatorMcpClient | None,
    microscope_client: MicroscopeMcpClient | None,
    repo_root: Path,
    allowed_paths: frozenset[str],
) -> Any:
    """Dispatch one tool call to read_repo_file, mock, or the right MCP client.

    Args:
        tool_name: Tool name from the model action.
        arguments: Validated tool arguments dict.
        params: Loop params (mock_tools flag, etc.).
        incubator_client: Real incubator MCP client, or None in mock mode.
        microscope_client: Real microscope MCP client, or None in mock mode.
        repo_root: Repository root for read_repo_file.
        allowed_paths: Allowlisted paths for read_repo_file.

    Returns:
        Tool result dict (always includes ok on failure paths).
    """
    if tool_name == "read_repo_file":
        return read_repo_file(
            arguments=arguments,
            repo_root=repo_root,
            allowed_relative_paths=allowed_paths,
        )
    if params.mock_tools:
        return run_mock_tool(tool_name=tool_name, arguments=arguments)
    try:
        if tool_name in MICROSCOPE_TOOL_NAMES:
            assert microscope_client is not None
            return await microscope_client.call_tool(tool_name, arguments)
        assert incubator_client is not None
        return await incubator_client.call_tool(tool_name, arguments)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _append_tool_result_to_history(
    history: list[dict[str, str]],
    model_text: str,
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
) -> None:
    """Append the model action and tool result to chat history.

    Args:
        history: Mutable Ollama message list (modified in place).
        model_text: Raw JSON text the model emitted for this step.
        tool_name: Tool that was invoked.
        arguments: Arguments passed to the tool.
        result: Raw tool return value.

    Returns:
        None.
    """
    tool_result_text = json.dumps(
        {
            "tool": tool_name,
            "arguments": arguments,
            "result": result_for_history(result),
        },
        ensure_ascii=True,
    )
    history.append({"role": "assistant", "content": model_text})
    history.append(
        {"role": "user", "content": f"Tool result: {tool_result_text}\nChoose next action."}
    )


async def run_tool_loop(
    params: ToolLoopParams,
    *,
    trace: Callable[[str], None],
    python_executable: str,
    abort_event: threading.Event | None = None,
    confirm_tool: Callable[[str, dict[str, Any]], bool] | None = None,
) -> ToolLoopResult:
    """Run until final answer, max steps, abort, or error.

    Args:
        params: Prompt, model, port, mock flag, step limit.
        trace: Log sink (print or GUI queue).
        python_executable: Python used to spawn MCP servers.
        abort_event: When set, stop before next model call or tool.
        confirm_tool: Optional per-tool approval (GUI); return False to skip.

    Returns:
        ToolLoopResult with status and optional answer/error.
    """
    incubator_client: IncubatorMcpClient | None = None
    microscope_client: MicroscopeMcpClient | None = None
    if not params.mock_tools:
        incubator_client = IncubatorMcpClient(
            IncubatorServerConfig(python_executable=python_executable)
        )
        microscope_client = MicroscopeMcpClient(
            MicroscopeServerConfig(python_executable=python_executable)
        )

    ollama_client = OllamaClient(
        OllamaConfig(
            command=params.ollama_command,
            model=params.model_name,
            host=params.ollama_host,
            timeout_seconds=params.timeout_seconds,
        )
    )

    # Load repo allowlist once at startup.
    try:
        repo_root = resolve_repo_root()
        index_path = default_file_index_path(repo_root)
        allowed_paths = load_allowlisted_paths_from_index(index_path)
        file_index_md = load_file_index_excerpt(index_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        trace(f"[init error] {exc}")
        return ToolLoopResult(status="error", error=str(exc))

    # Neural Prompt Guard 2 (22M) screening (protects CLI harness, GUI, and API entry points)
    try:
        try:
            from mcp_app.voice.prompt_guard import PromptGuardDetector
        except ModuleNotFoundError:
            from voice.prompt_guard import PromptGuardDetector
        pg = PromptGuardDetector()
        if pg.is_active:
            pg_res = pg.evaluate(params.prompt)
            if not pg_res["is_safe"]:
                trace(
                    f"[security alert] Prompt Guard 2 (22M) flagged prompt as "
                    f"{pg_res['verdict']} (confidence: {pg_res['score']})."
                )
                whitelisted = False
                if confirm_tool is not None:
                    whitelisted = await asyncio.to_thread(
                        confirm_tool,
                        f"WHITELIST_COMMAND:{pg_res['verdict']}",
                        {"prompt": params.prompt, "confidence": pg_res["score"]},
                    )
                if whitelisted:
                    pg.add_to_whitelist(params.prompt)
                    trace("[security notice] Prompt whitelisted by operator.")
                else:
                    return ToolLoopResult(
                        status="error",
                        error=(
                            f"[Security Rejection] Prompt Guard 2 (22M) flagged prompt as "
                            f"{pg_res['verdict']} (confidence: {pg_res['score']})."
                        ),
                    )
    except Exception:
        pass

    try:
        if microscope_client is not None:
            await microscope_client.connect()

        if incubator_client is not None and microscope_client is not None:
            incubator_tools = await incubator_client.list_tools()
            microscope_tools = await microscope_client.list_tools()
            base_tools = incubator_tools + microscope_tools
        else:
            base_tools = MOCK_TOOLS

        tools = merge_harness_tools(base_tools)
        tool_names = {tool["name"] for tool in tools}

        history: list[dict[str, str]] = [
            {"role": "system", "content": build_system_prompt(tools, file_index_md)},
            {
                "role": "user",
                "content": (
                    f"User objective: {params.prompt}\n"
                    f"Use incubator serial port '{params.port}' when a tool requires "
                    f"a `port` argument.\n"
                    f"For microscope_initialize, prefer xy_port="
                    f"'{params.microscope_xy_port}', "
                    f"kcube_serial='{params.microscope_kcube_serial}', "
                    f"camera_serial='{params.microscope_camera_serial}' "
                    "unless the user specifies otherwise."
                ),
            },
        ]

        for step in range(1, params.max_steps + 1):
            if abort_event is not None and abort_event.is_set():
                trace("\n[aborted] Stop requested before model step.")
                return ToolLoopResult(status="aborted", answer="")

            trace(f"\n[step {step}] requesting model…")
            try:
                model_text = await asyncio.to_thread(
                    ollama_client.chat_messages,
                    history,
                    params.model_name,
                )
            except OllamaClientError as exc:
                trace(f"[step {step}] Ollama error [{exc.kind}]: {exc}")
                return ToolLoopResult(status="error", error=str(exc))
            except Exception as exc:  # noqa: BLE001
                trace(f"[step {step}] Ollama error: {exc}")
                return ToolLoopResult(status="error", error=str(exc))

            trace(f"[step {step}] model raw:\n{model_text}")

            try:
                action = parse_agent_action(model_text)
            except (json.JSONDecodeError, ValueError, ValidationError) as exc:
                trace(f"[step {step}] parse/validation error: {exc}")
                history.append({"role": "assistant", "content": model_text})
                history.append({"role": "user", "content": _RETRY_JSON_HINT})
                continue

            if isinstance(action, FinalAction):
                answer = action.answer.strip()
                trace("\n[final]")
                trace(answer)
                return ToolLoopResult(status="completed", answer=answer)

            assert isinstance(action, ToolCallAction)
            tool_name = action.tool_name.strip()
            arguments = dict(action.arguments)

            if tool_name not in tool_names:
                history.append({"role": "assistant", "content": model_text})
                history.append(
                    {
                        "role": "user",
                        "content": (
                            f"Unknown tool '{tool_name}'. Use only provided tool names."
                        ),
                    }
                )
                continue

            if tool_name in INCUBATOR_TOOL_NAMES and "port" not in arguments:
                arguments["port"] = params.port
            arguments = _inject_microscope_defaults(tool_name, arguments, params)

            if abort_event is not None and abort_event.is_set():
                trace("\n[aborted] Stop requested before tool execution.")
                return ToolLoopResult(status="aborted", answer="")

            approved = True
            if confirm_tool is not None:
                approved = await asyncio.to_thread(confirm_tool, tool_name, arguments)

            if not approved:
                trace(f"[step {step}] tool {tool_name} declined by user.")
                result: Any = {"ok": False, "error": "User declined this tool call."}
            else:
                trace(f"[step {step}] calling tool={tool_name} args={arguments}")
                result = await _execute_tool(
                    tool_name=tool_name,
                    arguments=arguments,
                    params=params,
                    incubator_client=incubator_client,
                    microscope_client=microscope_client,
                    repo_root=repo_root,
                    allowed_paths=allowed_paths,
                )

            stdout_log = json.dumps(
                {
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result_for_stdout_log(tool_name, result),
                },
                ensure_ascii=True,
            )
            trace(f"[step {step}] tool result={stdout_log}")
            _append_tool_result_to_history(
                history, model_text, tool_name, arguments, result
            )

        trace("\n[final] Stopped: reached max_steps before a final answer.")
        return ToolLoopResult(status="max_steps", answer="")
    finally:
        if microscope_client is not None:
            await microscope_client.close()
