# Repository file index (harness `read_repo_file`)

Only paths listed in the **machine-readable allowlist** below may be read via the harness tool `read_repo_file`. To expose another file, add a row here and add the same path inside the allowlist block.

## Machine-readable allowlist

Parser rules:

- Everything between `<!-- allowlist-begin -->` and `<!-- allowlist-end -->` (inclusive boundaries on their own lines) is scanned.
- Each non-empty, non-comment line is one **repo-relative** path using forward slashes.
- Lines starting with `#` are ignored.

<!-- allowlist-begin -->
AGENTS.md
README.md
requirements.txt
mcp_host/mcp_tool_harness.py
mcp_host/mcp_tool_loop.py
mcp_host/agent_actions.py
mcp_host/harness_tools.py
mcp_host/repo_read.py
mcp_host/ollama_client.py
mcp_host/gui_app.py
mcp_host/incubator_client.py
mcp_host/microscope_client.py
mcp_servers/incubator_mcp_server.py
mcp_servers/incubator_outside.py
mcp_servers/microscope_mcp_server.py
mcp_servers/microscope_hardware.py
mcp_servers/microscope_cli.py
prompts/README.md
prompts/FILE_INDEX.md
prompts/incubator_tool_agent.system.txt
<!-- allowlist-end -->

## Human descriptions

| Path | Role |
|------|------|
| `AGENTS.md` | Agent and security principles for this repo. |
| `README.md` | Top-level setup, env vars, how to run GUI and harness. |
| `requirements.txt` | Python dependencies (core + microscope extras). |
| `mcp_host/mcp_tool_harness.py` | Ollama + MCP tool loop CLI (JSON actions, mock/real tools). |
| `mcp_host/mcp_tool_loop.py` | Shared async agent loop (incubator + microscope + read_repo_file). |
| `mcp_host/agent_actions.py` | Pydantic schemas for model `final` / `tool_call` JSON. |
| `mcp_host/harness_tools.py` | Harness-only tool descriptors (e.g. `read_repo_file`) and allowlist loading. |
| `mcp_host/repo_read.py` | Safe repo file read helper (path checks, size cap). |
| `mcp_host/ollama_client.py` | Local Ollama CLI and HTTP chat helpers. |
| `mcp_host/gui_app.py` | Tkinter GUI for streamed Ollama chat and MCP agent mode. |
| `mcp_host/incubator_client.py` | MCP stdio client for incubator server. |
| `mcp_host/microscope_client.py` | Long-lived MCP stdio client for microscope server. |
| `mcp_servers/incubator_mcp_server.py` | FastMCP server exposing incubator tools. |
| `mcp_servers/incubator_outside.py` | Serial driver and validation for Cytomat hardware. |
| `mcp_servers/microscope_mcp_server.py` | FastMCP server exposing microscope_* tools. |
| `mcp_servers/microscope_hardware.py` | XY/Z/camera driver for the lab microscope. |
| `mcp_servers/microscope_cli.py` | Human CLI for manual microscope bench testing (no LLM). |
| `prompts/README.md` | How to build Modelfile / custom Ollama models from prompts. |
| `prompts/FILE_INDEX.md` | This index and allowlist. |
| `prompts/incubator_tool_agent.system.txt` | Canonical system prompt for incubator + microscope JSON-tool agent. |
