# MCP Host + Ollama Lab Tools

Local-first lab automation scaffold: **Cytomat incubator** and **microscope** (XY/Z + camera) wrapped as MCP tools and driven by a local **Ollama** model.

- **Incubator MCP server**: serial driver in `mcp_servers/incubator_outside.py`
- **Microscope MCP server**: hardware driver in `mcp_servers/microscope_hardware.py`
- **MCP host + harness**: stdio clients, JSON tool loop, optional GUI
- **Ollama clients**: local CLI plus HTTP streaming for GUI/chat workflows

## How it fits together

```
User prompt → Ollama (JSON: tool_call | final)
           → mcp_tool_loop (shared by CLI harness and GUI)
           → incubator MCP server → incubator_outside.py → serial hardware
           → microscope MCP server → microscope_hardware.py → XY serial + Thorlabs Z/camera
           → (+ harness-only read_repo_file, not on either MCP server)
```

The model emits strict JSON each turn (`tool_call` or `final`). The host validates that shape, routes tools to the incubator or microscope server (microscope tools are prefixed `microscope_`), feeds results back, and loops until the model answers or hits a step limit.

## Files

- `mcp_servers/incubator_outside.py` — Serial driver for Cytomat incubator hardware.
- `mcp_servers/incubator_mcp_server.py` — MCP stdio server exposing incubator tools.
- `mcp_servers/microscope_hardware.py` — XY stage, Thorlabs Z-stage, camera, autofocus.
- `mcp_servers/microscope_mcp_server.py` — MCP stdio server exposing `microscope_*` tools.
- `mcp_servers/voice_mcp_server.py` — FastMCP speech gateway exposing E2E ingestion, ASR & filtration.
- `mcp_servers/microscope_cli.py` — Optional human CLI for bench testing (no LLM/MCP).
- `mcp_host/incubator_client.py` — Host-side MCP stdio client for incubator.
- `mcp_host/microscope_client.py` — Long-lived host-side MCP client for microscope.
- `mcp_host/voice_client.py` — Host-side MCP client for voice gateway.
- `mcp_host/mcp_tool_loop.py` — Shared async agent loop (CLI + GUI).
- `mcp_host/agent_actions.py` — Pydantic schemas for model `final` / `tool_call` JSON.
- `mcp_host/harness_tools.py` — Harness-only tools (e.g. `read_repo_file`) and allowlist loading.
- `voice/audio_filter.py` — 8th-order Butterworth LPF, ultrasonic anomaly detector, lab noise synthesis.
- `voice/wake_detector.py` — Deterministic out-of-band E-stop (<100ms) & wake-word gating.
- `voice/asr_engine.py` — Qwen3-ASR streaming perception engine (dialects, area profiles, TTFT <100ms).
- `voice/voice_shield.py` — VoiceSHIELD & RENNERVATE indirect prompt injection defense.
- `voice/intent_translator.py` — Spoken intent to ActionDAG compiler & topological validator.
- `voice/voice_agent_bridge.py` — Execution bridge connecting voice to hardware with modal confirmation.
- `voice/benchmark_cli.py` — Interactive manual testing & benchmarking CLI for dialects, noise, ultrasound.
- `voice/test_voice_mcp.py` — Full unittest suite for FastMCP voice gateway.
- `mcp_host/repo_read.py` — Safe repo file read helper.
- `mcp_host/ollama_client.py` — Ollama CLI wrapper plus HTTP streaming helpers.
- `mcp_host/gui_app.py` — Tkinter app with model picker and MCP agent mode.
- `mcp_host/mcp_tool_harness.py` — CLI harness for model-driven MCP tools.
- `prompts/` — System prompts and Ollama Modelfile workflow (see `prompts/README.md`).
- `scripts/generate_incubator_modelfile.sh` — Builds a Modelfile that bakes the lab tool agent prompt into a Qwen-based custom tag.
- `requirements.txt` — Python dependencies.
- `AGENTS.md` — Agent and security principles for this repo.

## Microscope tools (for experiment design)

The model can call these after `microscope_initialize`:

| Tool | Purpose |
|------|---------|
| `microscope_initialize` | Connect Z-stage, camera, XY stage |
| `microscope_get_status` | Read X/Y/Z |
| `microscope_goto_xy` / `microscope_move_x` / `microscope_move_y` | XY motion |
| `microscope_move_z` / `microscope_move_z_relative` | Z motion (meters) |
| `microscope_home_*` | Home axes |
| `microscope_snap` | Capture image → `image_path` |
| `microscope_autofocus` | Tenengrad scan → `optimal_z` |
| `microscope_close` | Disconnect hardware |

Captured images are written under `mcp_servers/microscope_results/`.

### Manual microscope CLI (bench testing)

To drive the scope without the LLM/MCP agent (same hardware driver):

```bash
python -m mcp_servers.microscope_cli
python -m mcp_servers.microscope_cli --xy-port COM3 --kcube-serial 26004971
```

Type `help` at the `microscope>` prompt for commands (`mx`, `snap`, `autofocus`, etc.).

## Prompts and Ollama custom models

```bash
ollama pull qwen3:8b
./scripts/generate_incubator_modelfile.sh
ollama create incubator-qwen8b -f prompts/Modelfile.incubator-qwen8b.generated
```

The baked prompt (`prompts/incubator_tool_agent.system.txt`) covers **both** incubator and microscope tools. Pick `incubator-qwen8b` in the GUI or set `OLLAMA_MODEL=incubator-qwen8b`.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
  ```bash
   pip install -r requirements.txt
  ```
3. Ensure `ollama` CLI is installed locally and your model is pulled.
4. For real microscope hardware, Thorlabs drivers / Kinesis support expected by `pylablib` must be installed on the machine.

## Environment Variables

- `OLLAMA_COMMAND` (default: `ollama`)
- `OLLAMA_MODEL` (default: `llama3.1`)
- `INCUBATOR_SERIAL_PORT` (default: `COM1`; on macOS/Linux use e.g. `/dev/ttyUSB0`)
- `MICROSCOPE_XY_PORT` (default: `COM3`)
- `MICROSCOPE_KCUBE_SERIAL` (default: `26004971`)
- `MICROSCOPE_CAMERA_SERIAL` (default: `25267`)
- `BIOAGENT_ROOT` (optional) — Absolute repo root for harness `read_repo_file` allowlist resolution.

## Run GUI App

```bash
python -m mcp_host.gui_app
```

**MCP agent:** use **Run MCP agent** for the same JSON tool loop as `mcp_tool_harness.py` (incubator + microscope + `read_repo_file`). Each tool call opens a **confirmation** dialog; **Stop / Abort** cancels between steps. Use **Mock tools** to avoid real hardware.

## Test MCP Tool Capabilities

```bash
python -m mcp_host.mcp_tool_harness --prompt "List what incubator and microscope tools you can use, then check incubator readiness"
```

Optional flags: `--max-steps` (default `6`), `--port` (incubator serial), `--mock-tools` (simulated incubator + microscope responses).

## Test Model Loop With Mock Tools

```bash
python -m mcp_host.mcp_tool_harness --mock-tools --prompt "Initialize the microscope, autofocus, snap an image, then summarize"
```

Mock mode keeps the same JSON action/tool loop but returns deterministic simulated tool results for both devices.
