# Prompts

This folder holds reusable system prompts and helpers to **bake them into Ollama** as a custom model (so you do not have to paste the block every session).

## Ollama: custom model with a fixed system prompt

Ollama supports a **Modelfile** with a `SYSTEM` directive. Running `ollama create` builds a new tag (e.g. `incubator-qwen8b`) whose default system message is your text. The GUI and API then select that tag like any other model.

1. Pull the base model (default in this repo: `qwen3:8b`):
  ```bash
   ollama pull qwen3:8b
  ```
2. Generate a Modelfile from `incubator_tool_agent.system.txt` (covers incubator **and** microscope tools):
  ```bash
   ./scripts/generate_incubator_modelfile.sh
  ```
   Override the base tag if yours differs (e.g. `qwen2.5:8b`):
3. Create the custom model:
  ```bash
   ollama create incubator-qwen8b -f prompts/Modelfile.incubator-qwen8b.generated
  ```
4. Use it everywhere you would pick a model:
  - **GUI**: choose `incubator-qwen8b` from Refresh Models (or set `OLLAMA_MODEL=incubator-qwen8b`).
  - **Harness**: `export OLLAMA_MODEL=incubator-qwen8b` then run `python -m mcp_host.mcp_tool_harness ...`.

**Note:** `mcp_tool_harness.py` already injects its own system prompt (live tool list + JSON schema from both MCP servers). If you use the baked model there, you get **two** system-style instructions unless you change the harness. For pure “memory + few-shot” experiments in the **GUI**, the baked model is ideal because the GUI only adds your editable system line on top—or you can clear the GUI system field to rely on the baked default.

## Manual paste (no custom model)

You can always paste `incubator_tool_agent.system.txt` into the GUI’s **System prompt** field, or prepend it in your own client. Ollama’s HTTP API also accepts a `system` message in `messages`; this repo’s `OllamaClient.stream_chat` already supports `system_prompt` from the GUI.

## Files


| File                                   | Purpose                                                                                                           |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `incubator_tool_agent.system.txt`      | Canonical JSON-tool agent system prompt (incubator + microscope).                                                 |
| `Modelfile.incubator-qwen8b.generated` | Produced by the script; safe to delete and regenerate.                                                            |
| `FILE_INDEX.md`                        | Repository file index, allowlist, and descriptions for harness `read_repo_file`. What files can the model access? |


## Harness `read_repo_file`

`python -m mcp_host.mcp_tool_harness` adds a harness-only tool `**read_repo_file`** (not on the MCP servers). The model may call it with `{"path": "<repo-relative>"}` only for paths listed between `<!-- allowlist-begin -->` / `<!-- allowlist-end -->` in `[FILE_INDEX.md](FILE_INDEX.md)`. To expose a new file, add its path there and a short row in the table. Optional env: `**BIOAGENT_ROOT`** — absolute repo root if the harness is not run from this checkout. Reads are capped (65 KiB) and UTF-8 with replacement; do not put secrets in allowlisted files.
