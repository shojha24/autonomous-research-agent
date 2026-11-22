#!/usr/bin/env bash
# Build prompts/Modelfile.incubator-qwen8b.generated from incubator_tool_agent.system.txt
# so Ollama can bake the system prompt: ollama create incubator-qwen8b -f <that file>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT="${ROOT}/prompts/incubator_tool_agent.system.txt"
OUT="${ROOT}/prompts/Modelfile.incubator-qwen8b.generated"
BASE_MODEL="${BASE_OLLAMA_MODEL:-qwen3:8b}"

if [[ ! -f "$PROMPT" ]]; then
  echo "Missing prompt file: $PROMPT" >&2
  exit 1
fi

{
  printf 'FROM %s\n\n' "$BASE_MODEL"
  printf 'SYSTEM """\n'
  cat "$PROMPT"
  printf '\n"""\n'
} > "$OUT"

echo "Wrote: $OUT"
echo "Pull base if needed:  ollama pull ${BASE_MODEL}"
echo "Create custom model:  ollama create incubator-qwen8b -f \"${OUT}\""
