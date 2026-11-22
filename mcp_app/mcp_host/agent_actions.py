"""Validate the JSON object the model emits each turn (tool_call or final).

The harness expects exactly one JSON object per model reply. These schemas
check shape before any tool runs.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

__all__ = [
    "FinalAction",
    "ToolCallAction",
    "parse_agent_action",
    "ValidationError",
]


class FinalAction(BaseModel):
    """Model ends the loop with a natural-language answer.

    Attributes:
        action: Must be the literal "final".
        answer: Text shown to the user as the final response.
    """

    model_config = ConfigDict(extra="ignore")

    action: Literal["final"]
    answer: str = ""


class ToolCallAction(BaseModel):
    """Model requests one MCP tool call.

    Attributes:
        action: Must be the literal "tool_call".
        tool_name: Name from the tool list in the system prompt.
        arguments: JSON object passed to the tool handler.
    """

    model_config = ConfigDict(extra="ignore")

    action: Literal["tool_call"]
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


_AGENT_ACTION_ADAPTER: TypeAdapter[FinalAction | ToolCallAction] = TypeAdapter(
    FinalAction | ToolCallAction
)


def parse_agent_action(raw_response: str) -> FinalAction | ToolCallAction:
    """Parse model text as JSON and validate it as a final or tool_call action.

    Args:
        raw_response: Raw assistant text (should be one JSON object).

    Returns:
        Validated FinalAction or ToolCallAction instance.

    Raises:
        json.JSONDecodeError: Text is not valid JSON.
        ValueError: JSON root is not an object.
        ValidationError: Fields or types do not match the schema.
    """
    payload = json.loads(raw_response.strip())
    if not isinstance(payload, dict):
        msg = "Model response must be a JSON object."
        raise ValueError(msg)
    return _AGENT_ACTION_ADAPTER.validate_python(payload)
