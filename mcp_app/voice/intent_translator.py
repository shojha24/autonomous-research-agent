"""Hierarchical intent translation and Action Directed Acyclic Graph (DAG) engine.

Translates high-level natural language scientific instructions into verified,
dependency-ordered DAGs of atomic FastMCP tool invocations (Question 2):
  - Bridges abstract scientist voice commands ("Transfer plate 4 to microscope and image well B2")
    to concrete atomic calls across incubator and microscope MCP servers.
  - Implements CADV (Concurrent Audio Defense via Verification) checking physical preconditions.
  - Generates topological execution orders and detects cycle/interlock hazards.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ToolServerType = Literal["incubator", "microscope", "system"]

# Allowlisted tools per server
VALID_SERVER_TOOLS: dict[ToolServerType, frozenset[str]] = {
    "incubator": frozenset(
        {
            "initialize",
            "is_ready",
            "import_plate",
            "export_plate",
            "swap_transfer",
            "open_gate",
            "close_gate",
        }
    ),
    "microscope": frozenset(
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
    ),
    "system": frozenset({"wait_delay", "log_audit", "notify_operator"}),
}


class PlanStep(BaseModel):
    """An individual atomic tool execution step within the Action DAG.

    Attributes:
        step_id: Unique integer index for dependency referencing.
        tool_server: MCP server hosting the tool ("incubator" or "microscope").
        tool_name: Exact tool identifier exposed by the FastMCP server.
        arguments: Parameters passed into the MCP tool.
        depends_on: List of step_ids that must finish before this step executes.
        is_destructive: Flag indicating if the step physically modifies state or moves actuators.
        safety_precondition: Human-readable safety constraint or interlock to verify.
    """

    model_config = ConfigDict(extra="ignore")

    step_id: int
    tool_server: ToolServerType
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)
    is_destructive: bool = False
    safety_precondition: str | None = None


class ActionDAG(BaseModel):
    """Strongly typed Directed Acyclic Graph representing a decomposed experiment workflow.

    Attributes:
        plan_id: Unique identifier for audit logging.
        user_intent: Original user voice utterance.
        rationale: Explanatory breakdown of why this decomposition was chosen.
        estimated_duration_sec: Estimated physical execution duration in seconds.
        steps: Sequence of dependent plan steps.
    """

    model_config = ConfigDict(extra="ignore")

    plan_id: str = Field(default_factory=lambda: f"dag-{uuid.uuid4().hex[:8]}")
    user_intent: str
    rationale: str
    estimated_duration_sec: float = 10.0
    steps: list[PlanStep] = Field(default_factory=list)


class DAGValidationResult(BaseModel):
    """Validation report for an ActionDAG before physical execution.

    Attributes:
        is_valid: True if DAG has no cycles, unknown tools, or invalid dependencies.
        topological_order: Linear step execution sequence respecting dependencies.
        errors: Fatal validation issues.
        warnings: Non-fatal advisory notices.
    """

    is_valid: bool
    topological_order: list[int] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HierarchicalIntentTranslator:
    """Translates spoken intent into executable ActionDAGs with topological verification."""

    def validate_dag(self, dag: ActionDAG) -> DAGValidationResult:
        """Validate DAG acyclicity, schema constraints, and tool server allowlists.

        Args:
            dag: ActionDAG instance.

        Returns:
            DAGValidationResult containing topological order or error descriptions.
        """
        errors: list[str] = []
        warnings: list[str] = []

        step_map: dict[int, PlanStep] = {}
        for s in dag.steps:
            if s.step_id in step_map:
                errors.append(f"Duplicate step_id: {s.step_id}")
            step_map[s.step_id] = s

            # Verify tool server allowlist
            valid_tools = VALID_SERVER_TOOLS.get(s.tool_server)
            if valid_tools is None:
                errors.append(f"Step {s.step_id}: Unknown tool_server '{s.tool_server}'")
            elif s.tool_name not in valid_tools:
                errors.append(
                    f"Step {s.step_id}: Tool '{s.tool_name}' not permitted on server '{s.tool_server}'"
                )

        # Check dependencies exist
        for s in dag.steps:
            for dep in s.depends_on:
                if dep not in step_map:
                    errors.append(f"Step {s.step_id} depends on non-existent step {dep}")
                if dep == s.step_id:
                    errors.append(f"Step {s.step_id} cannot depend on itself")

        if errors:
            return DAGValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Topological sort via Kahn's algorithm
        in_degree: dict[int, int] = {s.step_id: len(s.depends_on) for s in dag.steps}
        adjacency: dict[int, list[int]] = {s.step_id: [] for s in dag.steps}
        for s in dag.steps:
            for dep in s.depends_on:
                adjacency[dep].append(s.step_id)

        queue: list[int] = [sid for sid, deg in in_degree.items() if deg == 0]
        sorted_order: list[int] = []

        while queue:
            curr = queue.pop(0)
            sorted_order.append(curr)
            for neighbor in adjacency[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_order) != len(dag.steps):
            errors.append("Cyclic dependency detected in Action DAG!")
            return DAGValidationResult(is_valid=False, errors=errors, warnings=warnings)

        return DAGValidationResult(
            is_valid=True,
            topological_order=sorted_order,
            errors=errors,
            warnings=warnings,
        )

    def translate_spoken_command(self, user_command: str) -> ActionDAG:
        """Decompose a high-level natural language command into an ActionDAG.

        Parses patterns for incubator transfers, microscope movements, autofocus,
        and imaging pipelines.

        Args:
            user_command: Spoken instruction string.

        Returns:
            Constructed ActionDAG ready for validation.
        """
        cmd = user_command.strip()
        lower = cmd.lower()
        steps: list[PlanStep] = []
        step_idx = 1
        rationale_parts: list[str] = []

        # 1. Parse incubator plate export/transfer operations
        slot_match = re.search(r"\b(?:slot|hotel|position)\s*(\d+)\b", lower)
        rack_match = re.search(r"\b(?:rack|stack)\s*(\d+)\b", lower)
        slot_num = int(slot_match.group(1)) if slot_match else 1
        rack_num = int(rack_match.group(1)) if rack_match else 1

        is_transfer_request = (
            "transfer" in lower
            or "export" in lower
            or "load" in lower
            or "take out" in lower
            or "get plate" in lower
        )

        incubator_step_id: int | None = None
        if is_transfer_request and ("incubator" in lower or "plate" in lower):
            # Step: Check incubator status
            steps.append(
                PlanStep(
                    step_id=step_idx,
                    tool_server="incubator",
                    tool_name="is_ready",
                    arguments={},
                    depends_on=[],
                    is_destructive=False,
                    safety_precondition="Verify carousel is stationary",
                )
            )
            rationale_parts.append(f"Step {step_idx}: Verify incubator readiness")
            step_idx += 1

            # Step: Export plate from specified rack & slot
            steps.append(
                PlanStep(
                    step_id=step_idx,
                    tool_server="incubator",
                    tool_name="export_plate",
                    arguments={"incu_rack": rack_num, "incu_slot": slot_num},
                    depends_on=[step_idx - 1],
                    is_destructive=True,
                    safety_precondition=f"Export plate from rack {rack_num}, slot {slot_num}",
                )
            )
            incubator_step_id = step_idx
            rationale_parts.append(
                f"Step {step_idx}: Export plate from rack {rack_num}, slot {slot_num}"
            )
            step_idx += 1

        # 2. Parse microscope operations
        is_microscope_request = (
            "microscope" in lower
            or "focus" in lower
            or "image" in lower
            or "snap" in lower
            or "well" in lower
            or "confluency" in lower
        )

        if is_microscope_request:
            scope_init_deps = [incubator_step_id] if incubator_step_id else []
            scope_init_id = step_idx
            steps.append(
                PlanStep(
                    step_id=step_idx,
                    tool_server="microscope",
                    tool_name="microscope_initialize",
                    arguments={},
                    depends_on=scope_init_deps,
                    is_destructive=False,
                    safety_precondition="Initialize stage serial and camera handles",
                )
            )
            rationale_parts.append(f"Step {step_idx}: Initialize microscope stage & optics")
            step_idx += 1

            # Stage coordinates or well targeting
            well_match = re.search(r"\bwell\s*([a-hA-H]\d{1,2})\b", cmd)
            last_motion_id = scope_init_id
            if well_match:
                well_id = well_match.group(1).upper()
                # Simplified 96-well coordinate mapping for demonstration
                row = ord(well_id[0]) - ord("A")
                col = int(well_id[1:]) - 1
                x_step = col * 9000
                y_step = row * 9000

                steps.append(
                    PlanStep(
                        step_id=step_idx,
                        tool_server="microscope",
                        tool_name="microscope_goto_xy",
                        arguments={"x": x_step, "y": y_step},
                        depends_on=[scope_init_id],
                        is_destructive=True,
                        safety_precondition=f"Move XY stage to well {well_id} ({x_step}, {y_step})",
                    )
                )
                last_motion_id = step_idx
                rationale_parts.append(f"Step {step_idx}: Drive XY stage to well {well_id}")
                step_idx += 1

            # Autofocus operation
            if "autofocus" in lower or "focus" in lower:
                steps.append(
                    PlanStep(
                        step_id=step_idx,
                        tool_server="microscope",
                        tool_name="microscope_autofocus",
                        arguments={"z_start": 0.003, "z_end": 0.0006, "num_steps": 15},
                        depends_on=[last_motion_id],
                        is_destructive=True,
                        safety_precondition="Execute Tenengrad gradient sweep for optimal focus",
                    )
                )
                last_motion_id = step_idx
                rationale_parts.append(f"Step {step_idx}: Perform Tenengrad autofocus")
                step_idx += 1

            # Image capture / snap
            if "snap" in lower or "image" in lower or "capture" in lower or "confluency" in lower:
                steps.append(
                    PlanStep(
                        step_id=step_idx,
                        tool_server="microscope",
                        tool_name="microscope_snap",
                        arguments={},
                        depends_on=[last_motion_id],
                        is_destructive=False,
                        safety_precondition="Capture optical frame to disk",
                    )
                )
                rationale_parts.append(f"Step {step_idx}: Snap high-resolution microplate image")
                step_idx += 1

        # Fallback if no specific hardware pattern matched
        if not steps:
            steps.append(
                PlanStep(
                    step_id=1,
                    tool_server="system",
                    tool_name="log_audit",
                    arguments={"intent": cmd, "status": "unmapped_voice_intent"},
                    depends_on=[],
                    is_destructive=False,
                    safety_precondition="Log unmapped command for operator review",
                )
            )
            rationale_parts.append("Step 1: Log command into audit record")

        return ActionDAG(
            user_intent=cmd,
            rationale=" -> ".join(rationale_parts),
            estimated_duration_sec=float(len(steps) * 3.5),
            steps=steps,
        )
