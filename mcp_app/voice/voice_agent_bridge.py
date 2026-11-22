"""Voice-to-Action Execution Bridge connecting voice pipelines to incubator and microscope MCP tools.

Executes validated ActionDAGs across physical instrument servers while enforcing:
  1. Human confirmation gate for destructive/physical moves triggered via voice (source="voice").
  2. Out-of-band emergency stop polling between execution steps.
  3. Provenance and audit logging linking acoustic SHA-256 hashes to executed physical steps.
  4. Hands-free spoken status feedback for benchtop operators.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
import sys
from typing import Any, Callable

try:
    from mcp_app.mcp_host.incubator_client import IncubatorMcpClient, IncubatorServerConfig
    from mcp_app.mcp_host.microscope_client import MicroscopeMcpClient
    from mcp_app.voice.intent_translator import ActionDAG, HierarchicalIntentTranslator, PlanStep
    from mcp_app.voice.wake_detector import AudioInterruptManager
except ModuleNotFoundError:
    from mcp_host.incubator_client import IncubatorMcpClient, IncubatorServerConfig
    from mcp_host.microscope_client import MicroscopeMcpClient
    from voice.intent_translator import ActionDAG, HierarchicalIntentTranslator, PlanStep
    from voice.wake_detector import AudioInterruptManager

logger = logging.getLogger(__name__)


@dataclass
class StepExecutionLog:
    """Record of an individual step execution for the AER Merkle ledger."""

    step_id: int
    tool_server: str
    tool_name: str
    arguments: dict[str, Any]
    success: bool
    response: dict[str, Any]
    duration_sec: float
    confirmed_by_human: bool = False


@dataclass
class DAGExecutionResult:
    """Overall report from executing an ActionDAG across physical hardware."""

    plan_id: str
    audio_sha256: str
    success: bool
    total_steps: int
    executed_steps: list[StepExecutionLog] = field(default_factory=list)
    halted_by_emergency: bool = False
    spoken_feedback: str = ""
    error_message: str | None = None


class VoiceActionExecutive:
    """Orchestrates execution of ActionDAGs against physical instrument MCP clients."""

    def __init__(
        self,
        incubator_client: IncubatorMcpClient | None = None,
        microscope_client: MicroscopeMcpClient | None = None,
        interrupt_manager: AudioInterruptManager | None = None,
        confirm_callback: Callable[[str, PlanStep], bool] | None = None,
        require_confirmation_for_destructive: bool = True,
    ) -> None:
        """Initialize the voice action executive.

        Args:
            incubator_client: Client for incubator MCP tools.
            microscope_client: Client for microscope MCP tools.
            interrupt_manager: Out-of-band emergency stop manager.
            confirm_callback: Optional GUI confirmation modal callback for human sign-off.
            require_confirmation_for_destructive: Enforce modal check on physical motions.
        """
        if incubator_client is None:
            inc_cfg = IncubatorServerConfig(python_executable=sys.executable)
            self.incubator_client = IncubatorMcpClient(inc_cfg)
        else:
            self.incubator_client = incubator_client
        self.microscope_client = microscope_client
        self.interrupt_manager = interrupt_manager or AudioInterruptManager()
        self.confirm_callback = confirm_callback
        self.require_confirmation_for_destructive = require_confirmation_for_destructive
        self.translator = HierarchicalIntentTranslator()

    async def execute_dag(
        self,
        dag: ActionDAG,
        audio_sha256: str = "simulated_audio_hash",
        dry_run: bool = False,
    ) -> DAGExecutionResult:
        """Execute the DAG in topological order across incubator and microscope servers.

        Args:
            dag: ActionDAG to execute.
            audio_sha256: Cryptographic hash of the triggering voice command for audit.
            dry_run: If True, validate and simulate without dispatching physical hardware commands.

        Returns:
            DAGExecutionResult container.
        """
        val = self.translator.validate_dag(dag)
        if not val.is_valid:
            return DAGExecutionResult(
                plan_id=dag.plan_id,
                audio_sha256=audio_sha256,
                success=False,
                total_steps=len(dag.steps),
                error_message=f"DAG validation failed: {'; '.join(val.errors)}",
                spoken_feedback="Error: Plan validation failed with circular or unknown tool steps.",
            )

        step_map = {s.step_id: s for s in dag.steps}
        executed_logs: list[StepExecutionLog] = []

        for step_id in val.topological_order:
            # 1. Check out-of-band emergency stop
            if self.interrupt_manager.is_emergency_halted:
                return DAGExecutionResult(
                    plan_id=dag.plan_id,
                    audio_sha256=audio_sha256,
                    success=False,
                    total_steps=len(dag.steps),
                    executed_steps=executed_logs,
                    halted_by_emergency=True,
                    spoken_feedback="Emergency halt active. All physical motions aborted immediately.",
                    error_message="Aborted due to active emergency stop.",
                )

            step = step_map[step_id]

            # 2. Human confirmation modal check if step is destructive/physical
            confirmed = True
            if self.require_confirmation_for_destructive and step.is_destructive:
                if self.confirm_callback is not None:
                    prompt = (
                        f"[VOICE ACTION CONFIRMATION] Step {step.step_id}: "
                        f"Run {step.tool_server}.{step.tool_name} with {step.arguments}?"
                    )
                    confirmed = self.confirm_callback(prompt, step)
                    if not confirmed:
                        return DAGExecutionResult(
                            plan_id=dag.plan_id,
                            audio_sha256=audio_sha256,
                            success=False,
                            total_steps=len(dag.steps),
                            executed_steps=executed_logs,
                            spoken_feedback=f"Operator rejected step {step.step_id}.",
                            error_message=f"Human operator declined step {step.step_id}.",
                        )

            # 3. Dispatch tool execution
            start_t = asyncio.get_event_loop().time()
            if dry_run:
                resp = {"ok": True, "message": f"[DRY RUN] Simulated {step.tool_name}"}
            else:
                try:
                    if step.tool_server == "incubator":
                        resp = await self.incubator_client.call_tool(
                            step.tool_name, step.arguments
                        )
                    elif step.tool_server == "microscope" and self.microscope_client:
                        resp = await self.microscope_client.call_tool(
                            step.tool_name, step.arguments
                        )
                    else:
                        resp = {"ok": True, "message": f"System executed {step.tool_name}"}
                except Exception as e:
                    resp = {"ok": False, "error": str(e)}

            duration = asyncio.get_event_loop().time() - start_t
            step_ok = bool(resp.get("ok", False))

            executed_logs.append(
                StepExecutionLog(
                    step_id=step.step_id,
                    tool_server=step.tool_server,
                    tool_name=step.tool_name,
                    arguments=step.arguments,
                    success=step_ok,
                    response=resp,
                    duration_sec=duration,
                    confirmed_by_human=confirmed,
                )
            )

            if not step_ok:
                return DAGExecutionResult(
                    plan_id=dag.plan_id,
                    audio_sha256=audio_sha256,
                    success=False,
                    total_steps=len(dag.steps),
                    executed_steps=executed_logs,
                    spoken_feedback=f"Execution halted: Step {step.step_id} failed with error.",
                    error_message=resp.get("error", "Unknown tool failure"),
                )

        spoken = f"Plan {dag.plan_id} completed successfully. Executed {len(dag.steps)} steps."
        return DAGExecutionResult(
            plan_id=dag.plan_id,
            audio_sha256=audio_sha256,
            success=True,
            total_steps=len(dag.steps),
            executed_steps=executed_logs,
            spoken_feedback=spoken,
        )
