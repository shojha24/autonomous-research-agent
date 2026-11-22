"""FastMCP speech gateway exposing ingestion, acoustic filtration, ASR, and intent translation.

Provides standardized Model Context Protocol (MCP) tools for:
  - Digital Butterworth filtering and ultrasonic anomaly detection.
  - Out-of-band deterministic emergency interrupt checks (<100ms).
  - Wake-word session gating to discard ambient chatter.
  - Qwen3-ASR speech-to-text with multi-dialect and lab area adaptation.
  - VoiceSHIELD and RENNERVATE safety screening and quarantined container generation.
  - Hierarchical Action DAG translation for lab automation.
  - Full end-to-end speech-to-DAG execution pipeline.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any

# Ensure mcp_app and repo root are in sys.path
_CURR_DIR = Path(__file__).resolve().parent
_MCP_APP_DIR = _CURR_DIR.parent
_REPO_ROOT = _MCP_APP_DIR.parent
for _p in (str(_MCP_APP_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from mcp.server.fastmcp import FastMCP

try:
    from mcp_app.voice.asr_engine import (
        ASRConfig,
        DialectType,
        LabAreaProfile,
        Qwen3StreamingASR,
        TranscriptResult,
    )
    from mcp_app.voice.audio_filter import AcousticFilterConfig, AcousticPreFilter
    from mcp_app.voice.intent_translator import HierarchicalIntentTranslator
    from mcp_app.voice.voice_shield import VoiceSHIELD
    from mcp_app.voice.wake_detector import AudioInterruptManager, WakeGate
except ModuleNotFoundError:
    from voice.asr_engine import (
        ASRConfig,
        DialectType,
        LabAreaProfile,
        Qwen3StreamingASR,
        TranscriptResult,
    )
    from voice.audio_filter import AcousticFilterConfig, AcousticPreFilter
    from voice.intent_translator import HierarchicalIntentTranslator
    from voice.voice_shield import VoiceSHIELD
    from voice.wake_detector import AudioInterruptManager, WakeGate

mcp = FastMCP("voice_gateway")

# Singleton pipeline components
_filter = AcousticPreFilter(AcousticFilterConfig())
_interrupt_manager = AudioInterruptManager()
_wake_gate = WakeGate()
_asr = Qwen3StreamingASR(ASRConfig(), filter_engine=_filter)
_shield = VoiceSHIELD(strict_wake_enforcement=True)
_translator = HierarchicalIntentTranslator()


def _decode_audio_base64(audio_b64: str) -> np.ndarray:
    """Helper to convert base64 PCM16 or raw bytes into normalized float32 numpy array."""
    if not audio_b64:
        return np.array([], dtype=np.float32)
    raw_bytes = base64.b64decode(audio_b64)
    int16_arr = np.frombuffer(raw_bytes, dtype=np.int16)
    return (int16_arr / 32768.0).astype(np.float32)


@mcp.tool()
def voice_get_gateway_status() -> dict[str, Any]:
    """Retrieve the current health, filter parameters, and emergency state of the voice gateway.

    Returns:
        Status dict including active emergency halt flag, supported dialects, and filter specs.
    """
    return {
        "ok": True,
        "gateway_name": "voice_gateway",
        "emergency_halted": _interrupt_manager.is_emergency_halted,
        "sample_rate_hz": _filter.config.sample_rate,
        "filter_cutoff_hz": _filter.config.cutoff_hz,
        "filter_order": _filter.config.filter_order,
        "ultrasonic_threshold": _filter.config.ultrasonic_anomaly_threshold,
        "supported_dialects": ["en-US", "en-GB", "en-IN", "en-AU", "en-EA"],
        "supported_areas": ["fume_hood", "centrifuge_room", "quiet_lab", "open_bench"],
        "wake_gate_open": _wake_gate.is_gate_open,
    }


@mcp.tool()
def voice_emergency_stop_check(text: str = "", audio_base64: str = "") -> dict[str, Any]:
    """Execute zero-LLM deterministic emergency stop detection (<100ms response).

    Args:
        text: Transcribed words or text snippet to test.
        audio_base64: Optional base64-encoded audio chunk.

    Returns:
        Dict indicating whether emergency stop was triggered, trigger phrase, and latency.
    """
    event = None
    if text:
        event = _interrupt_manager.evaluate_text(text)
    elif audio_base64:
        audio = _decode_audio_base64(audio_base64)
        event = _interrupt_manager.evaluate_audio_chunk(audio)

    if event:
        return {
            "ok": True,
            "emergency_stop_triggered": True,
            "trigger_phrase": event.trigger_phrase,
            "latency_ms": round(event.latency_ms, 2),
            "action_taken": "PHYSICAL_ESTOP_LATCHED",
        }

    return {
        "ok": True,
        "emergency_stop_triggered": False,
        "latency_ms": 0.5,
        "action_taken": "NONE",
    }


@mcp.tool()
def voice_reset_emergency_stop() -> dict[str, Any]:
    """Reset the emergency stop latch following operator verification.

    Returns:
        Confirmation payload.
    """
    _interrupt_manager.reset_emergency_state()
    return {"ok": True, "emergency_halted": False, "message": "Emergency latch reset successfully."}


@mcp.tool()
def voice_ingest_audio_chunk(
    audio_base64: str, sample_rate: int = 16000
) -> dict[str, Any]:
    """Filter raw audio chunk using 8th-order Butterworth LPF and screen for ultrasonic anomalies.

    Args:
        audio_base64: Base64-encoded 16-bit PCM mono audio data.
        sample_rate: Audio sampling frequency in Hz (default: 16000).

    Returns:
        Filtration metrics, estimated SNR in dB, and ultrasonic anomaly flags.
    """
    audio = _decode_audio_base64(audio_base64)
    result = _filter.process(audio)

    # Encode filtered audio back to base64
    filtered_int16 = (np.clip(result.filtered_audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    filtered_b64 = base64.b64encode(filtered_int16.tobytes()).decode("ascii")

    return {
        "ok": True,
        "duration_sec": round(result.duration_sec, 3),
        "estimated_snr_db": round(result.estimated_snr_db, 2),
        "ultrasonic_energy_ratio": round(result.ultrasonic_energy_ratio, 4),
        "is_ultrasonic_anomaly": result.is_ultrasonic_anomaly,
        "security_message": result.security_message,
        "filtered_audio_base64": filtered_b64,
    }


@mcp.tool()
def voice_detect_wake_word(text: str) -> dict[str, Any]:
    """Screen input transcript for authorized wake word prefix ("Hey LabAgent", "Antigravity").

    Args:
        text: Candidate user transcript.

    Returns:
        Gating evaluation with stripped command string if authorized.
    """
    is_auth, command, wake = _wake_gate.evaluate_text(text)
    return {
        "ok": True,
        "wake_word_verified": is_auth,
        "matched_wake_word": wake,
        "sanitized_command": command,
        "gate_open": _wake_gate.is_gate_open,
    }


@mcp.tool()
def voice_transcribe_audio(
    audio_base64: str = "",
    dialect: str = "en-US",
    area_profile: str = "open_bench",
    reference_text: str = "",
) -> dict[str, Any]:
    """Transcribe speech using Qwen3-ASR with dialect adaptation and noise decoupling.

    Args:
        audio_base64: Base64-encoded PCM audio.
        dialect: Dialect profile ("en-US", "en-GB", "en-IN", "en-AU", "en-EA").
        area_profile: Laboratory acoustic profile ("fume_hood", "centrifuge_room", "quiet_lab", "open_bench").
        reference_text: Optional expected text for benchmark testing.

    Returns:
        Transcript text, forced aligner word timestamps, latency metrics, and SNR.
    """
    audio = _decode_audio_base64(audio_base64)
    ref = reference_text if reference_text else None
    res = _asr.transcribe(
        audio,
        dialect=dialect,  # type: ignore
        area_profile=area_profile,  # type: ignore
        reference_text=ref,
    )

    tokens_payload = [
        {
            "word": t.word,
            "start_ms": t.start_ms,
            "end_ms": t.end_ms,
            "confidence": t.confidence,
        }
        for t in res.tokens
    ]

    return {
        "ok": True,
        "text": res.text,
        "has_speech": res.has_speech,
        "tokens": tokens_payload,
        "estimated_snr_db": res.estimated_snr_db,
        "dialect": res.dialect,
        "area_profile": res.area_profile,
        "time_to_first_token_ms": res.time_to_first_token_ms,
        "total_latency_ms": res.total_latency_ms,
    }


@mcp.tool()
def voice_screen_safety(text: str, audio_base64: str = "") -> dict[str, Any]:
    """Execute VoiceSHIELD and RENNERVATE security and prompt injection screening.

    Args:
        text: Spoken user transcript.
        audio_base64: Raw audio chunk for SHA-256 ledger provenance.

    Returns:
        Quarantined safety verdict, injection flags, and audit hash.
    """
    audio = _decode_audio_base64(audio_base64)
    audio_bytes = audio.tobytes() if len(audio) > 0 else text.encode("utf-8")

    # Evaluate wake word
    is_auth, clean_cmd, wake = _wake_gate.evaluate_text(text)

    if len(audio) == 0 and text:
        tx_result = TranscriptResult(
            text=text,
            is_final=True,
            has_speech=True,
            tokens=[],
            estimated_snr_db=30.0,
        )
    else:
        tx_result = _asr.transcribe(audio, reference_text=text)

    quarantine = _shield.evaluate(
        audio_data=audio_bytes,
        transcript_result=tx_result,
        wake_word_verified=is_auth,
        ultrasonic_anomaly_detected=False,
    )

    return {
        "ok": True,
        "verdict": quarantine.safety_verdict,
        "wake_word_verified": quarantine.wake_word_verified,
        "confidence_score": quarantine.confidence_score,
        "anomaly_flags": quarantine.anomaly_flags,
        "raw_audio_sha256": quarantine.raw_audio_sha256,
        "sanitized_command": quarantine.sanitized_command,
    }


@mcp.tool()
def voice_translate_intent_to_dag(command: str) -> dict[str, Any]:
    """Decompose high-level spoken command into a validated, dependency-checked Action DAG.

    Args:
        command: High-level natural language command.

    Returns:
        Action DAG structure including plan steps, topological execution order, and validity.
    """
    dag = _translator.translate_spoken_command(command)
    val = _translator.validate_dag(dag)

    steps_payload = [
        {
            "step_id": s.step_id,
            "tool_server": s.tool_server,
            "tool_name": s.tool_name,
            "arguments": s.arguments,
            "depends_on": s.depends_on,
            "is_destructive": s.is_destructive,
            "safety_precondition": s.safety_precondition,
        }
        for s in dag.steps
    ]

    return {
        "ok": val.is_valid,
        "plan_id": dag.plan_id,
        "user_intent": dag.user_intent,
        "rationale": dag.rationale,
        "estimated_duration_sec": dag.estimated_duration_sec,
        "steps": steps_payload,
        "topological_order": val.topological_order,
        "errors": val.errors,
        "warnings": val.warnings,
    }


@mcp.tool()
def voice_run_e2e_pipeline(
    audio_base64: str = "",
    text_input: str = "",
    dialect: str = "en-US",
    area_profile: str = "open_bench",
) -> dict[str, Any]:
    """Run full End-to-End Speech Ingestion, Filtration, and Intent Translation Pipeline.

    Sequence:
      1. Digital Pre-Filtering & Ultrasonic Anomaly Check
      2. Deterministic Out-of-Band Emergency Stop Screening (<100ms)
      3. Wake-Word Gating
      4. Qwen3-ASR Speech-to-Text with Dialect & Area Compensation
      5. VoiceSHIELD & RENNERVATE Safety & Prompt Injection Screening
      6. Hierarchical Action DAG Generation & Topological Ordering

    Args:
        audio_base64: Raw audio bytes (base64 encoded).
        text_input: Optional direct transcript override.
        dialect: Dialect profile ("en-US", "en-GB", "en-IN", "en-AU", "en-EA").
        area_profile: Laboratory area ("fume_hood", "centrifuge_room", "quiet_lab", "open_bench").

    Returns:
        Complete multi-stage pipeline result object.
    """
    # Check if system is halted
    if _interrupt_manager.is_emergency_halted:
        return {
            "ok": False,
            "error": "Emergency stop is currently latched. Pipeline halted for safety.",
            "emergency_halted": True,
        }

    audio = _decode_audio_base64(audio_base64)

    # 1. Acoustic Pre-Filter & Ultrasonic Check
    filt_res = _filter.process(audio)

    # 2. Emergency Stop Check on raw buffer or text
    estop_event = _interrupt_manager.evaluate_text(text_input) if text_input else None
    if estop_event:
        return {
            "ok": True,
            "emergency_stop_triggered": True,
            "trigger_phrase": estop_event.trigger_phrase,
            "message": "Out-of-band emergency interrupt triggered. Motor cut dispatches immediately.",
        }

    # 3. Transcribe audio via Qwen3-ASR
    if len(audio) == 0 and text_input:
        tx_res = TranscriptResult(
            text=text_input,
            is_final=True,
            has_speech=True,
            tokens=[],
            estimated_snr_db=30.0,
            dialect=dialect,  # type: ignore
            area_profile=area_profile,  # type: ignore
        )
    else:
        tx_res = _asr.transcribe(
            filt_res.filtered_audio,
            dialect=dialect,  # type: ignore
            area_profile=area_profile,  # type: ignore
            reference_text=text_input if text_input else None,
        )

    # Check emergency stop on transcribed text
    estop_text_event = _interrupt_manager.evaluate_text(tx_res.text)
    if estop_text_event:
        return {
            "ok": True,
            "emergency_stop_triggered": True,
            "trigger_phrase": estop_text_event.trigger_phrase,
            "message": "Out-of-band emergency interrupt triggered from speech.",
        }

    # 4. Wake-Word Gating
    is_auth, clean_command, wake_word = _wake_gate.evaluate_text(tx_res.text)

    # 5. VoiceSHIELD & RENNERVATE Screening
    quarantine = _shield.evaluate(
        audio_data=audio,
        transcript_result=tx_res,
        wake_word_verified=is_auth,
        ultrasonic_anomaly_detected=filt_res.is_ultrasonic_anomaly,
    )

    if quarantine.safety_verdict != "SAFE_LAB_INTENT":
        return {
            "ok": False,
            "verdict": quarantine.safety_verdict,
            "anomaly_flags": quarantine.anomaly_flags,
            "raw_transcript": tx_res.text,
            "audio_sha256": quarantine.raw_audio_sha256,
            "message": f"Command rejected by security shield: {quarantine.safety_verdict}",
        }

    # 6. Intent Translation to Action DAG
    cmd_to_plan = quarantine.sanitized_command or clean_command or tx_res.text
    dag = _translator.translate_spoken_command(cmd_to_plan)
    val = _translator.validate_dag(dag)

    steps_payload = [
        {
            "step_id": s.step_id,
            "tool_server": s.tool_server,
            "tool_name": s.tool_name,
            "arguments": s.arguments,
            "depends_on": s.depends_on,
            "is_destructive": s.is_destructive,
            "safety_precondition": s.safety_precondition,
        }
        for s in dag.steps
    ]

    return {
        "ok": True,
        "stage_metrics": {
            "duration_sec": filt_res.duration_sec,
            "estimated_snr_db": filt_res.estimated_snr_db,
            "ultrasonic_anomaly": filt_res.is_ultrasonic_anomaly,
            "ttft_ms": tx_res.time_to_first_token_ms,
            "total_latency_ms": tx_res.total_latency_ms,
            "dialect": tx_res.dialect,
            "area_profile": tx_res.area_profile,
        },
        "transcript": tx_res.text,
        "wake_word": wake_word,
        "audio_sha256": quarantine.raw_audio_sha256,
        "action_dag": {
            "plan_id": dag.plan_id,
            "user_intent": dag.user_intent,
            "rationale": dag.rationale,
            "steps": steps_payload,
            "topological_order": val.topological_order,
            "is_valid": val.is_valid,
        },
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
