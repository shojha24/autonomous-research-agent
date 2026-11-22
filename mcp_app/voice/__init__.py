"""Voice ingestion, acoustic filtration, and hierarchical intent translation package.

Implements the end-to-end speech perception and safety gateway based on Questions 2 & 3
findings (Qwen3-ASR, 8th-order Butterworth LPF, ultrasonic anomaly defense, out-of-band
emergency interrupt, VoiceSHIELD safety screening, and hierarchical Action DAG translation).
"""

from __future__ import annotations

from mcp_app.voice.audio_filter import (
    AcousticFilterConfig,
    AcousticPreFilter,
    SyntheticNoiseGenerator,
)
from mcp_app.voice.wake_detector import AudioInterruptManager, WakeGate
from mcp_app.voice.asr_engine import (
    ASRConfig,
    DialectType,
    LabAreaProfile,
    Qwen3StreamingASR,
    TimedToken,
    TranscriptResult,
)
from mcp_app.voice.voice_shield import (
    QuarantinedVoiceInput,
    SafetyVerdict,
    VoiceSHIELD,
)
from mcp_app.voice.intent_translator import (
    ActionDAG,
    HierarchicalIntentTranslator,
    PlanStep,
)

__all__ = [
    "AcousticFilterConfig",
    "AcousticPreFilter",
    "SyntheticNoiseGenerator",
    "AudioInterruptManager",
    "WakeGate",
    "ASRConfig",
    "DialectType",
    "LabAreaProfile",
    "Qwen3StreamingASR",
    "TimedToken",
    "TranscriptResult",
    "QuarantinedVoiceInput",
    "SafetyVerdict",
    "VoiceSHIELD",
    "ActionDAG",
    "HierarchicalIntentTranslator",
    "PlanStep",
]
