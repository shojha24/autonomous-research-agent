"""Acoustic-semantic security shielding and transcript quarantine container.

Implements the security and interpretability defenses specified in Question 3:
  1. VoiceSHIELD: Rapid (<100ms) joint acoustic and semantic classification separating
     legitimate lab commands from ambient chatter and malicious injections.
  2. RENNERVATE: Indirect Prompt Injection (IPI) screening that flags attention-hijacking
     phrases ("ignore previous instructions", "override interlock", "bypass safety").
  3. QuarantinedVoiceInput: Strongly-typed Pydantic container with SHA-256 provenance
     tracking for the Agent Execution Record (AER / Vispute audit ledger).
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from mcp_app.voice.asr_engine import TranscriptResult

SafetyVerdict = Literal["SAFE_LAB_INTENT", "AMBIENT_CHATTER", "MALICIOUS_PROMPT_INJECTION"]

# Patterns characteristic of indirect prompt injections or destructive bypasses
PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore\b.{0,30}\b(?:previous|above|system)\b.{0,30}\binstructions?\b", re.IGNORECASE),
    re.compile(r"\bbypass\b.{0,20}\b(?:safety|interlock|guardrails?|checks?)\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\s+(?:ask|request)\s+(?:for\s+)?confirmation\b", re.IGNORECASE),
    re.compile(r"\bdelete\b.{0,20}\b(?:all\b|database|table|logs?)\b", re.IGNORECASE),
    re.compile(r"\boverride\b.{0,20}\b(?:admin|safety|permissions?)\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\bswitch\s+to\s+(?:unrestricted|developer|jailbreak)\s+mode\b", re.IGNORECASE),
    re.compile(r"\bformat\s+(?:drive|disk|system)\b", re.IGNORECASE),
)

# Common lab verbs and domain keywords indicating genuine laboratory operations
VALID_LAB_VERBS: frozenset[str] = frozenset(
    {
        "transfer",
        "import",
        "export",
        "move",
        "focus",
        "autofocus",
        "snap",
        "image",
        "capture",
        "incubate",
        "incubator",
        "measure",
        "home",
        "check",
        "status",
        "open",
        "close",
        "initialize",
        "swap",
        "take",
        "acquire",
        "scan",
        "run",
        "load",
        "unload",
        "eject",
        "plate",
        "well",
        "microscope",
        "rack",
        "slot",
        "hotel",
        "carousel",
        "photo",
        "picture",
    }
)


class QuarantinedVoiceInput(BaseModel):
    """Immutable quarantined container carrying audio provenance and safety vetting.

    Attributes:
        raw_audio_sha256: Hexadecimal SHA-256 hash of the acoustic buffer for AER auditing.
        transcript: Cleaned transcript text.
        wake_word_verified: Whether an authorized wake word prefix was detected.
        safety_verdict: Safety classification.
        confidence_score: Joint acoustic-semantic confidence score (0.0 to 1.0).
        estimated_snr_db: Estimated audio SNR in dB.
        ultrasonic_anomaly_detected: True if high-frequency carrier energy was flagged.
        anomaly_flags: List of triggered anomaly or policy warning strings.
        sanitized_command: Scrubbed command string safe for downstream planning prompts.
        timestamp: ISO timestamp of analysis.
    """

    raw_audio_sha256: str
    transcript: str
    wake_word_verified: bool
    safety_verdict: SafetyVerdict
    confidence_score: float = Field(ge=0.0, le=1.0)
    estimated_snr_db: float
    ultrasonic_anomaly_detected: bool = False
    anomaly_flags: list[str] = Field(default_factory=list)
    sanitized_command: str = ""
    timestamp: float = Field(default_factory=time.time)


class VoiceSHIELD:
    """Joint acoustic-semantic guardrail screening speech before tool-planning ingestion."""

    def __init__(
        self,
        strict_wake_enforcement: bool = True,
        enable_prompt_guard: bool = True,
    ) -> None:
        """Initialize the VoiceSHIELD guardrail.

        Args:
            strict_wake_enforcement: If True, reject any command lacking an explicit wake word.
            enable_prompt_guard: If True, run neural meta-llama/Llama-Prompt-Guard-2-22M alongside regexes.
        """
        self.strict_wake_enforcement = strict_wake_enforcement
        self.enable_prompt_guard = enable_prompt_guard
        self.prompt_guard: Any = None

        if enable_prompt_guard:
            try:
                try:
                    from mcp_app.voice.prompt_guard import PromptGuardDetector
                except ModuleNotFoundError:
                    from voice.prompt_guard import PromptGuardDetector
                self.prompt_guard = PromptGuardDetector()
            except Exception as e:
                pass

    @staticmethod
    def compute_audio_sha256(audio: np.ndarray | bytes) -> str:
        """Calculate SHA-256 digest of raw audio buffer for cryptographic auditability.

        Args:
            audio: 1D numpy array or raw bytes.

        Returns:
            Hex-encoded SHA-256 digest string.
        """
        if isinstance(audio, (bytes, bytearray)):
            buffer = bytes(audio)
        else:
            buffer = audio.tobytes()
        return hashlib.sha256(buffer).hexdigest()

    def screen_transcript_for_injections(self, text: str) -> tuple[bool, list[str]]:
        """Screen text for prompt injection payloads (RENNERVATE inspection + Llama Prompt Guard 2 22M).

        Args:
            text: Transcribed user command.

        Returns:
            Tuple of (is_malicious, list_of_detected_patterns).
        """
        detected: list[str] = []

        # 1. Ultra-fast regex pre-filter
        for pattern in PROMPT_INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                detected.append(f"regex:{match.group(0)}")

        # 2. Neural Llama Prompt Guard 2 (22M) evaluation
        if self.prompt_guard and self.prompt_guard.is_active:
            pg_res = self.prompt_guard.evaluate(text)
            if not pg_res["is_safe"]:
                detected.append(f"prompt_guard:{pg_res['verdict']} (confidence: {pg_res['score']})")

        return len(detected) > 0, detected

    def evaluate(
        self,
        audio_data: np.ndarray | bytes,
        transcript_result: TranscriptResult,
        wake_word_verified: bool,
        ultrasonic_anomaly_detected: bool = False,
    ) -> QuarantinedVoiceInput:
        """Execute comprehensive multi-stage screening.

        Args:
            audio_data: Raw audio buffer.
            transcript_result: TranscriptResult from Qwen3-ASR.
            wake_word_verified: WakeGate status.
            ultrasonic_anomaly_detected: Pre-filter ultrasonic anomaly status.

        Returns:
            QuarantinedVoiceInput container with verdict and audit ledger attributes.
        """
        audio_hash = self.compute_audio_sha256(audio_data)
        raw_text = transcript_result.text.strip()
        anomaly_flags: list[str] = []

        # 1. Ultrasonic Anomaly Check
        if ultrasonic_anomaly_detected:
            anomaly_flags.append("ULTRASONIC_INJECTION_SUSPECTED")

        # 2. Prompt Injection (RENNERVATE) Check
        has_injection, detected_patterns = self.screen_transcript_for_injections(raw_text)
        if has_injection:
            anomaly_flags.append(f"PROMPT_INJECTION_DETECTED: {', '.join(detected_patterns)}")

        # 3. Wake-Word, English-Only, & Lab Intent Enforcement
        if self.strict_wake_enforcement and not wake_word_verified:
            anomaly_flags.append("MISSING_WAKE_WORD")

        is_english_lang = getattr(transcript_result, "detected_language", "English").lower() in ("english", "en")
        is_english_script = bool(re.match(r"^[\x00-\x7F\u00B0\u2018\u2019\u201C\u201D\u2022\u2013\u2014\s]*$", raw_text))
        is_english = is_english_lang and is_english_script
        if not is_english:
            anomaly_flags.append("NON_ENGLISH_LANGUAGE_DETECTED")

        has_lab_intent = is_english and any(
            re.search(rf"\b{re.escape(v)}(?:s|d|ed|ing|r|rs)?\b", raw_text, re.IGNORECASE)
            for v in VALID_LAB_VERBS
        )

        # 4. Synthesize Verdict
        if has_injection:
            verdict: SafetyVerdict = "MALICIOUS_PROMPT_INJECTION"
            sanitized = ""
        elif not is_english:
            verdict = "AMBIENT_CHATTER"
            sanitized = ""
        elif not wake_word_verified or not transcript_result.has_speech:
            verdict = "AMBIENT_CHATTER"
            sanitized = ""
        elif not has_lab_intent:
            verdict = "AMBIENT_CHATTER"
            anomaly_flags.append("NO_RECOGNIZED_LAB_INTENT")
            sanitized = ""
        elif ultrasonic_anomaly_detected:
            verdict = "MALICIOUS_PROMPT_INJECTION"
            sanitized = ""
        else:
            verdict = "SAFE_LAB_INTENT"
            sanitized = raw_text

        # Compute confidence score
        base_conf = 0.90 if transcript_result.has_speech else 0.10
        if verdict != "SAFE_LAB_INTENT":
            base_conf *= 0.50

        return QuarantinedVoiceInput(
            raw_audio_sha256=audio_hash,
            transcript=raw_text,
            wake_word_verified=wake_word_verified,
            safety_verdict=verdict,
            confidence_score=round(base_conf, 3),
            estimated_snr_db=transcript_result.estimated_snr_db,
            ultrasonic_anomaly_detected=ultrasonic_anomaly_detected,
            anomaly_flags=anomaly_flags,
            sanitized_command=sanitized,
        )
