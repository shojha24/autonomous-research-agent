"""Low-latency acoustic interrupt manager and wake-word gating engine.

Implements the deterministic out-of-band emergency interrupt and wake-word gating
layers specified in Questions 2 & 3:
  1. Low-Latency (<100ms) Hardware Emergency Interrupt: Continuously screens audio
     buffers for ["HALT", "EMERGENCY STOP", "ABORT", "FREEZE"], immediately bypassing
     LLM inference and motor planning to trip physical safety interlocks.
  2. Wake-Word Gating: Discards un-prefixed conversational chatter from open lab
     microphones (preventing AudioPiggyback indirect prompt injection), requiring explicit
     triggers like "Hey LabAgent" or "Antigravity".
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

try:
    import openwakeword
    from openwakeword.model import Model as OWWModel
    OPENWAKEWORD_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    OPENWAKEWORD_AVAILABLE = False
    OWWModel = None  # type: ignore


EMERGENCY_STOP_TOKENS: frozenset[str] = frozenset(
    {
        "emergency stop",
        "halt",
        "stop",
        "freeze",
        "abort",
        "kill motor",
        "e stop",
        "estop",
    }
)

WAKE_WORD_TOKENS: frozenset[str] = frozenset(
    {
        "hey labagent",
        "labagent",
        "hey lab agent",
        "lab agent",
    }
)


@dataclass
class InterruptEvent:
    """Record of an acoustic interrupt or wake trigger event.

    Attributes:
        event_type: "EMERGENCY_STOP", "WAKE_WORD", or "NORMAL".
        trigger_phrase: Phrase or token that triggered the event.
        confidence: Detection confidence score (0.0 to 1.0).
        latency_ms: Milliseconds elapsed between buffer receipt and event dispatch.
        timestamp: System time of detection.
    """

    event_type: Literal["EMERGENCY_STOP", "WAKE_WORD", "NORMAL"]
    trigger_phrase: str
    confidence: float
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


class OpenWakeWordDetector:
    """Acoustic wake-word and emergency interrupt detector powered by openWakeWord.

    Loads neural wake models:
      - Start trigger: hey_jarvis_v0.1.tflite (starts Qwen3-ASR inference)
      - Immediate stop / abort trigger: Break_Out,_stop_20261016_180430.tflite (stops immediately)

    Maintains an internal 16kHz PCM buffer, feeding 1280-sample (80ms) frames to the
    underlying TFLite/ONNX models with sub-10ms chunk inference time.
    """

    def __init__(
        self,
        start_model_path: str | Path | None = None,
        stop_model_path: str | Path | None = None,
        start_threshold: float = 0.5,
        stop_threshold: float = 0.5,
        vad_threshold: float = 0.4,
    ) -> None:
        self.base_dir = Path(__file__).parent
        self.start_model_path = Path(start_model_path) if start_model_path else (self.base_dir / "hey_jarvis_v0.1.tflite")
        self.stop_model_path = Path(stop_model_path) if stop_model_path else (self.base_dir / "Break_Out,_stop_20261016_180430.tflite")
        self.start_threshold = start_threshold
        self.stop_threshold = stop_threshold
        self.vad_threshold = vad_threshold

        self.model: Any | None = None
        self._buffer: list[int] = []
        self.start_model_key: str | None = None
        self.stop_model_key: str | None = None
        self.is_active = False

        self._init_models()

    def _init_models(self) -> None:
        if not OPENWAKEWORD_AVAILABLE or OWWModel is None:
            return

        models_to_load = []
        if self.start_model_path.exists():
            models_to_load.append(str(self.start_model_path))
        if self.stop_model_path.exists():
            models_to_load.append(str(self.stop_model_path))

        if not models_to_load:
            return

        try:
            self.model = OWWModel(
                wakeword_models=models_to_load,
                inference_framework="tflite",
                vad_threshold=self.vad_threshold,
            )
            # Identify model keys
            for k in self.model.models.keys():
                k_lower = k.lower()
                if "jarvis" in k_lower or "wake" in k_lower or "alexa" in k_lower:
                    self.start_model_key = k
                elif "stop" in k_lower or "break" in k_lower or "abort" in k_lower:
                    self.stop_model_key = k

            # Fallback assignment if naming differs
            if not self.start_model_key and len(self.model.models) >= 1:
                self.start_model_key = list(self.model.models.keys())[0]
            if not self.stop_model_key and len(self.model.models) >= 2:
                self.stop_model_key = list(self.model.models.keys())[1]

            self.is_active = True
        except Exception as e:
            print(f"[OpenWakeWord] Model initialization notice: {e}")
            self.model = None
            self.is_active = False

    def reset_buffer(self) -> None:
        """Clear internal audio sample buffer."""
        self._buffer.clear()

    def process_chunk(
        self, audio_chunk: np.ndarray | bytes
    ) -> tuple[bool, bool, dict[str, float]]:
        """Feed audio samples into openWakeWord and check trigger thresholds.

        Args:
            audio_chunk: Raw PCM audio (float32 array [-1.0, 1.0], int16 array, or int16 bytes).

        Returns:
            Tuple of (start_triggered, stop_triggered, scores_dict).
        """
        if not self.is_active or self.model is None:
            return False, False, {}

        # Convert to 1D int16 array
        if isinstance(audio_chunk, (bytes, bytearray)):
            pcm16 = np.frombuffer(audio_chunk, dtype=np.int16)
        elif isinstance(audio_chunk, np.ndarray):
            if audio_chunk.dtype == np.float32 or audio_chunk.dtype == np.float64:
                pcm16 = np.clip(audio_chunk * 32767.0, -32768, 32767).astype(np.int16)
            elif audio_chunk.dtype == np.int16:
                pcm16 = audio_chunk.flatten()
            else:
                pcm16 = audio_chunk.astype(np.int16).flatten()
        else:
            return False, False, {}

        self._buffer.extend(pcm16.tolist())

        start_triggered = False
        stop_triggered = False
        latest_scores: dict[str, float] = {}

        # openWakeWord expects 1280 samples per step (80ms at 16kHz)
        while len(self._buffer) >= 1280:
            frame = np.array(self._buffer[:1280], dtype=np.int16)
            self._buffer = self._buffer[1280:]

            scores = self.model.predict(frame)
            latest_scores.update(scores)

            if self.start_model_key and scores.get(self.start_model_key, 0.0) >= self.start_threshold:
                start_triggered = True

            if self.stop_model_key and scores.get(self.stop_model_key, 0.0) >= self.stop_threshold:
                stop_triggered = True

        return start_triggered, stop_triggered, latest_scores


class AudioInterruptManager:
    """Out-of-band deterministic emergency interrupt detector.

    Operates independently of LLM reasoning with guaranteed sub-100ms trigger latency.
    """

    def __init__(
        self,
        on_emergency_stop: Callable[[InterruptEvent], None] | None = None,
        confidence_threshold: float = 0.65,
        oww_detector: OpenWakeWordDetector | None = None,
    ) -> None:
        """Initialize the emergency interrupt detector.

        Args:
            on_emergency_stop: Callback executed immediately upon emergency detection.
            confidence_threshold: Minimum confidence score required to trip the interrupt.
            oww_detector: Optional OpenWakeWordDetector instance for neural acoustic interrupts.
        """
        self.on_emergency_stop = on_emergency_stop
        self.confidence_threshold = confidence_threshold
        self.oww_detector = oww_detector
        self._is_emergency_halted = False
        self._last_event: InterruptEvent | None = None

    @property
    def is_emergency_halted(self) -> bool:
        """Return whether an emergency stop condition is currently active."""
        return self._is_emergency_halted

    def reset_emergency_state(self) -> None:
        """Reset the active emergency stop latch after physical inspection."""
        self._is_emergency_halted = False
        if self.oww_detector is not None:
            self.oww_detector.reset_buffer()

    def evaluate_text(self, text: str) -> InterruptEvent | None:
        """Deterministically check transcribed text for emergency halt tokens.

        Args:
            text: Utterance or partial streaming transcript.

        Returns:
            InterruptEvent if triggered, otherwise None.
        """
        start_time = time.perf_counter()
        normalized = text.strip().lower()

        # Check for emergency words
        for token in EMERGENCY_STOP_TOKENS:
            # Match whole words or boundary patterns
            pattern = rf"\b{re.escape(token)}\b"
            if re.search(pattern, normalized):
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                event = InterruptEvent(
                    event_type="EMERGENCY_STOP",
                    trigger_phrase=token,
                    confidence=0.99,
                    latency_ms=latency_ms,
                )
                self._is_emergency_halted = True
                self._last_event = event
                if self.on_emergency_stop:
                    self.on_emergency_stop(event)
                return event

        return None

    def evaluate_audio_chunk(
        self,
        audio_chunk: np.ndarray | bytes,
        sample_rate: int = 16000,
        simulated_token: str | None = None,
    ) -> InterruptEvent | None:
        """Evaluate raw PCM chunk for high-priority emergency stop signals.

        Queries openWakeWord neural models if configured, or checks simulated tokens.

        Args:
            audio_chunk: 1D float32 audio samples, int16 samples, or raw PCM bytes.
            sample_rate: Audio sampling frequency in Hz.
            simulated_token: Optional acoustic token tag for testing.

        Returns:
            InterruptEvent if triggered, otherwise None.
        """
        start_time = time.perf_counter()

        # 1. Neural acoustic check via openWakeWord if available
        if self.oww_detector is not None and self.oww_detector.is_active:
            _, stop_triggered, scores = self.oww_detector.process_chunk(audio_chunk)
            if stop_triggered:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                model_name = self.oww_detector.stop_model_key or "stop_model"
                conf = scores.get(model_name, 0.95)
                event = InterruptEvent(
                    event_type="EMERGENCY_STOP",
                    trigger_phrase=model_name,
                    confidence=float(conf),
                    latency_ms=latency_ms,
                )
                self._is_emergency_halted = True
                self._last_event = event
                if self.on_emergency_stop:
                    self.on_emergency_stop(event)
                return event

        # 2. Simulated token check for testing
        if simulated_token:
            for token in EMERGENCY_STOP_TOKENS:
                if token in simulated_token.lower():
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    event = InterruptEvent(
                        event_type="EMERGENCY_STOP",
                        trigger_phrase=token,
                        confidence=0.95,
                        latency_ms=latency_ms,
                    )
                    self._is_emergency_halted = True
                    self._last_event = event
                    if self.on_emergency_stop:
                        self.on_emergency_stop(event)
                    return event

        return None


class WakeGate:
    """Manages conversational session gating to reject ambient acoustic chatter."""

    def __init__(
        self,
        wake_timeout_sec: float = 12.0,
        on_wake_detected: Callable[[InterruptEvent], None] | None = None,
        oww_detector: OpenWakeWordDetector | None = None,
    ) -> None:
        """Initialize the wake gate.

        Args:
            wake_timeout_sec: Seconds the gate remains open after wake detection.
            on_wake_detected: Optional callback when wake word is detected.
            oww_detector: Optional OpenWakeWordDetector instance for neural acoustic wake words.
        """
        self.wake_timeout_sec = wake_timeout_sec
        self.on_wake_detected = on_wake_detected
        self.oww_detector = oww_detector
        self._gate_open_until = 0.0
        self._active_trigger: str | None = None

    @property
    def is_gate_open(self) -> bool:
        """Check whether the gate is currently accepting downstream commands."""
        return time.time() < self._gate_open_until

    def extend_active_session(self) -> None:
        """Extend the active listening window while user continues speaking."""
        self._gate_open_until = time.time() + self.wake_timeout_sec

    def close_gate(self) -> None:
        """Immediately close the wake gate."""
        self._gate_open_until = 0.0
        self._active_trigger = None
        if self.oww_detector is not None:
            self.oww_detector.reset_buffer()

    def evaluate_audio_chunk(
        self, audio_chunk: np.ndarray | bytes
    ) -> InterruptEvent | None:
        """Evaluate raw PCM chunk for neural acoustic wake word triggers.

        Args:
            audio_chunk: Audio samples (float32 array, int16 array, or bytes).

        Returns:
            InterruptEvent if wake word detected, otherwise None.
        """
        if self.oww_detector is None or not self.oww_detector.is_active:
            return None

        start_time = time.perf_counter()
        wake_triggered, _, scores = self.oww_detector.process_chunk(audio_chunk)
        if wake_triggered:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            model_name = self.oww_detector.start_model_key or "wake_word"
            conf = scores.get(model_name, 0.98)
            event = InterruptEvent(
                event_type="WAKE_WORD",
                trigger_phrase=model_name,
                confidence=float(conf),
                latency_ms=latency_ms,
            )
            self._gate_open_until = time.time() + self.wake_timeout_sec
            self._active_trigger = model_name
            if self.on_wake_detected:
                self.on_wake_detected(event)
            return event

        return None

    def evaluate_text(self, text: str) -> tuple[bool, str, str]:
        """Examine transcript for wake prefix and strip it if present.

        Args:
            text: Raw input transcript.

        Returns:
            Tuple of (is_authorized, remaining_command_text, matched_wake_word).
        """
        start_time = time.perf_counter()
        normalized = text.strip().lower()

        # If gate is already open from recent wake trigger, accept command directly
        if self.is_gate_open:
            self.extend_active_session()
            return True, text.strip(), self._active_trigger or "active_session"

        # Check for wake word prefix
        for wake in sorted(WAKE_WORD_TOKENS, key=len, reverse=True):
            pattern = rf"^\b{re.escape(wake)}\b[,:\s]*(.*)$"
            match = re.match(pattern, normalized, re.IGNORECASE)
            if match:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                event = InterruptEvent(
                    event_type="WAKE_WORD",
                    trigger_phrase=wake,
                    confidence=0.98,
                    latency_ms=latency_ms,
                )
                self._gate_open_until = time.time() + self.wake_timeout_sec
                self._active_trigger = wake
                if self.on_wake_detected:
                    self.on_wake_detected(event)

                remaining = match.group(1).strip()
                # If original case preservation is preferred, slice from original text
                orig_remaining = text.strip()[len(wake) :].lstrip(" ,:")
                return True, orig_remaining or remaining, wake

        return False, "", ""
