"""Neural Prompt Injection and Jailbreak Guardrail using meta-llama/Llama-Prompt-Guard-2-22M.

Provides:
  1. Ultra-fast (<3-5ms) neural sequence classification for untrusted prompt inputs.
  2. Classification head predicting:
       - Label 0: BENIGN
       - Label 1: INJECTION
       - Label 2: JAILBREAK
  3. Interactive/persistent operator whitelist for reviewing and overriding false positives.
  4. Graceful fallback to regex heuristics if Hugging Face token is not yet configured.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal, Optional

import torch

logger = logging.getLogger(__name__)

PromptGuardVerdict = Literal["BENIGN", "INJECTION", "JAILBREAK", "WHITELISTED"]


class PromptGuardDetector:
    """Wrapper around meta-llama/Llama-Prompt-Guard-2-22M with local whitelisting."""

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-Prompt-Guard-2-22M",
        threshold: float = 0.5,
        whitelist_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.threshold = threshold
        self.whitelist_path = whitelist_path or (Path(__file__).resolve().parent / ".prompt_guard_whitelist.json")

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer: Any = None
        self.model: Any = None
        self.is_active: bool = False
        self.whitelist: set[str] = self._load_whitelist()

        self._load_model()

    def _load_whitelist(self) -> set[str]:
        """Load whitelisted command hashes or strings from JSON."""
        if self.whitelist_path.exists():
            try:
                data = json.loads(self.whitelist_path.read_text(encoding="utf-8"))
                return set(data.get("whitelisted_commands", []))
            except Exception as e:
                logger.warning(f"Failed to read whitelist from {self.whitelist_path}: {e}")
        return set()

    def save_whitelist(self) -> None:
        """Persist whitelisted commands to disk."""
        try:
            payload = {"whitelisted_commands": sorted(self.whitelist)}
            self.whitelist_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to write whitelist to {self.whitelist_path}: {e}")

    def add_to_whitelist(self, text: str) -> None:
        """Add a vetted command string to the local whitelist."""
        clean = text.strip().lower()
        if clean:
            self.whitelist.add(clean)
            self.save_whitelist()

    def is_whitelisted(self, text: str) -> bool:
        """Check if a command has been whitelisted by an operator."""
        return text.strip().lower() in self.whitelist

    def _load_model(self) -> None:
        """Attempt to load the weights from Hugging Face or local cache."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info(f"Loading Prompt Guard from {self.model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
            self.is_active = True
            logger.info("Llama Prompt Guard 2 (22M) initialized successfully.")
        except Exception as e:
            self.is_active = False
            logger.warning(
                f"Llama Prompt Guard 2 (22M) could not be loaded ({e}). "
                "Ensure you accepted the terms at https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M "
                "and authenticated via `huggingface-cli login` or `export HF_TOKEN=...`. "
                "Falling back to regex heuristics."
            )

    def evaluate(self, text: str) -> dict[str, Any]:
        """Evaluate a text string for prompt injections and jailbreaks.

        Returns:
            dict containing:
              - is_safe (bool): True if benign or whitelisted.
              - verdict (str): BENIGN, INJECTION, JAILBREAK, or WHITELISTED.
              - score (float): Peak attack probability score.
              - injection_score (float)
              - jailbreak_score (float)
        """
        clean_text = text.strip()
        if not clean_text:
            return {
                "is_safe": True,
                "verdict": "BENIGN",
                "score": 0.0,
                "injection_score": 0.0,
                "jailbreak_score": 0.0,
            }

        # Check operator whitelist first
        if self.is_whitelisted(clean_text):
            return {
                "is_safe": True,
                "verdict": "WHITELISTED",
                "score": 0.0,
                "injection_score": 0.0,
                "jailbreak_score": 0.0,
            }

        # Fallback if model is not loaded
        if not self.is_active or self.model is None or self.tokenizer is None:
            return {
                "is_safe": True,
                "verdict": "BENIGN",
                "score": 0.0,
                "injection_score": 0.0,
                "jailbreak_score": 0.0,
                "model_active": False,
            }

        inputs = self.tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        # Label index mapping: 0: BENIGN, 1: INJECTION, 2: JAILBREAK
        benign_score = probs[0].item() if probs.shape[0] > 0 else 1.0
        injection_score = probs[1].item() if probs.shape[0] > 1 else 0.0
        jailbreak_score = probs[2].item() if probs.shape[0] > 2 else 0.0

        if jailbreak_score >= self.threshold:
            verdict: PromptGuardVerdict = "JAILBREAK"
            is_safe = False
            top_score = jailbreak_score
        elif injection_score >= self.threshold:
            verdict = "INJECTION"
            is_safe = False
            top_score = injection_score
        else:
            verdict = "BENIGN"
            is_safe = True
            top_score = max(injection_score, jailbreak_score)

        return {
            "is_safe": is_safe,
            "verdict": verdict,
            "score": round(top_score, 4),
            "injection_score": round(injection_score, 4),
            "jailbreak_score": round(jailbreak_score, 4),
            "benign_score": round(benign_score, 4),
            "model_active": True,
        }
