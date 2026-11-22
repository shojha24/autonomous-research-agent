"""Ollama client: list models, stream chat (GUI), and chat for the tool loop.

Uses local HTTP API at 127.0.0.1:11434 by default.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
from typing import Callable, Iterator, Literal
from urllib import error, request

OllamaErrorKind = Literal["connection", "timeout", "model_not_found", "http_error"]


class OllamaClientError(RuntimeError):
    """Ollama HTTP failure with a classified kind for clearer error messages."""

    def __init__(self, message: str, *, kind: OllamaErrorKind) -> None:
        """Create an Ollama error with a category code.

        Args:
            message: Human-readable error text.
            kind: One of connection, timeout, model_not_found, http_error.
        """
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class OllamaConfig:
    """Connection settings for local Ollama HTTP API.

    Attributes:
        command: Ollama CLI name (kept for compatibility; HTTP is primary).
        model: Default model tag when none is passed per call.
        host: Base URL, e.g. http://127.0.0.1:11434.
        timeout_seconds: HTTP request timeout in seconds.
    """

    command: str
    model: str
    host: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 300.0


def _read_http_error_body(exc: error.HTTPError, *, max_len: int = 500) -> str:
    """Read and trim the body of an HTTP error response.

    Args:
        exc: HTTPError from urllib.
        max_len: Maximum characters to return.

    Returns:
        Decoded body text, possibly truncated.
    """
    try:
        raw = exc.read()
    except OSError:
        return ""
    text = raw.decode("utf-8", errors="replace").strip()
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text


def _parse_ollama_error_message(body: str) -> str:
    """Extract the error field from an Ollama JSON error body.

    Args:
        body: Raw HTTP response body text.

    Returns:
        Error string from JSON, or the raw body if not JSON.
    """
    if not body:
        return ""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, str):
            return err
    return body


def _classify_url_error(
    exc: error.URLError,
    *,
    host: str,
    model: str | None,
    timeout_seconds: float,
) -> OllamaClientError:
    """Map a urllib error to a helpful OllamaClientError.

    Args:
        exc: URLError or nested HTTPError from urllib.
        host: Ollama base URL used in the request.
        model: Model tag used in the request, if any.
        timeout_seconds: Timeout that was configured for the request.

    Returns:
        OllamaClientError with an actionable message and kind.
    """
    if isinstance(exc, error.HTTPError):
        body = _read_http_error_body(exc)
        detail = _parse_ollama_error_message(body) or body or str(exc.reason)
        if exc.code == 404 or (
            "not found" in detail.lower() and "model" in detail.lower()
        ):
            return OllamaClientError(
                f"Ollama model not found: {model!r} at {host} (HTTP {exc.code}). "
                f"Run `ollama list` and use an installed tag. Detail: {detail}",
                kind="model_not_found",
            )
        return OllamaClientError(
            f"Ollama HTTP {exc.code} at {host} (model={model!r}). Detail: {detail}",
            kind="http_error",
        )

    reason_text = str(exc.reason)
    if isinstance(exc.reason, (TimeoutError, socket.timeout)) or "timed out" in reason_text.lower():
        return OllamaClientError(
            f"Ollama timed out after {timeout_seconds}s at {host} (model={model!r}). "
            "Try a smaller model or longer timeout.",
            kind="timeout",
        )
    if isinstance(exc.reason, ConnectionRefusedError) or "Connection refused" in reason_text:
        return OllamaClientError(
            f"Cannot connect to Ollama at {host} (connection refused). "
            "Start Ollama or run `ollama serve`.",
            kind="connection",
        )
    return OllamaClientError(
        f"Ollama network error at {host} (model={model!r}): {reason_text}",
        kind="connection",
    )


class OllamaClient:
    """Talk to local Ollama over HTTP."""

    def __init__(self, config: OllamaConfig) -> None:
        """Store Ollama connection settings.

        Args:
            config: Host, default model, and timeout settings.
        """
        self._config = config

    def list_models(self) -> list[str]:
        """Return sorted list of locally installed Ollama model names.

        Returns:
            Sorted model name strings from GET /api/tags.
        """
        response = self._request_json("GET", "/api/tags")
        models = response.get("models", [])
        names = [str(item.get("name")) for item in models if item.get("name")]
        return sorted(names)

    def stream_chat(
        self,
        user_prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[dict[str, str]]:
        """Stream chat chunks as normalized UI events.

        Args:
            user_prompt: User message text.
            model: Override default model from config.
            system_prompt: Optional system message prepended to chat.
            should_stop: Callable returning True to stop between chunks.

        Yields:
            Dicts with keys type (thinking|answer|status) and text.

        Raises:
            OllamaClientError: On connection, timeout, or HTTP failures.
        """
        target_model = model or self._config.model
        messages: list[dict[str, str]] = [{"role": "user", "content": user_prompt}]
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        payload: dict[str, object] = {
            "model": target_model,
            "stream": True,
            "messages": messages,
        }

        yield {"type": "status", "text": f"Streaming from model `{target_model}`..."}
        req = request.Request(
            f"{self._config.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self._config.timeout_seconds) as resp:
                for raw_line in resp:
                    if should_stop and should_stop():
                        yield {"type": "status", "text": "Stream stopped by user."}
                        return
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    message = chunk.get("message", {})
                    thinking = message.get("thinking")
                    if thinking:
                        yield {"type": "thinking", "text": str(thinking)}
                    content = message.get("content")
                    if content:
                        yield {"type": "answer", "text": str(content)}
                    if chunk.get("done"):
                        eval_ms = chunk.get("eval_duration")
                        if eval_ms is not None:
                            yield {"type": "status", "text": f"Completed. eval_duration={eval_ms}"}
                        else:
                            yield {"type": "status", "text": "Completed."}
                        return
        except error.URLError as exc:
            raise _classify_url_error(
                exc,
                host=self._config.host,
                model=target_model,
                timeout_seconds=self._config.timeout_seconds,
            ) from exc

    def chat_messages(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Run one non-streaming chat completion (used by MCP tool loop).

        Args:
            messages: Full chat history with role and content keys.
            model: Override default model from config.
            temperature: Ollama sampling temperature (0.0–1.0 typical).

        Returns:
            Assistant reply text from the response.

        Raises:
            OllamaClientError: On connection, timeout, or HTTP failures.
            RuntimeError: Response missing assistant content.
        """
        target_model = model or self._config.model
        payload: dict[str, object] = {
            "model": target_model,
            "stream": False,
            "messages": messages,
            "options": {"temperature": temperature},
        }
        req = request.Request(
            f"{self._config.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self._config.timeout_seconds) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except error.URLError as exc:
            raise _classify_url_error(
                exc,
                host=self._config.host,
                model=target_model,
                timeout_seconds=self._config.timeout_seconds,
            ) from exc

        message = raw.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Ollama response missing assistant content.")
        return content

    def _request_json(self, method: str, path: str) -> dict[str, object]:
        """Issue an HTTP request and parse JSON response.

        Args:
            method: HTTP method, e.g. GET.
            path: API path appended to host, e.g. /api/tags.

        Returns:
            Parsed JSON object from the response body.

        Raises:
            OllamaClientError: On network or HTTP failures.
            RuntimeError: Response is not a JSON object.
        """
        req = request.Request(f"{self._config.host}{path}", method=method)
        try:
            with request.urlopen(req, timeout=self._config.timeout_seconds) as resp:
                payload = resp.read().decode("utf-8")
        except error.URLError as exc:
            raise _classify_url_error(
                exc,
                host=self._config.host,
                model=self._config.model,
                timeout_seconds=self._config.timeout_seconds,
            ) from exc
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise RuntimeError("Unexpected Ollama API response shape.")
        return parsed
