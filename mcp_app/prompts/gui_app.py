"""Tkinter GUI for local Ollama chat and MCP agent runs.

Workers run Ollama/MCP in background threads. The main thread reads a queue
every 50ms and updates widgets (Tk is not thread-safe).
"""

from __future__ import annotations  # lets us use types before they are defined

import asyncio  # needed to run the async MCP loop in a worker thread
import os  # read default model/command from environment
import sys  # pass Python path when spawning MCP server subprocess
import threading  # run chat and MCP without freezing the window
from queue import Empty, Queue  # safe handoff between worker threads and UI
from tkinter import messagebox  # yes/no popup before running a tool
import tkinter as tk  # base GUI toolkit
from tkinter import ttk  # nicer-looking widgets

from mcp_host.mcp_tool_loop import (  # shared Ollama + MCP agent loop
    ToolLoopParams,
    ToolLoopResult,
    format_confirm_message,
    run_tool_loop,
)
from mcp_host.ollama_client import OllamaClient, OllamaConfig  # talk to local Ollama


class OllamaGuiApp:
    """Main window: streamed chat plus MCP tool loop with confirmations."""

    def __init__(self, root: tk.Tk, client: OllamaClient) -> None:
        """Build widgets, load models, and start the event poll loop.

        Args:
            root: Tk root window for this application.
            client: OllamaClient used for model list and chat/MCP calls.
        """
        self._root = root  # the main Tk window
        self._client = client  # wrapper around Ollama CLI/HTTP

        # --- Threading state ---
        self._event_queue: Queue[dict[str, object]] = Queue()  # worker -> UI messages
        self._mcp_confirm_reply_queue: Queue[bool] = Queue()  # UI -> worker yes/no
        self._worker: threading.Thread | None = None  # chat stream thread
        self._mcp_worker: threading.Thread | None = None  # MCP agent thread
        self._stop_event = threading.Event()  # user clicked stop during chat
        self._mcp_abort_event = threading.Event()  # user clicked stop during MCP
        self._is_streaming = False  # chat currently running
        self._mcp_running = False  # MCP currently running

        # --- Form fields (bound to widgets) ---
        self._model_var = tk.StringVar(value=client._config.model)  # selected Ollama model
        self._status_var = tk.StringVar(value="Idle")  # short status in header
        self._system_var = tk.StringVar(value="Be concise, operational, and safety-minded.")
        self._port_var = tk.StringVar(value=os.getenv("INCUBATOR_SERIAL_PORT", "COM1"))
        self._mock_mcp_var = tk.BooleanVar(value=False)  # skip real hardware
        self._max_steps_var = tk.StringVar(value="8")  # max MCP tool-loop steps
        self._error_var = tk.StringVar(value="")  # red error line at bottom

        self._build_ui()  # draw all widgets
        self._refresh_models()  # fill model dropdown from Ollama
        self._root.after(50, self._drain_events)  # poll queue every 50ms

    def _build_ui(self) -> None:
        """Create and pack all labels, inputs, panes, and buttons.

        Returns:
            None.
        """
        self._root.title("Bioagent Ollama Console")
        self._root.geometry("980x760")

        frame = ttk.Frame(self._root, padding=10)  # main container
        frame.pack(fill=tk.BOTH, expand=True)

        # --- Header: model picker + status ---
        header = ttk.Frame(frame)
        header.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(header, text="Model").pack(side=tk.LEFT)
        self._model_combo = ttk.Combobox(
            header,
            textvariable=self._model_var,
            width=28,
            state="readonly",  # user picks from list only
        )
        self._model_combo.pack(side=tk.LEFT, padx=(8, 8))
        ttk.Button(header, text="Refresh Models", command=self._refresh_models).pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self._status_var).pack(side=tk.RIGHT)

        # --- System prompt row ---
        system_row = ttk.Frame(frame)
        system_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(system_row, text="System prompt").pack(side=tk.LEFT)
        ttk.Entry(system_row, textvariable=self._system_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0)
        )

        # --- Output panes: trace on top, answer below ---
        panes = ttk.PanedWindow(frame, orient=tk.VERTICAL)
        panes.pack(fill=tk.BOTH, expand=True)

        thinking_container = ttk.Labelframe(panes, text="Thinking / MCP trace")
        self._thinking_text = tk.Text(thinking_container, wrap=tk.WORD, height=10, state=tk.DISABLED)
        self._thinking_text.pack(fill=tk.BOTH, expand=True)
        panes.add(thinking_container, weight=1)  # smaller pane

        answer_container = ttk.Labelframe(panes, text="Answer")
        self._answer_text = tk.Text(answer_container, wrap=tk.WORD, height=12, state=tk.DISABLED)
        self._answer_text.pack(fill=tk.BOTH, expand=True)
        panes.add(answer_container, weight=2)  # bigger pane

        # --- MCP settings ---
        mcp_row = ttk.Labelframe(frame, text="MCP agent (Ollama JSON tool loop)")
        mcp_row.pack(fill=tk.X, pady=(8, 0))
        inner = ttk.Frame(mcp_row)
        inner.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(inner, text="Serial port").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(inner, textvariable=self._port_var, width=22).grid(row=0, column=1, padx=(6, 16))
        ttk.Checkbutton(
            inner,
            text="Mock tools (no real incubator/microscope hardware)",
            variable=self._mock_mcp_var,
        ).grid(row=0, column=2, sticky=tk.W)
        ttk.Label(inner, text="Max steps").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Entry(inner, textvariable=self._max_steps_var, width=8).grid(
            row=1, column=1, sticky=tk.W, padx=(6, 0), pady=(6, 0)
        )
        ttk.Label(
            inner,
            text="Stop / Abort ends streaming or cancels MCP between steps.",
        ).grid(row=1, column=2, sticky=tk.W, padx=(16, 0), pady=(6, 0))

        # --- Error line ---
        err_row = ttk.Frame(frame)
        err_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(err_row, text="Errors").pack(side=tk.LEFT)
        ttk.Label(err_row, textvariable=self._error_var, foreground="#a00").pack(side=tk.LEFT, padx=(8, 0))

        # --- Shared prompt box (used by both chat and MCP) ---
        prompt_container = ttk.Labelframe(frame, text="Prompt (used for chat Send and MCP Run)")
        prompt_container.pack(fill=tk.BOTH, expand=False, pady=(8, 0))
        self._prompt_text = tk.Text(prompt_container, wrap=tk.WORD, height=6)
        self._prompt_text.pack(fill=tk.BOTH, expand=True)

        # --- Action buttons ---
        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X, pady=(8, 0))
        self._send_button = ttk.Button(controls, text="Send (stream chat)", command=self._start_stream)
        self._send_button.pack(side=tk.LEFT)
        self._run_mcp_button = ttk.Button(controls, text="Run MCP agent", command=self._start_mcp_agent)
        self._run_mcp_button.pack(side=tk.LEFT, padx=(8, 0))
        self._stop_button = ttk.Button(controls, text="Stop / Abort", command=self._stop_all, state=tk.DISABLED)
        self._stop_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Clear", command=self._clear_outputs).pack(side=tk.LEFT, padx=(8, 0))

    def _refresh_models(self) -> None:
        """Fetch local Ollama models and populate the model dropdown.

        Returns:
            None. Sets error/status vars on failure.
        """
        try:
            models = self._client.list_models()  # ask Ollama what is installed
        except Exception as exc:  # noqa: BLE001
            self._error_var.set(f"Model fetch failed: {exc}")
            self._status_var.set("Model list error")
            return

        if not models:  # Ollama returned nothing
            self._status_var.set("No local models found.")
            return

        self._model_combo["values"] = models  # fill dropdown
        if self._model_var.get() not in models:  # old selection gone
            self._model_var.set(models[0])  # pick first model
        self._error_var.set("")
        self._status_var.set(f"Ready ({self._model_var.get()})")

    def _get_prompt(self, empty_message: str = "Enter a prompt first.") -> str | None:
        """Read and validate the shared prompt text box.

        Args:
            empty_message: Error string shown when prompt is empty.

        Returns:
            Trimmed prompt text, or None if empty (error var is set).
        """
        prompt = self._prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            self._error_var.set(empty_message)
            return None
        return prompt

    def _get_model(self) -> str | None:
        """Read and validate the selected Ollama model name.

        Returns:
            Trimmed model name, or None if missing (error var is set).
        """
        model = self._model_var.get().strip()
        if not model:
            self._error_var.set("Select a model first.")
            return None
        return model

    def _busy(self) -> bool:
        """Check whether chat streaming or MCP agent is already running.

        Returns:
            True if either run is in progress.
        """
        return self._is_streaming or self._mcp_running

    def _set_running_ui(self, *, streaming: bool, mcp: bool, status: str) -> None:
        """Update flags and disable start buttons while a run is active.

        Args:
            streaming: Whether chat streaming is active.
            mcp: Whether MCP agent is active.
            status: Text for the header status label.

        Returns:
            None.
        """
        self._is_streaming = streaming
        self._mcp_running = mcp
        self._error_var.set("")
        self._send_button.configure(state=tk.DISABLED)
        self._run_mcp_button.configure(state=tk.DISABLED)
        self._stop_button.configure(state=tk.NORMAL)
        self._status_var.set(status)

    def _start_stream(self) -> None:
        """Start streamed Ollama chat in a background thread.

        Returns:
            None. Posts stream events to _event_queue for the main thread.
        """
        if self._busy():
            self._error_var.set("Wait for the current run to finish or press Stop.")
            return

        prompt = self._get_prompt()
        model = self._get_model()
        if prompt is None or model is None:
            return

        self._stop_event.clear()  # fresh run
        self._set_running_ui(streaming=True, mcp=False, status=f"Streaming ({model})...")
        self._append_text(self._thinking_text, f"\n[stream] model={model}\n")

        def worker() -> None:
            """Stream from Ollama and push events to the main thread.

            Returns:
                None. Always posts a done event in finally.
            """
            try:
                for event in self._client.stream_chat(
                    user_prompt=prompt,
                    model=model,
                    system_prompt=self._system_var.get().strip() or None,
                    should_stop=self._stop_event.is_set,  # checked between chunks
                ):
                    self._event_queue.put(event)  # main thread will render it
            except Exception as exc:  # noqa: BLE001
                self._event_queue.put({"type": "status", "text": f"Error: {exc}"})
            finally:
                self._event_queue.put({"type": "done"})  # always reset UI

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _wait_mcp_confirm(self, tool_name: str, arguments: dict[str, object]) -> bool:
        """Ask the user on the main thread whether to run a tool.

        Called from the MCP worker thread. Blocks until the user clicks Yes/No.

        Args:
            tool_name: MCP tool the model wants to call.
            arguments: JSON arguments for that tool.

        Returns:
            True if the user approved the call.
        """
        # Tell main thread to show the dialog.
        self._event_queue.put(
            {"type": "mcp_confirm", "tool": tool_name, "arguments": arguments}
        )
        # Wait here until main thread puts True/False in the reply queue.
        return bool(self._mcp_confirm_reply_queue.get())

    def _start_mcp_agent(self) -> None:
        """Start the MCP tool loop in a background thread.

        Returns:
            None. Posts mcp_* events to _event_queue for the main thread.
        """
        if self._busy():
            self._error_var.set("Wait for the current run to finish or press Stop.")
            return

        prompt = self._get_prompt(empty_message="Enter a prompt (objective) first.")
        model = self._get_model()
        if prompt is None or model is None:
            return

        try:
            max_steps = max(1, int(self._max_steps_var.get().strip()))
        except ValueError:
            self._error_var.set("Max steps must be an integer.")
            return

        self._mcp_abort_event.clear()  # fresh run
        self._set_running_ui(streaming=False, mcp=True, status=f"MCP agent ({model})…")

        # Settings passed into the shared tool loop.
        params = ToolLoopParams(
            prompt=prompt,
            max_steps=max_steps,
            port=self._port_var.get().strip() or "COM1",
            mock_tools=bool(self._mock_mcp_var.get()),
            model_name=model,
            ollama_command=self._client._config.command,
            ollama_host=self._client._config.host,
            timeout_seconds=self._client._config.timeout_seconds,
            microscope_xy_port=os.getenv("MICROSCOPE_XY_PORT", "COM3"),
            microscope_kcube_serial=os.getenv("MICROSCOPE_KCUBE_SERIAL", "26004971"),
            microscope_camera_serial=os.getenv("MICROSCOPE_CAMERA_SERIAL", "25267"),
        )

        def trace(msg: str) -> None:
            """Forward one MCP log line to the UI event queue.

            Args:
                msg: Log line from run_tool_loop.

            Returns:
                None.
            """
            self._event_queue.put({"type": "mcp_trace", "text": msg})

        def mcp_worker() -> None:
            """Run asyncio.run(run_tool_loop) and post completion to the queue.

            Returns:
                None. Posts mcp_done, mcp_error, or mcp_state events.
            """
            try:
                result = asyncio.run(
                    run_tool_loop(
                        params,
                        trace=trace,
                        python_executable=sys.executable,
                        abort_event=self._mcp_abort_event,
                        confirm_tool=self._wait_mcp_confirm,
                    )
                )
                self._event_queue.put({"type": "mcp_done", "result": result})
            except Exception as exc:  # noqa: BLE001
                self._event_queue.put({"type": "mcp_error", "text": str(exc)})
            finally:
                self._event_queue.put({"type": "mcp_state", "mcp_running": False})

        self._mcp_worker = threading.Thread(target=mcp_worker, daemon=True)
        self._mcp_worker.start()

    def _stop_all(self) -> None:
        """Signal chat stop and/or request MCP abort between steps.

        Returns:
            None.
        """
        self._stop_event.set()  # stop chat between chunks
        if self._mcp_running:
            self._mcp_abort_event.set()  # stop MCP between tool steps
            self._status_var.set("Abort requested…")

    def _reset_buttons(self) -> None:
        """Re-enable Send/MCP buttons and disable Stop when idle.

        Returns:
            None.
        """
        self._send_button.configure(state=tk.NORMAL)
        if not self._mcp_running:
            self._run_mcp_button.configure(state=tk.NORMAL)
        if not self._is_streaming and not self._mcp_running:
            self._stop_button.configure(state=tk.DISABLED)

    def _finish_stream_ui(self) -> None:
        """Reset UI state after streaming chat completes.

        Returns:
            None.
        """
        self._is_streaming = False
        self._reset_buttons()
        # Keep error/abort text unless we are going back to ready.
        if not str(self._status_var.get()).startswith(("Error", "Abort")):
            self._status_var.set(f"Ready ({self._model_var.get()})")

    def _finish_mcp_ui(self) -> None:
        """Reset UI state after the MCP worker thread exits.

        Returns:
            None.
        """
        self._mcp_running = False
        self._reset_buttons()

    def _handle_mcp_done(self, result: ToolLoopResult) -> None:
        """Display the final MCP loop result in trace and answer panes.

        Args:
            result: Outcome from run_tool_loop (status, answer, error).

        Returns:
            None.
        """
        if result.status == "completed":
            self._replace_text(self._answer_text, result.answer)
            self._append_text(self._thinking_text, f"\n[agent done] {result.status}\n")
            self._error_var.set("")
            self._status_var.set(f"Ready ({self._model_var.get()})")
            return

        if result.status == "error":
            self._error_var.set(result.error or "Unknown error")
            self._append_text(self._thinking_text, f"\n[agent error] {result.error}\n")
            self._status_var.set("MCP agent error")
            return

        if result.status == "aborted":
            self._append_text(self._thinking_text, "\n[agent aborted]\n")
            self._status_var.set("MCP agent aborted")
            self._error_var.set("")
            return

        # Other end states (e.g. hit max steps).
        self._append_text(self._thinking_text, f"\n[agent ended] {result.status}\n")
        self._error_var.set("")
        self._status_var.set(f"Ready ({self._model_var.get()})")

    def _handle_mcp_confirm(self, event: dict[str, object]) -> None:
        """Show yes/no dialog for a tool call and reply to the worker.

        Args:
            event: Queue event with tool name and arguments keys.

        Returns:
            None. Puts True/False on _mcp_confirm_reply_queue.
        """
        tool = str(event.get("tool", ""))
        args = event.get("arguments")
        if not isinstance(args, dict):
            args = {}
        approved = messagebox.askyesno(
            "Confirm tool",
            format_confirm_message(tool, args),
            parent=self._root,
        )
        self._mcp_confirm_reply_queue.put(approved)  # unblocks worker thread

    def _drain_events(self) -> None:
        """Process queued worker events on the Tk main thread, then reschedule.

        Returns:
            None. Schedules itself again via root.after(50, ...).
        """
        while True:
            try:
                event = self._event_queue.get_nowait()  # non-blocking read
            except Empty:
                break  # nothing left this tick

            event_type = str(event.get("type", ""))
            text = str(event.get("text", ""))

            # Route each event type to the right UI update.
            if event_type == "thinking":
                self._append_text(self._thinking_text, text)
            elif event_type == "answer":
                self._append_text(self._answer_text, text)
            elif event_type == "status":
                self._append_text(self._thinking_text, f"\n[status] {text}\n")
                self._status_var.set(text)
            elif event_type == "done":
                self._finish_stream_ui()
            elif event_type == "mcp_trace":
                self._append_text(self._thinking_text, text)
            elif event_type == "mcp_confirm":
                self._handle_mcp_confirm(event)
            elif event_type == "mcp_done":
                result = event.get("result")
                if isinstance(result, ToolLoopResult):
                    self._handle_mcp_done(result)
            elif event_type == "mcp_error":
                err = str(event.get("text", "Unknown"))
                self._error_var.set(err)
                self._append_text(self._thinking_text, f"\n[agent exception] {err}\n")
                self._status_var.set("MCP agent failed")
            elif event_type == "mcp_state":
                if not bool(event.get("mcp_running", True)):
                    self._finish_mcp_ui()

        # Schedule next poll (never touch widgets from worker threads).
        self._root.after(50, self._drain_events)

    def _clear_outputs(self) -> None:
        """Clear trace pane, answer pane, and error line.

        Returns:
            None.
        """
        self._replace_text(self._thinking_text, "")
        self._replace_text(self._answer_text, "")
        self._error_var.set("")

    def _replace_text(self, widget: tk.Text, value: str) -> None:
        """Replace all text in a read-only Text widget.

        Args:
            widget: Target tk.Text widget.
            value: New full contents.

        Returns:
            None.
        """
        widget.configure(state=tk.NORMAL)  # unlock
        widget.delete("1.0", tk.END)  # clear
        widget.insert(tk.END, value)  # write
        widget.configure(state=tk.DISABLED)  # lock again

    def _append_text(self, widget: tk.Text, value: str) -> None:
        """Append text to a read-only Text widget and scroll to bottom.

        Args:
            widget: Target tk.Text widget.
            value: Text to append at the end.

        Returns:
            None.
        """
        widget.configure(state=tk.NORMAL)  # unlock
        widget.insert(tk.END, value)  # add at end
        widget.see(tk.END)  # scroll down
        widget.configure(state=tk.DISABLED)  # lock again


def main() -> None:
    """Create the Tk app, wire Ollama client, and start the event loop.

    Reads OLLAMA_MODEL and OLLAMA_COMMAND from the environment.

    Returns:
        None.
    """
    root = tk.Tk()
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
    command = os.getenv("OLLAMA_COMMAND", "ollama")
    client = OllamaClient(OllamaConfig(command=command, model=model_name))
    OllamaGuiApp(root, client)
    root.mainloop()  # run until window closes


if __name__ == "__main__":
    main()
