"""Workflow state and logging helpers for the AutoPore GUI."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


WORKFLOW_STEPS = ("generate_input", "run_simulation", "plot_result")

STEP_LABELS = {
    "generate_input": "Generate input",
    "run_simulation": "Run simulation",
    "plot_result": "Plot results",
}

STEP_NODES = {
    "generate_input": ("image_generation", "image_import", "pore_extraction"),
    "run_simulation": ("quasi_static", "dynamic"),
    "plot_result": ("plot_result",),
}

NODE_TO_STEP = {
    node: step
    for step, nodes in STEP_NODES.items()
    for node in nodes
}


class WorkflowTracker:
    """Track one top-to-bottom workflow cycle and write a JSON-lines log."""

    def __init__(
        self,
        log_file: str | Path,
        task_mode: str = "single_run",
        pnm_mode: str = "quasi_static",
    ):
        self.log_file = Path(log_file)
        self.cycle_id = 0
        self.task_mode = task_mode
        self.pnm_mode = pnm_mode
        self.completed_steps: list[str] = []
        self.completed_nodes: list[str] = []
        self.started_from: str | None = None
        self.last_step_index = -1
        self.active_step: str | None = None
        self.active_nodes: list[str] = []
        self.uq_realization_index: int | None = None
        self.uq_total: int | None = None
        self.closed = False
        self.last_event = "Session started."
        self._start_new_cycle("Session started.", started_from=None)

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _branch(self) -> str:
        return f"{self.task_mode}/{self.pnm_mode}"

    def _step_index(self, step: str) -> int:
        return WORKFLOW_STEPS.index(step)

    def _node_step_index(self, node: str) -> int:
        step = NODE_TO_STEP.get(node, node)
        if step in WORKFLOW_STEPS:
            return self._step_index(step)
        return len(WORKFLOW_STEPS)

    def _label(self, step: str) -> str:
        return STEP_LABELS.get(step, step)

    def _step_sequence_text(self, steps: list[str] | tuple[str, ...]) -> str:
        return " -> ".join(self._label(step) for step in steps)

    def _blocked_forward_message(self, wanted_step: str, missing_steps: list[str]) -> str:
        required_sequence = missing_steps + [wanted_step]
        next_step = missing_steps[0] if missing_steps else wanted_step
        return (
            f"I did not run {self._label(wanted_step)} because that step is not allowed yet. "
            f"To call {self._label(wanted_step)}, complete this sequence first: "
            f"{self._step_sequence_text(required_sequence)}. "
            f"The next allowed step is {self._label(next_step)}."
        )

    def _blocked_backward_message(self, wanted_step: str) -> str:
        current_step = WORKFLOW_STEPS[self.last_step_index]
        required_sequence = list(WORKFLOW_STEPS[: self._step_index(wanted_step) + 1])
        return (
            f"I did not run {self._label(wanted_step)} because this workflow can only move "
            f"from top to bottom, and the current cycle has already reached "
            f"{self._label(current_step)}. To call {self._label(wanted_step)}, start over, "
            f"then complete: {self._step_sequence_text(required_sequence)}."
        )

    def _write_log(self, event: str, **payload: Any) -> None:
        record = {
            "time": self._now(),
            "event": event,
            "cycle_id": self.cycle_id,
            "task_mode": self.task_mode,
            "pnm_mode": self.pnm_mode,
            **payload,
        }
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            # Logging should never prevent the model workflow from running.
            pass

    def _clear_log(self) -> None:
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.log_file.write_text("", encoding="utf-8")
        except OSError:
            # Logging should never prevent the model workflow from running.
            pass

    def _start_new_cycle(self, reason: str, started_from: str | None = None) -> None:
        self.cycle_id += 1
        self.completed_steps = []
        self.completed_nodes = []
        self.started_from = started_from
        self.last_step_index = -1
        self.active_step = None
        self.active_nodes = []
        self.uq_realization_index = None
        self.uq_total = None
        self.closed = False
        self.last_event = reason
        self._write_log(
            "cycle_started",
            reason=reason,
            started_from=started_from,
        )

    def set_modes(self, task_mode: str, pnm_mode: str) -> None:
        if task_mode == self.task_mode and pnm_mode == self.pnm_mode:
            return
        old_branch = self._branch()
        self.task_mode = task_mode
        self.pnm_mode = pnm_mode
        self._start_new_cycle(
            f"Workflow branch changed from {old_branch} to {self._branch()}.",
            started_from=None,
        )

    def reset(self, reason: str = "Workflow reset.") -> None:
        self._start_new_cycle(reason, started_from=None)

    def note(self, event: str, **payload: Any) -> None:
        realization_index = payload.get("realization_index", payload.get("index"))
        total = payload.get("total")
        if realization_index is not None and total is not None:
            self.set_uq_progress(realization_index, total)
        self.last_event = event.replace("_", " ").capitalize()
        self._write_log(event, **payload)

    def set_uq_progress(self, realization_index: int | None, total: int | None) -> None:
        try:
            self.uq_realization_index = (
                int(realization_index) if realization_index is not None else None
            )
            self.uq_total = int(total) if total is not None else None
        except (TypeError, ValueError):
            self.uq_realization_index = None
            self.uq_total = None

    def _trim_highlights_from(self, step: str) -> None:
        index = self._step_index(step)
        self.completed_steps = [
            completed
            for completed in self.completed_steps
            if self._step_index(completed) < index
        ]
        self.completed_nodes = [
            completed
            for completed in self.completed_nodes
            if self._node_step_index(completed) < index
        ]
        self.last_step_index = index - 1

    def _highlight_upstream_of(
        self,
        step: str,
        completed_nodes: list[str] | None = None,
    ) -> None:
        index = self._step_index(step)
        self.completed_steps = list(WORKFLOW_STEPS[:index])
        if completed_nodes is not None:
            self.completed_nodes = [
                node
                for node in completed_nodes
                if self._node_step_index(node) < index
            ]
        self.last_step_index = index - 1

    def start_uq_realization(self, realization_index: int, total: int, case: str | None = None) -> None:
        """Clear and replay input/simulation highlights for one UQ realization."""
        self.task_mode = "uq_run"
        self.set_uq_progress(realization_index, total)
        self.closed = False
        self.started_from = "generate_input"
        self.completed_steps = []
        self.completed_nodes = []
        self.last_step_index = -1
        self.active_step = "generate_input"
        self.active_nodes = ["image_generation"]
        case_text = f" ({case})" if case else ""
        self.last_event = (
            f"UQ realization {realization_index}/{total}{case_text}: generating input."
        )
        self._write_log(
            "uq_realization_started",
            realization_index=realization_index,
            total=total,
            case=case,
        )

    def switch_active_nodes(
        self,
        step: str,
        completed_nodes: list[str] | None = None,
        active_nodes: list[str] | None = None,
        message: str | None = None,
        source: str = "workflow_progress",
    ) -> None:
        """Advance the active visual node inside a still-running workflow step."""
        if step != self.active_step:
            return
        for node in completed_nodes or []:
            if node not in self.completed_nodes:
                self.completed_nodes.append(node)
        self.active_nodes = list(dict.fromkeys(active_nodes or []))
        self.last_event = message or f"Progressed within {self._label(step)}."
        self._write_log(
            "step_progress",
            step=step,
            source=source,
            completed_nodes=list(self.completed_nodes),
            active_nodes=list(self.active_nodes),
            message=self.last_event,
        )

    def prepare_step(
        self,
        step: str,
        source: str = "execute_model",
        active_nodes: list[str] | None = None,
        upstream_nodes: list[str] | None = None,
    ) -> tuple[bool, str]:
        if step not in WORKFLOW_STEPS:
            allowed_steps = self._step_sequence_text(WORKFLOW_STEPS)
            message = (
                f"I did not run anything because {step} is not a known workflow step. "
                f"Allowed workflow order is: {allowed_steps}."
            )
            self.last_event = message
            self._write_log(
                "step_blocked",
                step=step,
                source=source,
                reason=message,
            )
            return False, message

        index = self._step_index(step)
        if self.closed:
            self._start_new_cycle(
                "Previous closed loop finished after plotting; starting a new cycle.",
                started_from=step,
            )
            if index > 0:
                self._highlight_upstream_of(step, completed_nodes=upstream_nodes)
        elif self.last_step_index < 0:
            self.started_from = step
            if index > 0:
                self._highlight_upstream_of(step, completed_nodes=upstream_nodes)
                self.last_event = (
                    f"Starting from the middle at {self._label(step)} using existing files."
                )
                self._write_log(
                    "cycle_started_from_middle",
                    step=step,
                    source=source,
                )
        else:
            if index < self.last_step_index:
                message = self._blocked_backward_message(step)
                self._write_log(
                    "step_blocked",
                    step=step,
                    source=source,
                    reason=message,
                )
                self.last_event = message
                return False, message

            if index == self.last_step_index and step in self.completed_steps:
                self._trim_highlights_from(step)

            if index > self.last_step_index + 1:
                missing_steps = list(WORKFLOW_STEPS[self.last_step_index + 1:index])
                message = self._blocked_forward_message(step, missing_steps)
                self._write_log(
                    "step_blocked",
                    step=step,
                    source=source,
                    reason=message,
                )
                self.last_event = message
                return False, message

        self._trim_highlights_from(step)
        self.active_step = step
        self.active_nodes = list(dict.fromkeys(active_nodes or [step]))
        self.last_event = f"Started {self._label(step)}."
        self._write_log(
            "step_started",
            step=step,
            source=source,
            active_nodes=list(self.active_nodes),
        )
        return True, ""

    def complete_step(
        self,
        step: str,
        status: str,
        message: str | None = None,
        source: str = "execute_model",
        completed_nodes: list[str] | None = None,
    ) -> None:
        success_like = status in {"success", "manually_closed"}
        if success_like and step in WORKFLOW_STEPS:
            if step not in self.completed_steps:
                self.completed_steps.append(step)
            for node in completed_nodes or [step]:
                if node not in self.completed_nodes:
                    self.completed_nodes.append(node)
            self.last_step_index = max(self.last_step_index, self._step_index(step))
            self.active_step = None
            self.active_nodes = []
            if step == "plot_result":
                self.closed = True
                self.last_event = (
                    "Closed loop completed after plotting. Next you can start over "
                    "or start from a middle step using existing files."
                )
            else:
                self.last_event = f"Finished {self._label(step)}."
        elif message:
            self.active_step = None
            self.active_nodes = []
            self.last_event = message
        else:
            self.active_step = None
            self.active_nodes = []
            self.last_event = f"{self._label(step)} ended with status {status}."

        self._write_log(
            "step_finished",
            step=step,
            status=status,
            source=source,
            message=message,
            completed_steps=list(self.completed_steps),
            completed_nodes=list(self.completed_nodes),
            active_step=self.active_step,
            active_nodes=list(self.active_nodes),
            closed=self.closed,
        )
        if success_like and step == "plot_result" and self.closed:
            self._clear_log()

    def complete_multiple(
        self,
        steps: list[str],
        status: str,
        message: str | None = None,
        source: str = "run_realizations",
    ) -> None:
        for step in steps:
            self.complete_step(step, status, message=message, source=source)

    def next_allowed_steps(self) -> list[str]:
        if self.closed or self.last_step_index < 0:
            return list(WORKFLOW_STEPS)
        if self.last_step_index + 1 < len(WORKFLOW_STEPS):
            return [WORKFLOW_STEPS[self.last_step_index + 1]]
        return []

    def snapshot(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "task_mode": self.task_mode,
            "pnm_mode": self.pnm_mode,
            "branch": self._branch(),
            "completed_steps": list(self.completed_steps),
            "highlighted_steps": list(self.completed_steps),
            "completed_nodes": list(self.completed_nodes),
            "highlighted_nodes": list(self.completed_nodes),
            "active_step": self.active_step,
            "active_nodes": list(self.active_nodes),
            "uq_realization_index": self.uq_realization_index,
            "uq_total": self.uq_total,
            "started_from": self.started_from,
            "closed": self.closed,
            "last_event": self.last_event,
            "next_allowed_steps": self.next_allowed_steps(),
            "log_file": str(self.log_file),
        }

    def format_status(self) -> str:
        lines = [
            "Workflow Tracker",
            f"Branch: {self._branch()}",
            f"Cycle: {self.cycle_id} ({'closed' if self.closed else 'open'})",
            f"Log: {self.log_file}",
            "",
            "Progress:",
        ]

        start_index = self._step_index(self.started_from) if self.started_from else 0
        for step in WORKFLOW_STEPS:
            if step in self.completed_steps:
                status = "done"
            elif step == self.active_step:
                status = "running"
            elif self.started_from and self._step_index(step) < start_index:
                status = "existing files"
            else:
                status = "pending"
            lines.append(f"  [{status}] {self._label(step)}")

        lines.append("")
        lines.append("Next allowed:")
        allowed = self.next_allowed_steps()
        if self.closed:
            if self.task_mode == "uq_run":
                lines.append("  Start over: generate_input & run_simulation")
                lines.append("  Start from middle: plot_result")
            else:
                lines.append("  Start over: generate_input")
                lines.append("  Start from middle: run_simulation or plot_result")
        elif not allowed:
            lines.append("  None until this cycle is reset.")
        else:
            for step in allowed:
                lines.append(f"  {self._label(step)}")
            if self.last_step_index >= 0:
                lines.append("  Start over only after reset/start over.")

        lines.append("")
        lines.append(f"Last event: {self.last_event}")
        return "\n".join(lines)
