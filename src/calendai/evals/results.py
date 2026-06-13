"""Result models shared by the runner, scorers, and report."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CheckResult(BaseModel):
    layer: str  # "end_state" | "trajectory" | "judge" | "reply"
    name: str
    passed: bool
    detail: str = ""


class RunResult(BaseModel):
    run_index: int
    checks: list[CheckResult] = Field(default_factory=list)
    final_replies: list[str] = Field(default_factory=list)
    error: str | None = None  # uncaught harness/agent error aborting the run

    @property
    def passed(self) -> bool:
        return self.error is None and all(c.passed for c in self.checks)


class ScenarioResult(BaseModel):
    scenario_id: str
    description: str
    tags: list[str] = Field(default_factory=list)
    runs: list[RunResult] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.passed) / len(self.runs)

    @property
    def passed(self) -> bool:
        """A scenario passes only if EVERY run passes (no flakiness allowed)."""
        return bool(self.runs) and all(r.passed for r in self.runs)

    def failing_checks(self) -> list[CheckResult]:
        return [c for r in self.runs for c in r.checks if not c.passed]
