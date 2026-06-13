"""Render ScenarioResults into EVALUATION.md: success rates + failure analysis."""

from __future__ import annotations

from collections import defaultdict

from calendai.evals.results import ScenarioResult


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100 * numerator / denominator:.0f}%"


def _overall(results: list[ScenarioResult]) -> tuple[int, int, int, int]:
    scen_pass = sum(1 for r in results if r.passed)
    total_runs = sum(len(r.runs) for r in results)
    run_pass = sum(1 for r in results for run in r.runs if run.passed)
    return scen_pass, len(results), run_pass, total_runs


def render_report(
    results: list[ScenarioResult],
    *,
    agent_model: str,
    utility_model: str,
    generated_at: str,
) -> str:
    scen_pass, scen_total, run_pass, run_total = _overall(results)
    lines: list[str] = []
    lines.append("# CalendAI - Evaluation Report")
    lines.append("")
    lines.append(f"_Generated {generated_at}_")
    lines.append("")
    lines.append(f"- **Agent model:** `{agent_model}`")
    lines.append(f"- **Utility model (extraction + judge):** `{utility_model}`")
    lines.append(
        f"- **Scenarios passed:** {scen_pass}/{scen_total} ({_pct(scen_pass, scen_total)})  "
        "- a scenario passes only if every repeated run passes (no flakiness allowed)."
    )
    lines.append(
        f"- **Individual runs passed:** {run_pass}/{run_total} ({_pct(run_pass, run_total)})"
    )
    lines.append("")
    lines.append(_tag_section(results))
    lines.append(_scenario_table(results))
    lines.append(_failure_section(results))
    return "\n".join(lines).rstrip() + "\n"


def _tag_section(results: list[ScenarioResult]) -> str:
    by_tag: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        for tag in r.tags or ["untagged"]:
            by_tag[tag].append(r.passed)
    lines = [
        "## Success rate by capability",
        "",
        "| Capability | Scenarios | Passed |",
        "|---|---|---|",
    ]
    for tag in sorted(by_tag):
        flags = by_tag[tag]
        lines.append(f"| {tag} | {len(flags)} | {_pct(sum(flags), len(flags))} |")
    lines.append("")
    return "\n".join(lines)


def _scenario_table(results: list[ScenarioResult]) -> str:
    lines = [
        "## Per-scenario results",
        "",
        "| Scenario | Tags | Runs | Pass rate | Status |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        status = "✅ pass" if r.passed else "❌ FAIL"
        run_pass = sum(1 for run in r.runs if run.passed)
        lines.append(
            f"| `{r.scenario_id}` | {', '.join(r.tags)} | {len(r.runs)} | "
            f"{run_pass}/{len(r.runs)} | {status} |"
        )
    lines.append("")
    return "\n".join(lines)


def _failure_section(results: list[ScenarioResult]) -> str:
    failing = [r for r in results if not r.passed]
    lines = ["## Failure analysis", ""]
    if not failing:
        lines.append("No failures. Every scenario passed on every run.")
        lines.append("")
        return "\n".join(lines)
    for r in failing:
        lines.append(f"### `{r.scenario_id}` - {r.description}")
        run_pass = sum(1 for run in r.runs if run.passed)
        lines.append(f"Passed {run_pass}/{len(r.runs)} runs. Distinct failing checks:")
        lines.append("")
        for detail in _unique_failures(r):
            lines.append(f"- {detail}")
        lines.append("")
    return "\n".join(lines)


def _unique_failures(result: ScenarioResult) -> list[str]:
    seen: dict[str, str] = {}
    for run in result.runs:
        if run.error:
            seen.setdefault(f"error::{run.error}", f"**[run error]** {run.error}")
        for check in run.checks:
            if not check.passed:
                key = f"{check.layer}::{check.name}"
                detail = f" - {check.detail}" if check.detail else ""
                seen.setdefault(key, f"**[{check.layer}]** {check.name}{detail}")
    return list(seen.values())
