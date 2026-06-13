"""Run the eval suite against the real Anthropic API and write EVALUATION.md.

    python -m calendai.evals.cli                     # all scenarios -> EVALUATION.md
    python -m calendai.evals.cli --filter memory     # only scenarios tagged/ided 'memory'
    python -m calendai.evals.cli --no-judge          # skip LLM-judge checks (free-er)

This costs real tokens. The harness logic is unit-tested separately with a
scripted client (tests/test_evals.py), so iterate there first.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from calendai.core.config import get_settings
from calendai.evals.report import render_report
from calendai.evals.results import ScenarioResult
from calendai.evals.runner import run_scenario
from calendai.evals.schema import Scenario, load_scenarios

DEFAULT_SCENARIO_DIR = Path(__file__).resolve().parents[3] / "evals" / "scenarios"
DEFAULT_OUT = Path(__file__).resolve().parents[3] / "EVALUATION.md"


def _matches(scenario: Scenario, needle: str | None) -> bool:
    if not needle:
        return True
    return needle in scenario.id or needle in scenario.tags


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run the CalendAI eval suite")
    parser.add_argument("--scenarios", default=str(DEFAULT_SCENARIO_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--filter", default=None, help="only run scenarios whose id/tags match")
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-judge checks")
    parser.add_argument("--runs", type=int, default=None, help="override runs-per-scenario")
    args = parser.parse_args(argv)

    import anthropic  # local import: the harness tests never need the SDK

    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    scenarios = [s for s in load_scenarios(args.scenarios) if _matches(s, args.filter)]
    if not scenarios:
        print(f"no scenarios matched filter {args.filter!r} in {args.scenarios}")
        return 1

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        if args.runs is not None:
            scenario = scenario.model_copy(update={"runs": args.runs})
        print(f"running {scenario.id} ({scenario.runs} run(s))...", flush=True)
        result = run_scenario(
            scenario,
            agent_client=client,
            agent_model=settings.calendai_agent_model,
            utility_client=client,
            utility_model=settings.calendai_utility_model,
            run_judge=not args.no_judge,
        )
        status = "PASS" if result.passed else "FAIL"
        print(f"  -> {status} ({sum(1 for r in result.runs if r.passed)}/{len(result.runs)} runs)")
        results.append(result)

    report = render_report(
        results,
        agent_model=settings.calendai_agent_model,
        utility_model=settings.calendai_utility_model,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )
    Path(args.out).write_text(report, encoding="utf-8")
    passed = sum(1 for r in results if r.passed)
    print(f"\n{passed}/{len(results)} scenarios passed. Report written to {args.out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
