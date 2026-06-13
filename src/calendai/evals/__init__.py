"""Automated evaluation pipeline: declarative scenarios, layered scoring, report.

A scenario (YAML) describes a deterministic world (frozen clock, seeded
calendar + facts, optional injected provider failures) and a sequence of
user turns across one or more sessions. A new session in the list simulates
a process RESTART: the calendar (external, in-memory provider) persists, but
the SQLite store holding profile memory + traces is closed and reopened -
which is precisely how we prove long-term memory survives a restart.

Three scoring layers, in order of trust:
1. end-state assertions (primary, objective): the calendar and stored facts
   after the run;
2. trajectory assertions (from traces): which tools were / were not called;
3. an LLM judge (utility model) for response-quality rubrics objective
   assertions cannot capture.

The runner hits the real Anthropic API (that is the point - we measure the
real agent, not the plumbing); its scoring pieces are pure and unit-tested
without the network, and the runner accepts an injected client so a full
scenario can be exercised end-to-end with a scripted model for free.
"""
