# ADR 0002: Defer duplicate-code detection

**Status:** accepted
**Date:** 2026-11-03

## Context

jscpd (copy-paste detection) needs Node.js. This repo has no other Node dependency, so installing a whole
JS toolchain for one lint check is a disproportionate footprint for a Python project.

## Decision

Skip jscpd for now. Revisit duplicate-code detection after the report (the stage where the codebase is
most likely to have accumulated copy-paste, e.g. near-identical plotting code per report section) using
one of:
- `pylint` with only the `duplicate-code` checker enabled (pure Python, no new runtime).
- jscpd, if Node ends up installed anyway for another reason by then.

## Consequences

No duplicate-code check runs in CI until this is revisited. This is an accepted gap, not an oversight —
tracked here so it doesn't get silently forgotten.
