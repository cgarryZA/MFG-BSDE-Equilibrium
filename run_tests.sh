#!/bin/bash
# Run all tests. Fast tests (default) take <1 min. Full suite (--slow) ~5 min.
#
# Usage:
#   ./run_tests.sh        — fast tests only (smoke tests + sanity checks)
#   ./run_tests.sh --slow — include slow tests (full training accuracy)
#   ./run_tests.sh -k foo — only tests matching 'foo'
#
# Run this BEFORE kicking off any long experiment.

cd "$(dirname "$0")"

if [[ "$1" == "--slow" ]]; then
    shift
    python -m pytest tests/ "$@"
else
    python -m pytest tests/ -m "not slow" "$@"
fi
