#!/usr/bin/env bash
# Run every test suite against a built and sourced workspace.
#
#   source /opt/ros/humble/setup.bash && source <ws>/install/setup.bash
#   ./tests/run_all.sh
#
# test_imports runs first, so a broken import is reported once.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/tests:${PYTHONPATH:-}"
fail=0
for t in tests/test_imports.py tests/test_*.py; do
    [[ "$t" == "tests/test_imports.py" && -n "${_seen_imports:-}" ]] && continue
    [[ "$t" == "tests/test_imports.py" ]] && _seen_imports=1
    printf '%-24s ' "$(basename "$t" .py)"
    if out=$(/usr/bin/python3 "$t" 2>&1); then
        echo "$out" | tail -1
    else
        echo "$out" | tail -1
        fail=1
    fi
done
exit $fail
