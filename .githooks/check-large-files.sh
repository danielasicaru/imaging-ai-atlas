#!/bin/sh
# Refuses to stage large raw-data files directly. See README.
set -eu

staged_list=$(mktemp)
trap 'rm -f "$staged_list"' EXIT
git diff --cached --name-only --diff-filter=ACM > "$staged_list"

status=0
while IFS= read -r f; do
    [ -f "$f" ] || continue
    case "$f" in
        data/*.csv|data/*.parquet|data/*.json|data/*.jsonl) ;;
        *) continue ;;
    esac
    size=$(wc -c < "$f")
    if [ "$size" -gt 1048576 ]; then
        echo "refusing to commit $f (>1MB): raw data stays out of git (see data/raw/SHA256SUMS)" >&2
        status=1
    fi
done < "$staged_list"

exit "$status"
