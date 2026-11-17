#!/usr/bin/env bash
set -euo pipefail
export GH_PROMPT_DISABLED=1
repo=the-fall-moon/git_test_demo
api="repos/$repo/actions"
usage() {
    echo "Usage: bash wsl-gpu-tests.sh run BRANCH PYTHON CUDA | resume RUN_ID" >&2
    exit 2
}

case "${1:-}" in
    run)
        [[ $# == 4 && -n $2 ]] || usage
        case "$3" in 3.10|3.11|3.12|3.13) ;; *) usage ;; esac
        case "$4" in cu132|cu134) ;; *) usage ;; esac
        id=$(gh api --hostname github.com -X POST "$api/workflows/wsl-gpu-tests.yml/dispatches" \
            -H 'X-GitHub-Api-Version: 2026-03-10' -f "ref=$2" \
            -f "inputs[python_version]=$3" -f "inputs[cuda_build]=$4" --jq '.workflow_run_id')
        ;;
    resume)
        [[ $# == 2 ]] || usage
        id=$2
        ;;
    *) usage ;;
esac
if [[ ! $id =~ ^[1-9][0-9]*$ ]]; then
    echo "Missing numeric run ID. Check the workflow run list before dispatching again." >&2
    exit 2
fi

echo "Run ID $id | https://github.com/$repo/actions/runs/$id"
while true; do
    state=$(gh api --hostname github.com "$api/runs/$id" \
        --jq '[.status, (.conclusion // "-"), .run_attempt, .head_sha] | @tsv')
    read -r status conclusion attempt sha <<< "$state"
    echo "$id $sha $status $conclusion"
    [[ $status == completed ]] && break
    sleep 20
done

out="gpu-results/$id/$attempt"
mkdir -p "$out"
if [[ $conclusion != success ]]; then
    gh run view "$id" -R "$repo" --attempt "$attempt" --log-failed > "$out/failed.log" || echo "Failed log unavailable." >&2
fi
gh run download "$id" -R "$repo" -n "gpu-results-$id-$attempt" -D "$out"
echo "Report: $out"
[[ $conclusion == success ]]
