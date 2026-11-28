#!/usr/bin/env bash
# Hold the training host's shared GPU lock until the actual training process exits.
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: bash train_locked.sh M4_ROOT DATA SPLIT BASE RUN_ID" >&2
    exit 2
fi
root=$(realpath -e -- "$1")
data=$(realpath -e -- "$2")
split=$(realpath -e -- "$3")
base=$(realpath -e -- "$4")
id=$5
if [[ ! $id =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid run ID: $id" >&2
    exit 2
fi
py="$root/env/bin/python"
script="$(dirname "$(realpath -e -- "$0")")/train.py"
[[ -x $py && -d $data && -s $split && -s $base/config.json ]]
exec 9<"$root/../.split_training.lock"
if ! flock -n 9; then
    echo "Shared training GPU lock is busy" >&2
    exit 1
fi
mkdir -p "$root/runs"
run="$root/runs/$id"
mkdir "$run"
exec >"$run/train.log" 2>&1
trap 'rc=$?; echo "$rc" > "$run/exit-code"' EXIT
printf '%s\n' "$$" > "$run/pid"
ps -o pid,pgid,lstart,args -p "$$" > "$run/process.txt"
printf 'data=%s\nsplit=%s\nbase=%s\n' "$data" "$split" "$base" > "$run/inputs.txt"
cp -- "$split" "$run/split.json"
export HF_HOME="$root/cache/huggingface" HF_HUB_OFFLINE=1
export TORCH_HOME="$root/cache/torch" TRITON_CACHE_DIR="$root/cache/triton"
"$py" -B -u "$script" "$data" "$run/split.json" "$base" "$run/tuned"
