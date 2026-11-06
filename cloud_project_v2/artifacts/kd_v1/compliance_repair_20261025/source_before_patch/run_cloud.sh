#!/usr/bin/env bash
set -Eeuo pipefail

# Run manually on Linux. --dry-run only prints commands; it never starts Python.
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
IMAGE_ROOT="${IMAGE_ROOT:-/home/featurize/data/cxr_first}"
WORK_ROOT="${WORK_ROOT:-/home/featurize/work}"
RUN_NAME="${RUN_NAME:-015_cloud_tiny_b32_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_ROOT/runs/$RUN_NAME}"
LOG_FILE="${LOG_FILE:-$WORK_ROOT/logs/$RUN_NAME.log}"
SESSION_NAME="${SESSION_NAME:-$RUN_NAME}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-32}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
EPOCHS="${EPOCHS:-25}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
PATIENCE="${PATIENCE:-5}"
MIN_DELTA="${MIN_DELTA:-0.0001}"
BACKBONE_LR="${BACKBONE_LR:-0.00005}"
HEAD_LR="${HEAD_LR:-0.0005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
DROPOUT="${DROPOUT:-0.1}"
DROP_PATH="${DROP_PATH:-0.05}"
AMP_DTYPE="${AMP_DTYPE:-auto}"
POS_WEIGHT_POWER="${POS_WEIGHT_POWER:-0}"
POS_WEIGHT_CAP="${POS_WEIGHT_CAP:-3}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
TORCH_CUDNN_V8_API_DISABLED="${TORCH_CUDNN_V8_API_DISABLED:-0}"
MODE="${1:-}"
EXIT_FILE="$LOG_FILE.exitcode"

if [[ $# -gt 1 || ( -n "$MODE" && "$MODE" != --dry-run && "$MODE" != --foreground && "$MODE" != --worker ) ]]; then
    echo 'Usage: bash run_cloud.sh [--dry-run|--foreground]' >&2
    exit 2
fi

command=("$PYTHON_BIN" -u "$PROJECT_DIR/train_tiny.py"
    --image-root "$IMAGE_ROOT" --work-root "$WORK_ROOT"
    --train-manifest "$PROJECT_DIR/data/clean_v1_reviewed/train_clean.csv"
    --valid-manifest "$PROJECT_DIR/data/clean_v1_reviewed/valid_manifest.csv"
    --output "$OUTPUT_DIR" --image-size 384
    --batch-size "$BATCH_SIZE" --valid-batch-size "$VALID_BATCH_SIZE" --num-workers "$NUM_WORKERS"
    --prefetch-factor "$PREFETCH_FACTOR"
    --accumulation-steps "$ACCUMULATION_STEPS" --epochs "$EPOCHS" --warmup-epochs "$WARMUP_EPOCHS"
    --patience "$PATIENCE" --min-delta "$MIN_DELTA"
    --backbone-lr "$BACKBONE_LR" --head-lr "$HEAD_LR" --weight-decay "$WEIGHT_DECAY"
    --dropout "$DROPOUT" --drop-path "$DROP_PATH" --amp-dtype "$AMP_DTYPE"
    --pos-weight-power "$POS_WEIGHT_POWER" --pos-weight-cap "$POS_WEIGHT_CAP")
case "$GRADIENT_CHECKPOINTING" in
    1) command+=(--gradient-checkpointing) ;;
    0) ;;
    *) printf 'ERROR: GRADIENT_CHECKPOINTING must be 0 or 1\n' >&2; exit 2 ;;
esac

if [[ "$MODE" == --dry-run ]]; then
    printf 'Python command: '
    printf '%q ' "${command[@]}"
    printf '\nSession: %s\nLog: %s\n' "$SESSION_NAME" "$LOG_FILE"
    exit 0
fi

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
timestamp() { TZ=Asia/Hong_Kong date --iso-8601=seconds; }

# Record bootstrap failures too. Previously Python/version/path checks ran before
# redirection, and a failed tmux pane could disappear without leaving any error.
if [[ "$MODE" == --worker ]]; then
    [[ -f "$LOG_FILE" ]] || fail "Worker requires an existing launch log: $LOG_FILE"
    exec >>"$LOG_FILE" 2>&1
else
    mkdir -p -- "$(dirname -- "$LOG_FILE")"
    [[ ! -e "$EXIT_FILE" ]] || fail "Exit status exists; select a new RUN_NAME: $EXIT_FILE"
    (set -o noclobber; : >"$LOG_FILE") || fail "Log exists; select a new RUN_NAME: $LOG_FILE"
    # Foreground mode now really shows errors in the terminal as well as the log.
    exec > >(tee -a -- "$LOG_FILE") 2>&1
fi
trap 'result=$?; printf "[%s] ERROR: launcher line %s exited %s\n" "$(timestamp)" "$LINENO" "$result" >&2' ERR
if [[ "$MODE" == --foreground || "$MODE" == --worker ]]; then
    record_exit() {
        result=$?
        trap - EXIT
        printf '[%s] Training process exited %s\n' "$(timestamp)" "$result"
        printf '%s\n' "$result" >"$EXIT_FILE"
    }
    trap record_exit EXIT
fi
printf '[%s] Launcher mode=%s run=%s\n' "$(timestamp)" "${MODE:-tmux}" "$RUN_NAME"
printf 'Runtime: Tiny/GAP/BCE batch_images=%s eval_batch=%s accumulation=%s amp=%s checkpoint=%s\n' \
    "$BATCH_SIZE" "$VALID_BATCH_SIZE" "$ACCUMULATION_STEPS" "$AMP_DTYPE" "$GRADIENT_CHECKPOINTING"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python not found: $PYTHON_BIN"
# Preserve the active venv even if an older tmux server has a different PATH.
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
command[0]="$PYTHON_BIN"
"$PYTHON_BIN" -c 'import sys; print("Python:", sys.executable, sys.version.split()[0], flush=True); assert sys.platform.startswith("linux") and sys.version_info[:2] == (3, 12), "Use Linux Python 3.12 for requirements.txt"'
[[ ! -e "$OUTPUT_DIR" ]] || fail "Output exists; select a new RUN_NAME: $OUTPUT_DIR"
for manifest in train_clean.csv valid_manifest.csv test_manifest.csv; do
    [[ -f "$PROJECT_DIR/data/clean_v1_reviewed/$manifest" ]] || fail "Missing manifest: $manifest"
done
if [[ -d "$IMAGE_ROOT/train" ]]; then
    EFFECTIVE_IMAGE_ROOT="$IMAGE_ROOT"
elif [[ -d "$IMAGE_ROOT/cxr_first/train" ]]; then
    EFFECTIVE_IMAGE_ROOT="$IMAGE_ROOT/cxr_first"
else
    fail "Expected $IMAGE_ROOT/train or $IMAGE_ROOT/cxr_first/train"
fi
for split in train valid test; do
    [[ -d "$EFFECTIVE_IMAGE_ROOT/$split" ]] || fail "Missing image split directory: $split"
done

# Limit native threads per loader worker on the 16-core host.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export TORCH_CUDNN_V8_API_DISABLED

if [[ "$MODE" == --foreground || "$MODE" == --worker ]]; then
    cd -- "$PROJECT_DIR"
    printf '[%s] Starting Python training process\n' "$(timestamp)"
    "${command[@]}"
    exit 0
fi

command -v tmux >/dev/null 2>&1 || fail 'tmux is not installed (sudo apt-get install tmux).'
if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
    fail "tmux session already exists: $SESSION_NAME"
fi
mkdir -p -- "$WORK_ROOT" "$(dirname -- "$LOG_FILE")"
launch=(env "IMAGE_ROOT=$IMAGE_ROOT" "WORK_ROOT=$WORK_ROOT" "RUN_NAME=$RUN_NAME"
    "OUTPUT_DIR=$OUTPUT_DIR" "LOG_FILE=$LOG_FILE" "SESSION_NAME=$SESSION_NAME"
    "PYTHON_BIN=$PYTHON_BIN" "BATCH_SIZE=$BATCH_SIZE" "NUM_WORKERS=$NUM_WORKERS"
    "VALID_BATCH_SIZE=$VALID_BATCH_SIZE"
    "PREFETCH_FACTOR=$PREFETCH_FACTOR"
    "ACCUMULATION_STEPS=$ACCUMULATION_STEPS"
    "EPOCHS=$EPOCHS" "WARMUP_EPOCHS=$WARMUP_EPOCHS" "PATIENCE=$PATIENCE" "MIN_DELTA=$MIN_DELTA"
    "BACKBONE_LR=$BACKBONE_LR" "HEAD_LR=$HEAD_LR" "WEIGHT_DECAY=$WEIGHT_DECAY"
    "DROPOUT=$DROPOUT" "DROP_PATH=$DROP_PATH" "AMP_DTYPE=$AMP_DTYPE"
    "POS_WEIGHT_POWER=$POS_WEIGHT_POWER" "POS_WEIGHT_CAP=$POS_WEIGHT_CAP"
    "GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING"
    "TORCH_CUDNN_V8_API_DISABLED=$TORCH_CUDNN_V8_API_DISABLED"
    bash "$PROJECT_DIR/run_cloud.sh" --worker)
printf -v launch_command '%q ' "${launch[@]}"
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_DIR" "$launch_command" \
    \; set-option -w -t "$SESSION_NAME:0" remain-on-exit on
# Catch immediate bootstrap failures; session creation alone is not training success.
for attempt in {1..10}; do
    if [[ -f "$EXIT_FILE" ]]; then
        read -r result <"$EXIT_FILE"
        [[ "$result" == 0 ]] || fail "Training exited $result; see $LOG_FILE (failed pane retained)."
        break
    fi
    sleep 0.2
done
printf 'Created tmux session: %s\nLog: %s\nOutput: %s\n' "$SESSION_NAME" "$LOG_FILE" "$OUTPUT_DIR"
printf 'Check image batch progress in the log to confirm training. Exit status: %s\n' "$EXIT_FILE"
printf 'Follow log: tail -f %q\n' "$LOG_FILE"
