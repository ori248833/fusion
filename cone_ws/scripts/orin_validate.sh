#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/home/juziwei/cone_ws"
PARAMS_FILE="$WORKSPACE/configs/yolo_detector.yaml"
BAG_PATH=""
IMAGE_TOPIC=""
BAG_RATE="1.0"
DURATION="20"
CHECK_DEBUG="false"
LOG_DIR=""

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/orin_validate.sh --bag <rosbag_path> [options]

Required:
  --bag <path>               rosbag path

Options:
  --workspace <path>         ROS workspace (default: /home/juziwei/cone_ws)
  --params-file <path>       detector param file (default: <workspace>/configs/yolo_detector.yaml)
  --image-topic <topic>      bag play topic (default: read from params file)
  --bag-rate <float>         rosbag play rate (default: 1.0)
  --duration <sec>           topic hz sample seconds (default: 20)
  --check-debug              also sample /yolo/debug_image rate
  --log-dir <path>           custom log directory
  -h, --help                 show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bag)
      BAG_PATH="$2"
      shift 2
      ;;
    --workspace)
      WORKSPACE="$2"
      shift 2
      ;;
    --params-file)
      PARAMS_FILE="$2"
      shift 2
      ;;
    --image-topic)
      IMAGE_TOPIC="$2"
      shift 2
      ;;
    --bag-rate)
      BAG_RATE="$2"
      shift 2
      ;;
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --check-debug)
      CHECK_DEBUG="true"
      shift
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$BAG_PATH" ]]; then
  echo "Missing required --bag <path>" >&2
  usage
  exit 1
fi

if [[ ! -d "$WORKSPACE" ]]; then
  echo "Workspace not found: $WORKSPACE" >&2
  exit 1
fi

if [[ ! -f "$PARAMS_FILE" ]]; then
  echo "Params file not found: $PARAMS_FILE" >&2
  exit 1
fi

if [[ ! -d "$BAG_PATH" ]]; then
  echo "Bag path not found: $BAG_PATH" >&2
  exit 1
fi

MODEL_PATH="$(awk '/model_path:/{print $2; exit}' "$PARAMS_FILE")"
if [[ -z "$MODEL_PATH" || ! -f "$MODEL_PATH" ]]; then
  echo "Model not found (from params): $MODEL_PATH" >&2
  exit 1
fi

if [[ -z "$IMAGE_TOPIC" ]]; then
  IMAGE_TOPIC="$(awk '/image_topic:/{print $2; exit}' "$PARAMS_FILE")"
fi
if [[ -z "$IMAGE_TOPIC" ]]; then
  echo "Failed to resolve image_topic from params. Use --image-topic explicitly." >&2
  exit 1
fi

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$WORKSPACE/runs/orin_validation_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$LOG_DIR"

DETECTOR_PID=""
BAG_PID=""

cleanup() {
  set +e
  if [[ -n "$BAG_PID" ]]; then
    kill "$BAG_PID" 2>/dev/null || true
    wait "$BAG_PID" 2>/dev/null || true
  fi
  if [[ -n "$DETECTOR_PID" ]]; then
    kill "$DETECTOR_PID" 2>/dev/null || true
    wait "$DETECTOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

source /opt/ros/humble/setup.bash
source "$WORKSPACE/install/setup.bash"

echo "[1/5] Start detector node..."
ros2 run cone_detector yolo_detector --ros-args --params-file "$PARAMS_FILE" \
  > "$LOG_DIR/detector.log" 2>&1 &
DETECTOR_PID=$!

sleep 3
if ! kill -0 "$DETECTOR_PID" 2>/dev/null; then
  echo "Detector failed to start. See $LOG_DIR/detector.log" >&2
  exit 1
fi

echo "[2/5] Start bag play..."
ros2 bag play "$BAG_PATH" --clock --topics "$IMAGE_TOPIC" -r "$BAG_RATE" \
  > "$LOG_DIR/bag.log" 2>&1 &
BAG_PID=$!

echo "[3/5] Wait for first message..."
timeout 8s ros2 topic echo /yolo/cones --once > "$LOG_DIR/cones_once.log" 2>&1 || true

echo "[4/5] Measure /yolo/cones rate for ${DURATION}s..."
timeout "${DURATION}s" ros2 topic hz /yolo/cones > "$LOG_DIR/hz_cones.log" 2>&1 || true

if [[ "$CHECK_DEBUG" == "true" ]]; then
  echo "[4/5] Measure /yolo/debug_image rate for ${DURATION}s..."
  timeout "${DURATION}s" ros2 topic hz /yolo/debug_image > "$LOG_DIR/hz_debug.log" 2>&1 || true
fi

parse_rate() {
  local file="$1"
  awk '/average rate:/{rate=$3} END{if(rate=="") print "0"; else print rate}' "$file"
}

CONES_RATE="$(parse_rate "$LOG_DIR/hz_cones.log")"
HAS_CONES_MSG="false"
if grep -q "cones:" "$LOG_DIR/cones_once.log"; then
  HAS_CONES_MSG="true"
fi

DEBUG_RATE="N/A"
if [[ "$CHECK_DEBUG" == "true" ]]; then
  DEBUG_RATE="$(parse_rate "$LOG_DIR/hz_debug.log")"
fi

PASS="true"
if ! awk "BEGIN{exit !($CONES_RATE > 0.0)}"; then
  PASS="false"
fi
if [[ "$HAS_CONES_MSG" != "true" ]]; then
  PASS="false"
fi

SUMMARY="$LOG_DIR/summary.txt"
{
  echo "=== Orin Validation Summary ==="
  echo "workspace: $WORKSPACE"
  echo "params_file: $PARAMS_FILE"
  echo "bag_path: $BAG_PATH"
  echo "image_topic: $IMAGE_TOPIC"
  echo "bag_rate: $BAG_RATE"
  echo "sample_duration_sec: $DURATION"
  echo "cones_once_received: $HAS_CONES_MSG"
  echo "cones_avg_hz: $CONES_RATE"
  if [[ "$CHECK_DEBUG" == "true" ]]; then
    echo "debug_avg_hz: $DEBUG_RATE"
  fi
  echo "result: $PASS"
} | tee "$SUMMARY"

echo "[5/5] Logs saved in: $LOG_DIR"
if [[ "$PASS" != "true" ]]; then
  echo "Validation FAILED. Check detector.log, bag.log, hz logs." >&2
  exit 2
fi

echo "Validation PASSED."
