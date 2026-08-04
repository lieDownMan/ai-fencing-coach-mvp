#!/usr/bin/env bash
# Build, boot, launch, and screenshot the AI Fencing Coach iOS app in the Simulator.
# Run from the `frontend/` directory (or pass FRONTEND_DIR).
set -euo pipefail

FRONTEND_DIR="${FRONTEND_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DEVICE_NAME="${DEVICE_NAME:-AI Fencing Coach Test}"
DEVICE_TYPE="${DEVICE_TYPE:-com.apple.CoreSimulator.SimDeviceType.iPhone-16}"
RUNTIME="${RUNTIME:-com.apple.CoreSimulator.SimRuntime.iOS-26-5}"
LOG_FILE="${LOG_FILE:-/tmp/flutter_run_ios.log}"

cmd="${1:-}"
shift || true

find_or_create_device() {
  local udid
  udid=$(xcrun simctl list devices | grep -F "$DEVICE_NAME (" | grep -v unavailable | head -1 | sed -E 's/.*\(([0-9A-F-]{36})\).*/\1/')
  if [ -z "$udid" ]; then
    udid=$(xcrun simctl create "$DEVICE_NAME" "$DEVICE_TYPE" "$RUNTIME")
  fi
  echo "$udid"
}

case "$cmd" in
  boot)
    udid=$(find_or_create_device)
    xcrun simctl boot "$udid" 2>/dev/null || true
    echo "$udid"
    ;;

  build)
    cd "$FRONTEND_DIR"
    flutter pub get
    flutter build ios --no-codesign --debug
    ;;

  run)
    udid="${1:?usage: driver.sh run <udid>}"
    cd "$FRONTEND_DIR"
    nohup flutter run -d "$udid" --debug > "$LOG_FILE" 2>&1 &
    echo "started flutter run (PID $!), log: $LOG_FILE"
    for i in $(seq 1 90); do
      grep -qE "A Dart VM Service|Lost connection|Error launching|Could not launch" "$LOG_FILE" 2>/dev/null && break
      sleep 2
    done
    tail -5 "$LOG_FILE"
    ;;

  screenshot)
    udid="${1:?usage: driver.sh screenshot <udid> <out.png>}"
    out="${2:?usage: driver.sh screenshot <udid> <out.png>}"
    mkdir -p "$(dirname "$out")"
    xcrun simctl io "$udid" screenshot "$out"
    ;;

  record)
    udid="${1:?usage: driver.sh record <udid> <out.mp4> [seconds]}"
    out="${2:?usage: driver.sh record <udid> <out.mp4> [seconds]}"
    seconds="${3:-10}"
    mkdir -p "$(dirname "$out")"
    xcrun simctl io "$udid" recordVideo --codec h264 -f "$out" &
    rec_pid=$!
    sleep "$seconds"
    kill -INT "$rec_pid"
    wait "$rec_pid" 2>/dev/null || true
    echo "recorded ${seconds}s to $out"
    ;;

  shutdown)
    udid="${1:?usage: driver.sh shutdown <udid>}"
    xcrun simctl shutdown "$udid" || true
    ;;

  *)
    echo "usage: driver.sh {boot|build|run <udid>|screenshot <udid> <out.png>|record <udid> <out.mp4> [seconds]|shutdown <udid>}" >&2
    exit 1
    ;;
esac
