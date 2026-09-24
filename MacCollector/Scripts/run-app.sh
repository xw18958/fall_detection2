#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
COLLECTOR_DIR=${SCRIPT_DIR:h}
APP_DIR=${COLLECTOR_DIR}/.build/AirPodsMotionCollector.app

cd "$COLLECTOR_DIR"
swift build -c release

mkdir -p "$APP_DIR/Contents/MacOS"
cp ".build/release/AirPodsMotionCollector" "$APP_DIR/Contents/MacOS/AirPodsMotionCollector"
cp "Info.plist" "$APP_DIR/Contents/Info.plist"
codesign --force --sign - "$APP_DIR"

exec "$APP_DIR/Contents/MacOS/AirPodsMotionCollector" "$@"
