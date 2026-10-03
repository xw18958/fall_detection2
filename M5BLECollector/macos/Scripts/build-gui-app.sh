#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
COLLECTOR_DIR=${SCRIPT_DIR:h}
DIST_DIR=${COLLECTOR_DIR}/dist
PACKAGE_DIR=${DIST_DIR}/M5\ Collector\ v1
APP_DIR=${PACKAGE_DIR}/M5\ Data\ Collector.app

cd "$COLLECTOR_DIR"
rm -rf "$DIST_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
cp READ_ME_FIRST.txt "$PACKAGE_DIR/READ_ME_FIRST.txt"

swift build -c release --product M5CollectorApp --arch arm64
BIN_DIR=$(swift build -c release --product M5CollectorApp --arch arm64 --show-bin-path)
cp "$BIN_DIR/M5CollectorApp" "$APP_DIR/Contents/MacOS/M5CollectorApp"
cp M5CollectorApp-Info.plist "$APP_DIR/Contents/Info.plist"

if [[ -n "${DEVELOPER_ID_APPLICATION:-}" ]]; then
    codesign --force --options runtime --timestamp --sign "$DEVELOPER_ID_APPLICATION" "$APP_DIR"
else
    codesign --force --sign - "$APP_DIR"
fi

plutil -lint "$APP_DIR/Contents/Info.plist"
codesign --verify --deep --strict "$APP_DIR"
ARCHS=$(lipo -archs "$APP_DIR/Contents/MacOS/M5CollectorApp")
[[ "$ARCHS" == "arm64" ]] || { echo "Unexpected architecture: $ARCHS" >&2; exit 1; }

ditto -c -k --keepParent "$PACKAGE_DIR" "$DIST_DIR/M5Collector-v1-arm64.zip"

echo "Built: $APP_DIR"
echo "Shareable ZIP: $DIST_DIR/M5Collector-v1-arm64.zip"
if [[ -z "${DEVELOPER_ID_APPLICATION:-}" ]]; then
    echo "Note: this build is ad-hoc signed. Use Developer ID signing/notarization before broad external distribution."
fi
