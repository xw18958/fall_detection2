#!/bin/zsh
set -euo pipefail
ROOT=${0:A:h:h}
TASK_TMP=$(mktemp -d "${TMPDIR:-/tmp}/m5ble-smoke.XXXXXX")
trap 'rm -rf "$TASK_TMP"' EXIT
python3 -m unittest discover -s "$ROOT/tests" -p 'test_*.py'
clang++ -std=c++17 -Wall -Wextra -Werror -fsanitize=address,undefined \
  "$ROOT/tests/recording_buffer_test.cc" -o "$TASK_TMP/buffer-test"
"$TASK_TMP/buffer-test" "$TASK_TMP/wire-fixture.bin"
swift run --package-path "$ROOT/macos" M5BLECollectorSmokeTests "$TASK_TMP/wire-fixture.bin"
echo "All fast smoke tests passed. No device was accessed."
