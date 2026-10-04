#!/bin/zsh
set -euo pipefail
ROOT=${0:A:h:h}
TASK_TMP=$(mktemp -d "${TMPDIR:-/tmp}/m5ble-smoke.XXXXXX")
trap 'rm -rf "$TASK_TMP"' EXIT
python3 -m unittest discover -s "$ROOT/tests" -p 'test_*.py'
clang++ -std=c++17 -Wall -Wextra -Werror -fsanitize=address,undefined \
  "$ROOT/tests/recording_buffer_test.cc" -o "$TASK_TMP/buffer-test"
"$TASK_TMP/buffer-test" "$TASK_TMP/wire-fixture.bin"
clang++ -std=c++17 -Wall -Wextra -Werror -fsanitize=address,undefined \
  -I"$ROOT/tests/display_stubs" "$ROOT/tests/ui_battery_test.cc" -o "$TASK_TMP/ui-test"
"$TASK_TMP/ui-test"
clang++ -std=c++17 -Wall -Wextra -Werror -fsanitize=address,undefined \
  "$ROOT/tests/layer_norm_math_test.cc" -o "$TASK_TMP/layer-norm-test"
"$TASK_TMP/layer-norm-test"
clang++ -std=c++17 -Wall -Wextra -Werror -fsanitize=address,undefined \
  "$ROOT/tests/power_policy_test.cc" -o "$TASK_TMP/power-policy-test"
"$TASK_TMP/power-policy-test"
swift run --package-path "$ROOT/macos" M5BLECollectorSmokeTests "$TASK_TMP/wire-fixture.bin"
echo "All fast smoke tests passed. No device was accessed."
