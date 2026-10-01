#!/bin/zsh
set -euo pipefail
COLLECTOR_ROOT=${0:A:h}
cd "$COLLECTOR_ROOT/macos"
exec ./Scripts/run-app.sh "$@"
