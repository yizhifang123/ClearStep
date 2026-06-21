#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_EXEC="$ROOT/mindbridge_macos/build/MindBridge.app/Contents/MacOS/MindBridgeApp"

if [ ! -x "$APP_EXEC" ]; then
  "$ROOT/mindbridge_macos/build_app.sh" >/dev/null
fi

exec "$APP_EXEC"
