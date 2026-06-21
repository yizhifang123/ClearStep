#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_EXEC="$ROOT/native_macos/build/Depression EEG Real Case.app/Contents/MacOS/RealCaseApp"

if [ ! -x "$APP_EXEC" ]; then
  "$ROOT/native_macos/build_app.sh" >/dev/null
fi

if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "GEMINI_API_KEY is not set. The app will still launch and allow a session key paste."
fi

exec "$APP_EXEC"
