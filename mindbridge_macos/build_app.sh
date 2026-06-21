#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_ROOT="$ROOT/mindbridge_macos"
APP_NAME="MindBridge.app"
BUILD_DIR="$APP_ROOT/build"
APP_BUNDLE="$BUILD_DIR/$APP_NAME"
EXECUTABLE="$APP_BUNDLE/Contents/MacOS/MindBridgeApp"

cd "$APP_ROOT"
swift build -c release --arch arm64

rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources"
cp "$APP_ROOT/.build/arm64-apple-macosx/release/MindBridgeApp" "$EXECUTABLE"
cp "$APP_ROOT/AppInfo.plist" "$APP_BUNDLE/Contents/Info.plist"
printf "%s\n" "$ROOT" > "$APP_BUNDLE/Contents/Resources/RepoRoot.txt"
chmod +x "$EXECUTABLE"

echo "$APP_BUNDLE"
