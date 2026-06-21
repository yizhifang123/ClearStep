#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_ROOT="$ROOT/native_macos"
APP_NAME="Depression EEG Real Case.app"
BUILD_DIR="$APP_ROOT/build"
APP_BUNDLE="$BUILD_DIR/$APP_NAME"
EXECUTABLE="$APP_BUNDLE/Contents/MacOS/RealCaseApp"

cd "$APP_ROOT"
swift build -c release --arch arm64

rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources"
cp "$APP_ROOT/.build/arm64-apple-macosx/release/RealCaseApp" "$EXECUTABLE"
cp "$APP_ROOT/Resources/real_case_bundle.json" "$APP_BUNDLE/Contents/Resources/real_case_bundle.json"
cp "$APP_ROOT/AppInfo.plist" "$APP_BUNDLE/Contents/Info.plist"
chmod +x "$EXECUTABLE"

echo "$APP_BUNDLE"
