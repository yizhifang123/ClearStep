from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "mindbridge_macos"


def test_mindbridge_macos_swift_wrapper_exists():
    package = APP / "Package.swift"
    source = APP / "Sources" / "MindBridgeApp" / "main.swift"

    assert package.exists()
    assert source.exists()

    text = source.read_text()
    assert "import WebKit" in text
    assert "WKWebView" in text
    assert "StreamlitServer" in text
    assert "app.py" in text
    assert "GEMINI_API_KEY" in text
    assert "ANTHROPIC_API_KEY" in text
    assert "OPENAI_API_KEY" in text


def test_mindbridge_macos_build_script_creates_app_bundle():
    script = APP / "build_app.sh"
    text = script.read_text()

    assert "swift build" in text
    assert "MindBridge.app" in text
    assert "Contents/MacOS" in text
    assert "RepoRoot.txt" in text


def test_mindbridge_macos_plist_names_app():
    plist = (APP / "AppInfo.plist").read_text()

    assert "<string>MindBridge</string>" in plist
    assert "<string>MindBridgeApp</string>" in plist
    assert "com.local.mindbridge.webapp" in plist


def test_mindbridge_macos_has_no_reload_toolbar_button():
    source = (APP / "Sources" / "MindBridgeApp" / "main.swift").read_text()

    assert 'Button("Reload")' not in source
    assert "ToolbarItemGroup" not in source
