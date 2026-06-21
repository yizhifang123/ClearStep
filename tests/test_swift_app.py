from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "native_macos"


def test_swift_package_and_app_source_exist():
    package = APP / "Package.swift"
    source = APP / "Sources" / "RealCaseApp" / "main.swift"

    assert package.exists()
    assert source.exists()

    package_text = package.read_text()
    source_text = source.read_text()
    assert "RealCaseApp" in package_text
    assert "SwiftUI" in source_text
    assert "Streamlit" not in source_text
    assert "Python" not in source_text


def test_real_case_bundle_contains_native_runtime_data_only():
    bundle = APP / "Resources" / "real_case_bundle.json"
    data = json.loads(bundle.read_text())

    assert data["schemaVersion"] == 1
    assert data["metrics"]["nSubjects"] == 58
    assert data["metrics"]["nMdd"] == 30
    assert len(data["subjects"]) == 58
    assert {"id", "label", "probabilityMDD", "contributions"} <= set(data["subjects"][0])
    assert len(data["subjects"][0]["contributions"]) == 8


def test_native_build_script_targets_macos_bundle():
    script = APP / "build_app.sh"
    text = script.read_text()

    assert "swift build" in text
    assert "Depression EEG Real Case.app" in text
    assert "Contents/MacOS" in text


def test_native_env_launcher_preserves_gemini_key_without_embedding_it():
    script = APP / "run_with_env.sh"
    text = script.read_text()

    assert "GEMINI_API_KEY" in text
    assert "Contents/MacOS/RealCaseApp" in text
    assert "AIza" not in text


def test_native_dashboard_is_scrollable_and_buttons_have_top_space():
    source = (APP / "Sources" / "RealCaseApp" / "main.swift").read_text()

    assert "ScrollView(.vertical" in source
    assert ".scrollIndicators(.visible)" in source
    assert "decisionButtonTopSpacer" in source
    assert ".padding(.top, decisionButtonTopSpacer)" in source


def test_native_app_uses_gemini_without_hardcoding_secret():
    source = (APP / "Sources" / "RealCaseApp" / "main.swift").read_text()

    assert "struct GeminiClient" in source
    assert "GEMINI_API_KEY" in source
    assert "generativelanguage.googleapis.com/v1beta/models/" in source
    assert ":generateContent" in source
    assert "AIza" not in source


def test_gemini_features_are_limited_to_explanation_layer():
    source = (APP / "Sources" / "RealCaseApp" / "main.swift").read_text()

    assert "Generate Summary" in source
    assert "Caveats" in source
    assert "Handoff Note" in source
    assert "Do not diagnose" in source
    assert "Do not recommend medication" in source
    assert "research prototype" in source
