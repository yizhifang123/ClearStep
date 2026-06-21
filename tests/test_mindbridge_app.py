from pathlib import Path

import pytest

import llm


ROOT = Path(__file__).resolve().parents[1]


def test_patient_picker_only_allows_demo_patients():
    source = (ROOT / "app.py").read_text()

    assert "Build a custom patient instead" not in source
    assert "Generate a new patient" not in source
    assert "custom_seed" not in source
    assert "Custom synthetic patient" not in source
    assert "ARCHETYPE_LABELS" not in source
    assert "import random" not in source


def test_run_analysis_uses_fake_spinner_and_hardcoded_response():
    source = (ROOT / "app.py").read_text()

    assert "hardcoded_analysis_response" in source
    assert "time.sleep(2)" in source
    assert "calling Gemini" not in source
    assert "explain_with_gemini" not in source
    assert "Gemini API key" not in source
    assert "st.text_input" not in source
    assert "demo_seed=" not in source
    assert "Demo mode:" not in source


def test_physician_note_safety_caption_is_not_rendered():
    source = (ROOT / "app.py").read_text()

    assert "Synthetic / mock notes only" not in source
    assert "never paste a real patient's records" not in source
    assert "The AI grounds its explanation in this note" not in source


def test_explain_with_gemini_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(llm.NoKeyError) as excinfo:
        llm.explain_with_gemini("patient", "signal", "guidelines", "resources")

    message = str(excinfo.value)
    assert "live Gemini" in message
    assert "GEMINI_API_KEY" in message
    assert ".env" in message
    assert "paste" not in message
    assert "session key" not in message


def test_explain_with_gemini_forces_gemini_provider(monkeypatch):
    calls = {}

    def fake_explain_live(provider, key, patient_block, signal_block,
                          guidelines_block, resources_block, note_block=""):
        calls["provider"] = provider
        calls["key"] = key
        return {"clinician": {}, "family": {}}

    monkeypatch.setattr(llm, "_explain_live", fake_explain_live)

    out = llm.explain_with_gemini(
        "patient", "signal", "guidelines", "resources", api_key="session-key"
    )

    assert calls == {"provider": "gemini", "key": "session-key"}
    assert out["mode"] == "live"
    assert out["provider"] == "gemini"


def test_hardcoded_analysis_response_is_renderable():
    out = llm.hardcoded_analysis_response()

    assert out["mode"] == "hardcoded"
    assert out["provider"] == "presentation-demo"
    assert out["clinician"]["signal_summary"]
    assert out["clinician"]["evidence_to_consider"]
    assert out["clinician"]["evidence_to_consider"][0]["point"]
    assert out["clinician"]["evidence_to_consider"][0]["source"]
    assert out["clinician"]["safety_flags"]
    assert out["clinician"]["caveat"]
    assert out["family"]["summary"]
    assert out["family"]["what_matters"]
    assert out["family"]["next_steps"]
    assert out["family"]["next_steps"][0]["step"]
    assert out["family"]["next_steps"][0]["timeframe"]
    assert out["family"]["resource_ids"]
