"""Friendly demo patients for the UI: a human name, a one-line vignette, and a
SYNTHETIC physician note that the app can ground its explanation in (note RAG).

All notes are mock/synthetic — written to look realistic and jargon-heavy so the
family view has something to translate. No real patient data. The (seed, archetype)
drive the real EEG model; the cached demo text in llm.py is keyed by the same seed.
"""

DEMO_PATIENTS = {
    "Jordan, 15 — low mood and trouble sleeping": {
        "seed": 16,
        "archetype": "control_like",
        "vignette": "Brought in by a parent after about six weeks of low mood, "
                    "slipping grades, and pulling away from friends.",
        "note": (
            "Adolescent psychiatry intake — SYNTHETIC / MOCK note.\n"
            "15 y/o presenting with ~6 weeks of depressed mood, anhedonia, and initial "
            "insomnia. Declining academic performance and increasing social withdrawal. "
            "Appetite reduced.\n"
            "PHQ-A score 16 (moderately severe). Denies active SI; reports passive "
            "thoughts that 'things would be easier if I wasn't here,' no plan or intent. "
            "No prior psychiatric history.\n"
            "Plan: discuss evidence-based psychotherapy (CBT) +/- SSRI; safety plan "
            "reviewed with patient and caregiver; follow up in 1–2 weeks."
        ),
    },
    "Sam, 16 — irritable and avoiding school": {
        "seed": 350,
        "archetype": "control_like",
        "vignette": "Referred by a school counselor for irritability, fatigue, and a "
                    "few recent missed days.",
        "note": (
            "School-based counseling referral — SYNTHETIC / MOCK note.\n"
            "16 y/o with irritability, fatigue, and recent school avoidance (3 days "
            "missed this month). Mood reactive — still enjoys time with close friends. "
            "Sleep mildly disrupted.\n"
            "PHQ-A score 9 (mild). No SI. Notable family stressors (recent move).\n"
            "Plan: active monitoring, brief supportive counseling, re-screen with PHQ-A "
            "in 2 weeks."
        ),
    },
    "Alex, 14 — withdrawn, parent worried": {
        "seed": 235,
        "archetype": "melancholic_leaning",
        "vignette": "Parent reports Alex has been 'not themselves' for a couple of "
                    "months; Alex downplays it.",
        "note": (
            "Pediatric primary care note — SYNTHETIC / MOCK note.\n"
            "14 y/o; parent reports becoming withdrawn and 'not themselves' over ~2 "
            "months. Flat affect on exam with limited eye contact. Reports low energy "
            "but minimizes symptoms.\n"
            "PHQ-A score 12 (moderate) despite a low self-report. No SI elicited today.\n"
            "Plan: refer to behavioral health for full evaluation; caregiver "
            "psychoeducation provided."
        ),
    },
}


def patient_labels():
    return list(DEMO_PATIENTS.keys())


# Friendly names for the EEG model's hidden "archetype" (used in the custom builder).
ARCHETYPE_LABELS = {
    "control_like": "Typical / control-like pattern",
    "melancholic_leaning": "Melancholic-leaning pattern (illustrative)",
    "atypical_leaning": "Atypical-leaning pattern (illustrative)",
}
