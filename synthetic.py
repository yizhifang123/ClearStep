"""Layer B — SYNTHETIC patient generator. NOT real data; illustrative only.

Generates internally-consistent synthetic **adolescent-male** cases (the
project's target population — which no open EEG dataset covers) with:
  - EEG feature summaries (same feature names/order as Layer A)
  - hormonal markers (cortisol) calibrated to published reference ranges
  - a hidden archetype used only to keep the synthetic signals self-consistent

Everything here is simulated to illustrate the *envisioned* workflow. It is not
a model output and not validated. See RESEARCH.md §1.3 for why cortisol cannot
subtype a real patient.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Salivary cortisol reference anchors (nmol/L) — (mean, sd) — calibrated to
# typical adolescent diurnal values (CIRCORT/MIDUS-style). Simulated.
CORTISOL_REF = {
    "morning_nmol_l": (12.0, 6.0),    # at waking
    "car_increase_pct": (50.0, 25.0),  # cortisol awakening response (% rise, 30-45 min)
    "evening_nmol_l": (3.0, 1.5),
    "diurnal_slope": (-0.55, 0.20),    # log-decline per hour (steeper = healthier)
}

BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]
EEG_FEATURE_NAMES = (
    [f"rel_{b}_global" for b in BAND_ORDER]
    + [f"rel_{b}_frontal" for b in BAND_ORDER]
    + [f"rel_{b}_posterior" for b in BAND_ORDER]
    + ["faa_F4_F3"]
)

ARCHETYPES = ["control_like", "melancholic_leaning", "atypical_leaning"]
_ARCH_LABEL = {
    "control_like": "control-like",
    "melancholic_leaning": "melancholic-leaning (illustrative)",
    "atypical_leaning": "atypical-leaning (illustrative)",
}

# Curated cases for a smooth live demo: (seed, archetype). Illustrative only.
CURATED_CASES = {
    "Case A — melancholic leaning (high cortisol)": (101, "melancholic_leaning"),
    "Case B — atypical leaning (blunted cortisol)": (202, "atypical_leaning"),
    "Case C — control-like": (303, "control_like"),
}


@dataclass
class SyntheticPatient:
    case_id: str
    age: int
    sex: str
    archetype: str
    eeg: dict
    cortisol: dict

    def to_dict(self) -> dict:
        return asdict(self)

    def eeg_vector(self, feature_names) -> list[float]:
        """EEG features ordered to match a given Layer A feature_names list."""
        return [float(self.eeg.get(n, np.nan)) for n in feature_names]


def generate_synthetic_patient(seed: int, archetype: str | None = None) -> SyntheticPatient:
    """Deterministic synthetic case. `seed` makes it reproducible (no RNG globals)."""
    rng = np.random.default_rng(seed)
    arch = archetype or str(rng.choice(ARCHETYPES))

    # Relative band power per region via Dirichlet (guarantees sums to 1).
    base = np.array([0.30, 0.18, 0.22, 0.20, 0.10])  # delta..gamma
    shift = {
        "control_like": np.zeros(5),
        "melancholic_leaning": np.array([0.05, 0.06, -0.02, -0.05, -0.04]),
        "atypical_leaning": np.array([0.03, 0.04, 0.00, -0.04, -0.03]),
    }[arch]

    def region_rel():
        conc = np.clip(base + shift, 0.02, None) * 40.0
        return rng.dirichlet(conc)

    eeg: dict[str, float] = {}
    for region in ("global", "frontal", "posterior"):
        vals = region_rel()
        for b, v in zip(BAND_ORDER, vals):
            eeg[f"rel_{b}_{region}"] = float(v)
    # FAA: depression-leaning -> more negative (greater relative right activation)
    faa_mean = {"control_like": 0.05, "melancholic_leaning": -0.20,
                "atypical_leaning": -0.08}[arch]
    eeg["faa_F4_F3"] = float(rng.normal(faa_mean, 0.12))

    # Cortisol consistent with archetype: melancholic ~ hyper; atypical ~ blunted.
    mult = {"control_like": (1.0, 1.0, 1.0),
            "melancholic_leaning": (1.5, 1.2, 1.8),
            "atypical_leaning": (0.6, 0.5, 0.9)}[arch]

    def draw(key, m):
        mean, sd = CORTISOL_REF[key]
        return float(max(0.1, rng.normal(mean * m, sd)))

    cortisol = {
        "morning_nmol_l": draw("morning_nmol_l", mult[0]),
        "car_increase_pct": draw("car_increase_pct", mult[1]),
        "evening_nmol_l": draw("evening_nmol_l", mult[2]),
        "diurnal_slope": float(rng.normal(*CORTISOL_REF["diurnal_slope"])
                               * (0.6 if arch == "melancholic_leaning" else 1.0)),
    }
    return SyntheticPatient(
        case_id=f"SYN-{seed:03d}", age=int(rng.integers(14, 19)), sex="male",
        archetype=arch, eeg=eeg, cortisol=cortisol,
    )


def archetype_label(arch: str) -> str:
    return _ARCH_LABEL.get(arch, arch)


def illustrative_subtype_leaning(cortisol: dict) -> dict:
    """ILLUSTRATIVE ONLY — soft, explicitly-uncertain leaning from morning cortisol.

    Group-level associations (melancholic ~ hypercortisolism, atypical ~ blunted)
    are NOT a clinical test: huge individual overlap, and data-driven subtyping
    failed on cortisol (RESEARCH.md §1.3). This logistic mapping exists purely to
    demonstrate the workflow, never to subtype a real patient.
    """
    z = (cortisol["morning_nmol_l"] - 12.0) / 6.0
    p_mel = 1.0 / (1.0 + np.exp(-z))
    return {"melancholic": float(p_mel), "atypical": float(1.0 - p_mel)}
