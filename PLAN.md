# ClearStep — Winning Plan (post-council)

**Goal:** Win/place in the USAII Global AI Hackathon 2026, High School track, Challenge 1
"Help is Hard to Find." Deadline: tonight, 11:59 PM ET.

**Decision (post-council):** Build the **Merge** on the **ClearStep base**. Product name =
**ClearStep** (repo already named ClearStep; on-brief; sheds the "MindBridge + medical"
screener risk). The depression/EEG model is **benched as a deliberate honesty cameo**, not
run in the product.

---

## The product in one sentence

**A stressed parent or teen pastes any confusing mental-health document — an ER discharge,
an insurance denial, a school counselor letter — and ClearStep turns it into plain
language, what matters most, a clear checklist, and *trustworthy* next steps — while being
honest about what it does not know and refusing to ever diagnose.**

- **User (specific, non-expert):** a stressed parent/teen holding a confusing mental-health
  document. (Anchored on the user's real personal story in the video — see §Video.)
- **Why AI, not a web search (the qualifier gap, answered):** a search engine can't read
  *this* letter, judge *its* urgency, rewrite *its* jargon at a 6th-grade level, and tell
  you *which* of the results is trustworthy for *your* situation. That's per-document
  NLP + summarization + classification + retrieval — not something a search bar does.

## The signature (what makes a judge remember us) — "honest AI"

Three things, together, are the distinctiveness the council said we lack:

1. **The "we built a real AI and deliberately refused to use it" cameo.** Our team trained a
   real depression-detection model on brain-wave data (LOSO AUC ~0.95). Then we **left it out
   of ClearStep** — because a number can't tell a scared parent what their kid needs, and
   "your child is 82% depressed" does harm, not help. *The most responsible thing our AI does
   is refuse to guess.* (Static "About our AI" section + the video's climax. Converts the
   EEG work from a liability into our signature responsible-AI statement.)
2. **The trust ribbon.** Every resource we surface shows *who runs it, is it free, is it
   confidential, is it available now* — because "help is hard to find" is really "I don't
   know which of these to trust."
3. **The "what we don't know" stamp.** Every result is labeled: *"This is general
   information, not advice about your specific situation,"* plus a first-class **"ClearStep
   will never diagnose you or your child"** banner and a one-tap route to a human.

---

## How this answers EVERY council issue

| Council issue | Fix in this plan |
|---|---|
| Off-brief: clinician-primary, EEG-headline (bleeds Problem Understanding 30%) | No clinician, no EEG in the product. The only user is the stressed family. |
| EEG model is a liability (non-transferable, "medical-decision tool the brief banned") | Model is **benched** and shown only as a *refusal* — restraint, not a medical decision. |
| Honesty engine is the rare asset → Responsible AI Award | It is now the **signature** (cameo + trust ribbon + "what we don't know"). |
| Round-1 text fields are the gate | All 5 rewritten family-first, honesty-forward (Phase 4). |
| Video must cold-open on a specific person | Cold-open on the user's **real personal story** (Phase 5). |
| Demote the clinician view | Gone entirely. |
| Distinctiveness / "feels basic" (the core fear) | The "refused to use our own model" twist + trust ribbon = a story judges retell. |
| Deploy: timebox / don't over-polish | ClearStep base = 3 deps, trivial Streamlit Cloud deploy (Phase 6). |
| Name "MindBridge" + medical framing trips the screener | Name = ClearStep; no medical framing. |
| "Why AI over a web search" (qualifier gap) | Answered explicitly in copy + video. |
| Specific user, not vague | Named real person in the first 30 seconds. |
| Provider naming inconsistency (Claude vs Gemini) | Port MindBridge's provider-agnostic `llm.py`; standardize copy on the provider actually used (Gemini, free tier). |

---

## Build plan (phased; ~6 hours)

### Phase 0 — Base reconciliation (~20 min)
- Work in `/Users/admin/Documents/clearstep` (the base).
- Port MindBridge's **provider-agnostic `llm.py`** (Claude / OpenAI / **Gemini**) so the
  user's free Gemini key works; add `load_dotenv()` + `transport="rest"` Gemini fix.
- Keep ClearStep's `analyze()` JSON shape (summary / key_points / checklist / watch_for /
  resource_ids) and `DEMO_RESPONSES` key-free demo.

### Phase 1 — The honesty/trust layer (the standout) (~2 hr)
1. **Trust ribbon.** Add structured fields to every `resources.json` record:
   `run_by`, `free` (bool), `confidential` (bool), `availability` (e.g. "24/7"). Render a
   compact ribbon on each resource card (who runs it · free · confidential · available now).
2. **"What we don't know" stamp.** A consistent per-result label: *"General information, not
   advice about your specific situation."*
3. **"ClearStep will never diagnose you or your child"** — a first-class UI banner + a
   one-tap "talk to a real person" (988 / Crisis Text Line) on every result.
4. Reuse ClearStep's existing always-on crisis line + "original text shown" + source links.

### Phase 2 — The EEG honesty cameo (~45 min)
- A static **"About our AI — and what it refuses to do"** section (expander or short page):
  the trained-model story + real metrics (AUC ~0.95, Mumtaz, LOSO, p<0.001), then the
  refusal: *we benched it on purpose.* **No `model.pkl`, no numpy/joblib in the deploy** —
  narrative + numbers only. (Lift numbers from `mindbridge/metrics.json`.)

### Phase 3 — Examples + demo coverage (~30 min)
- Add an **insurance-denial** example (brief explicitly lists it) + 1 more; write matching
  `DEMO_RESPONSES` so all examples run key-free for the video.

### Phase 4 — The 5 Round-1 fields, family-first (the gate) (~60 min)
- Rewrite all 5 in `SUBMISSION.md`: Elevator pitch, About, AI Architecture, Human-in-the-Loop,
  Responsible AI Guardrail — lead with the real user, foreground the honesty signature,
  explicitly answer "why AI not a web search," zero medical-decision framing. Keep under char
  limits. (These gate whether a human ever sees the project — highest EV writing.)

### Phase 5 — Video script (~45 min to script; user records)
- Cold-open (0–25s) on the **real personal story**. Hero shot: paste a confusing doc → plain
  language + checklist + trust ribbon. Climax: the "we built a real model and refused to use
  it" twist. Close on "confusion → clarity → action." (See §Video once story is provided.)

### Phase 6 — Deploy + finalize (~30 min, timeboxed)
- Deploy ClearStep to **Streamlit Community Cloud** (3 deps, no model → easy); add
  `GEMINI_API_KEY` via Streamlit Secrets. If not green in 20 min → ship a screen-recording +
  "runs locally" and move on.
- Repo: push merged ClearStep to GitHub `main` (preserve MindBridge on a `mindbridge`
  branch). Confirm repo name = ClearStep. Update README.

---

## Risks & explicit CUTS (what NOT to do)
- **Do NOT** put the EEG model live in the product (reintroduces the banned medical-decision
  framing). Cameo only.
- **Do NOT** add PDF/photo/OCR upload — brief says "pastes"; it's a time-trap.
- **Do NOT** over-polish the app or chase features after Phase 4 — the text fields + video are
  where points live.
- **Timebox the deploy** to 20 min; a clean recording is functionally equivalent for judges.

## Open items (need from the user)
1. **The real personal story** for the cold-open (who / the confusing moment / what they
   needed / the feeling). The single highest-value input.
2. **Name confirmation:** recommend keeping **ClearStep** (on-brief, repo matches). Override?
