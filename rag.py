"""Retrieval (the 'R' in RAG) over two curated, PUBLIC corpora:

  - guidelines.json : public clinical-guideline snippets, used to GROUND the
    clinician-facing evidence so the AI cites real guidance instead of inventing it.
  - resources.json  : public support services (988, 211, ...), used to give the
    family real, linked next-step help.

Both are small curated corpora matched with plain TF-IDF cosine similarity —
transparent and easy to audit. We never retrieve over a real patient's data.
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).parent


def _load(name):
    with open(HERE / name, encoding="utf-8") as f:
        return json.load(f)


class TfidfRetriever:
    """Generic TF-IDF retriever over a list of dict records."""

    def __init__(self, records, text_fields, always_show_key=None):
        self.records = records
        self.always_show_key = always_show_key
        corpus = [" ".join(str(r.get(f, "")) for f in text_fields) for r in records]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query_text, top_k=4):
        scores = cosine_similarity(
            self.vectorizer.transform([query_text]), self.matrix
        )[0]
        order = sorted(range(len(self.records)), key=lambda i: scores[i], reverse=True)

        chosen, seen = [], set()
        if self.always_show_key:  # safety floor (e.g. crisis lines) first
            for i, r in enumerate(self.records):
                if r.get(self.always_show_key):
                    chosen.append(r)
                    seen.add(i)
        for i in order:
            if i in seen:
                continue
            if len([c for c in chosen if not c.get(self.always_show_key)]) >= top_k:
                break
            chosen.append(self.records[i])
            seen.add(i)
        return chosen

    def by_ids(self, ids):
        idx = {r["id"]: r for r in self.records}
        return [idx[i] for i in ids if i in idx]

    def always_show(self):
        if not self.always_show_key:
            return []
        return [r for r in self.records if r.get(self.always_show_key)]


def guideline_retriever():
    return TfidfRetriever(_load("guidelines.json"), ["topic", "text", "tags"])


def resource_retriever():
    return TfidfRetriever(
        _load("resources.json"), ["name", "description", "tags"], always_show_key="always_show"
    )


def retrieve_note_excerpts(note_text, query, top_k=4):
    """Document-level RAG over a single (synthetic) physician note.

    Splits the note into chunks and TF-IDF-retrieves the chunks most relevant to the
    query, so a long note is distilled to what matters before grounding the model.
    Returns chunks in their original order (readability). Empty list for a blank note.
    """
    import re

    chunks = [c.strip() for c in re.split(r"(?<=[.\n])\s+", note_text) if len(c.strip()) > 15]
    if not chunks:
        return []
    if len(chunks) <= top_k:
        return chunks
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(chunks)
    scores = cosine_similarity(vec.transform([query]), matrix)[0]
    keep = sorted(sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k])
    return [chunks[i] for i in keep]


def guidelines_block(records):
    return "\n".join(
        f"- id={r['id']} | {r['topic']}: {r['text']} (Source: {r['source']})"
        for r in records
    )


def resources_block(records):
    return "\n".join(
        f"- id={r['id']} | {r['name']} | {r['contact']} | {r['description']}"
        for r in records
    )
