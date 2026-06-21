"""Retrieval over a curated, PUBLIC mental-health resource directory.

This is the "R" in RAG. We never retrieve over a person's private data — only
over a small, vetted list of public services (988, Crisis Text Line, 211, ...).
Each result carries a source link so the reader can verify it themselves.

Retrieval is a plain TF-IDF cosine match: cheap, transparent, and good enough
for a small curated corpus. The top matches are handed to the language model so
its answer is *grounded* in real, linked resources instead of invented ones.
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RESOURCES_PATH = Path(__file__).parent / "resources.json"


def load_resources():
    """Load the curated resource directory from disk."""
    with open(RESOURCES_PATH, encoding="utf-8") as f:
        return json.load(f)


class ResourceRetriever:
    """Ranks public resources by relevance to a document via TF-IDF cosine."""

    def __init__(self, resources=None):
        self.resources = resources if resources is not None else load_resources()
        # Index on name + description + tags so a query can match any of them.
        corpus = [
            f"{r['name']} {r['description']} {r['tags']}" for r in self.resources
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query_text: str, top_k: int = 5):
        """Return the top_k most relevant resources, plus any 'always_show' ones.

        Crisis resources flagged always_show (988, Crisis Text Line) are forced
        into the set regardless of the match score — a safety floor, not a
        ranking decision.
        """
        scores = cosine_similarity(
            self.vectorizer.transform([query_text]), self.matrix
        )[0]
        ranked = sorted(
            range(len(self.resources)), key=lambda i: scores[i], reverse=True
        )

        chosen, seen = [], set()
        # Safety floor first.
        for i, r in enumerate(self.resources):
            if r.get("always_show"):
                chosen.append(r)
                seen.add(i)
        # Then the best matches.
        for i in ranked:
            if i in seen:
                continue
            if len(chosen) >= top_k + sum(r.get("always_show", False) for r in self.resources):
                break
            chosen.append(self.resources[i])
            seen.add(i)
        return chosen

    @staticmethod
    def to_block(resources) -> str:
        """Render resources as a compact text block for the model prompt."""
        lines = []
        for r in resources:
            lines.append(
                f"- id={r['id']} | {r['name']} | {r['contact']} | {r['description']}"
            )
        return "\n".join(lines)

    def by_ids(self, ids):
        """Look up full resource records by id (for rendering the result UI)."""
        index = {r["id"]: r for r in self.resources}
        return [index[i] for i in ids if i in index]

    def always_show(self):
        """The crisis resources that appear on every result, no matter what."""
        return [r for r in self.resources if r.get("always_show")]
