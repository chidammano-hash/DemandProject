"""Cold-start DFU routing via metadata k-NN.

Gen-4 Stream G / AI-2 Phase 1 (scaffold).

A newly-onboarded DFU has no sales history (or fewer than
``cold_start_min_months`` months) so the champion tree ensemble refuses it
and the rolling-mean fallback bats a poor baseline. This module finds the
K nearest existing DFUs by a simple standardized feature vector over
``(category, brand, abc_class, avg_price)`` — callers can then transplant
a seasonality index, prepend the neighbor histories as a prompt prefix
for the FM spine, or seed the ML cluster.

No embeddings. No ANN index. Cosine distance over a compact numpy matrix
is plenty for the onboarding use case (thousands of DFUs at most).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DFUMetadata:
    """Minimal metadata vector used for cold-start k-NN."""

    dfu_id: str
    category: str
    brand: str
    abc_class: str
    avg_price: float


@dataclass(frozen=True)
class NearestNeighbor:
    """One result row from ``nearest_neighbors``."""

    dfu_id: str
    similarity: float  # cosine similarity in [-1, 1]; higher = closer
    metadata: DFUMetadata


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

# Known ABC buckets receive an ordinal embedding so A/B/C carry a
# natural ordering. Any unknown class falls to the middle bucket.
_ABC_ORDINAL: dict[str, float] = {"A": 1.0, "B": 0.5, "C": 0.0, "D": -0.5}


def _encode_one(meta: DFUMetadata, categories: Sequence[str], brands: Sequence[str]) -> np.ndarray:
    """Encode a single DFU into a fixed-length numeric vector.

    Layout: [one-hot(category), one-hot(brand), abc_ordinal, avg_price].
    """
    cat_vec = np.zeros(len(categories), dtype=float)
    if meta.category in categories:
        cat_vec[categories.index(meta.category)] = 1.0
    brand_vec = np.zeros(len(brands), dtype=float)
    if meta.brand in brands:
        brand_vec[brands.index(meta.brand)] = 1.0
    abc = _ABC_ORDINAL.get(meta.abc_class, 0.25)
    price = float(meta.avg_price)
    return np.concatenate([cat_vec, brand_vec, [abc], [price]])


def _standardize(matrix: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-std per column; constant columns are zeroed."""
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    # Avoid divide-by-zero on constant columns.
    safe_stds = np.where(stds == 0.0, 1.0, stds)
    out = (matrix - means) / safe_stds
    out[:, stds == 0.0] = 0.0
    return out


def _cosine_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Return cosine similarity between ``query`` and every row of ``corpus``."""
    # Both are already standardized; guard zero-norm rows.
    q_norm = np.linalg.norm(query)
    c_norms = np.linalg.norm(corpus, axis=1)
    denom = q_norm * c_norms
    sims = np.where(denom > 0, corpus @ query / np.where(denom == 0.0, 1.0, denom), 0.0)
    return sims


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def nearest_neighbors(
    new_dfu: DFUMetadata,
    candidates: Sequence[DFUMetadata],
    *,
    k: int = 5,
) -> list[NearestNeighbor]:
    """Return top-K most-similar existing DFUs by cosine similarity.

    Args:
        new_dfu: the newly-onboarded DFU's metadata (no sales history).
        candidates: existing DFUs with sufficient history that we can
                    reuse as analogs.
        k: number of neighbors to return. Fewer are returned if the
           candidate set is smaller.

    Raises:
        ValueError: when ``candidates`` is empty.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")
    if k <= 0:
        raise ValueError("k must be positive")

    # Build the shared category / brand vocabulary from the union.
    categories = sorted({c.category for c in candidates} | {new_dfu.category})
    brands = sorted({c.brand for c in candidates} | {new_dfu.brand})

    corpus_rows = np.stack([_encode_one(c, categories, brands) for c in candidates])
    query_row = _encode_one(new_dfu, categories, brands).reshape(1, -1)
    combined = np.vstack([corpus_rows, query_row])
    standardized = _standardize(combined)
    corpus_std = standardized[:-1]
    query_std = standardized[-1]

    sims = _cosine_similarity(query_std, corpus_std)
    top_idx = np.argsort(-sims)[: min(k, len(candidates))]
    return [
        NearestNeighbor(
            dfu_id=candidates[i].dfu_id,
            similarity=float(sims[i]),
            metadata=candidates[i],
        )
        for i in top_idx
    ]


def build_prompt_prefix(neighbors: Sequence[NearestNeighbor]) -> str:
    """Format a short few-shot prefix for an FM cold-start prompt.

    The prefix is intentionally compact — real Chronos/LLM pipelines
    should supplement this with the neighbors' sales histories.
    """
    lines = ["# Cold-start neighbors (k-NN by metadata)"]
    for nb in neighbors:
        lines.append(
            f"- {nb.dfu_id} (sim={nb.similarity:.3f}): "
            f"category={nb.metadata.category}, brand={nb.metadata.brand}, "
            f"abc={nb.metadata.abc_class}, avg_price={nb.metadata.avg_price:.2f}"
        )
    return "\n".join(lines)


__all__ = [
    "DFUMetadata",
    "NearestNeighbor",
    "nearest_neighbors",
    "build_prompt_prefix",
]
