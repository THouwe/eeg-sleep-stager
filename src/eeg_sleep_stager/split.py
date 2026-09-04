"""Subject-wise, seeded train/val/test split -> splits.json (S5).

The whole honesty argument of this project rests here: splits are made by
``subject_id``, never by epoch, so no epoch from a test subject can leak into
training. The split is deterministic under a seed and is checked so that every
split covers all five AASM stages (trivially satisfied on Sleep-EDF, where each
subject-night contains every stage, but asserted so a degenerate subset fails
loudly rather than silently training on a missing class).

`assign_splits` is pure and unit-tested; `make_split` reads the epoch store
(via pandas/pyarrow — no Spark needed) and writes splits.json.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Mapping, Optional

from .config import N_CLASSES, Config

FULL_STAGES = frozenset(range(N_CLASSES))


def _alloc(n: int, frac: float) -> int:
    """Number of subjects for a split fraction (at least 1 when frac > 0)."""
    if frac <= 0.0:
        return 0
    return max(1, int(round(n * frac)))


def assign_splits(
    subject_stages: Mapping[str, set[int]],
    seed: int,
    val_frac: float,
    test_frac: float,
) -> dict[str, list[str]]:
    """Assign subjects to train/val/test deterministically and disjointly.

    Args:
        subject_stages: subject_id -> set of stage ids present for that subject.
        seed: RNG seed; the same seed always yields the same assignment.
        val_frac, test_frac: fraction of subjects for val / test.

    Returns:
        {"train": [...], "val": [...], "test": [...]} with sorted subject ids.

    Raises:
        ValueError: fewer than 3 subjects, or a split fails to cover all stages.
    """
    subjects = sorted(subject_stages)
    n = len(subjects)
    if n < 3:
        raise ValueError(
            f"need at least 3 subjects for a train/val/test split, got {n}"
        )

    # Deterministic shuffle from a fixed base order.
    order = subjects[:]
    random.Random(seed).shuffle(order)

    n_test = _alloc(n, test_frac)
    n_val = _alloc(n, val_frac)
    # Guarantee a non-empty training split.
    while n - n_test - n_val < 1:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1

    test = order[:n_test]
    val = order[n_test:n_test + n_val]
    train = order[n_test + n_val:]
    splits = {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }

    # Disjointness (subject-wise leakage guard).
    seen: set[str] = set()
    for subs in splits.values():
        for s in subs:
            assert s not in seen, f"subject {s} appears in more than one split"
            seen.add(s)
    assert seen == set(subjects)

    # Every split must cover all five stages.
    for name, subs in splits.items():
        covered: set[int] = set()
        for s in subs:
            covered |= subject_stages[s]
        missing = FULL_STAGES - covered
        if missing:
            raise ValueError(
                f"split '{name}' is missing stage(s) {sorted(missing)}; "
                f"try a different --seed or more subjects"
            )

    return splits


def _read_subject_stages(epochs_path: Path) -> dict[str, set[int]]:
    """Read subject -> set of stages present, from the partitioned epoch store."""
    import pandas as pd

    df = pd.read_parquet(epochs_path, columns=["subject_id", "stage"])
    return {
        str(subj): set(int(s) for s in grp["stage"].unique())
        for subj, grp in df.groupby("subject_id", observed=True)
    }


def make_split(
    cfg: Config,
    seed: Optional[int] = None,
    val_frac: Optional[float] = None,
    test_frac: Optional[float] = None,
) -> Path:
    """Compute the subject-wise split and write splits.json.

    CLI overrides fall back to the values in config. Requires the epoch store
    from `etl` to exist.
    """
    seed = cfg.split.seed if seed is None else seed
    val_frac = cfg.split.val_frac if val_frac is None else val_frac
    test_frac = cfg.split.test_frac if test_frac is None else test_frac

    epochs_path = cfg.paths.epochs_path
    if not epochs_path.exists():
        raise FileNotFoundError(
            f"Epoch store not found: {epochs_path}. Run `etl` first."
        )

    subject_stages = _read_subject_stages(epochs_path)
    splits = assign_splits(subject_stages, seed, val_frac, test_frac)

    payload = {
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
    }
    splits_path = cfg.paths.splits_path
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Subject-wise split (seed={seed}) -> {splits_path}")
    for name in ("train", "val", "test"):
        subs = splits[name]
        print(f"  {name:>5}: {len(subs):3d} subjects  {subs}")
    return splits_path
