"""Subject-wise split tests (split.py) — the leakage guard is the whole point."""

from __future__ import annotations

import pytest

from eeg_sleep_stager.split import FULL_STAGES, assign_splits


def _subjects_all_stages(n: int) -> dict[str, set[int]]:
    # Every subject has all five stages (realistic for Sleep-EDF).
    return {f"SC{i:02d}": set(FULL_STAGES) for i in range(n)}


def test_splits_are_subject_disjoint_and_cover_everyone():
    subject_stages = _subjects_all_stages(20)
    splits = assign_splits(subject_stages, seed=42, val_frac=0.15, test_frac=0.15)
    train, val, test = splits["train"], splits["val"], splits["test"]
    # Disjoint.
    assert set(train) & set(val) == set()
    assert set(train) & set(test) == set()
    assert set(val) & set(test) == set()
    # Complete partition.
    assert set(train) | set(val) | set(test) == set(subject_stages)


def test_split_is_deterministic_under_seed():
    subject_stages = _subjects_all_stages(20)
    a = assign_splits(subject_stages, seed=42, val_frac=0.15, test_frac=0.15)
    b = assign_splits(subject_stages, seed=42, val_frac=0.15, test_frac=0.15)
    assert a == b


def test_different_seed_changes_assignment():
    subject_stages = _subjects_all_stages(40)
    a = assign_splits(subject_stages, seed=1, val_frac=0.15, test_frac=0.15)
    b = assign_splits(subject_stages, seed=2, val_frac=0.15, test_frac=0.15)
    # Overwhelmingly likely to differ for 40 subjects.
    assert a != b


def test_split_sizes_match_fractions():
    subject_stages = _subjects_all_stages(20)
    splits = assign_splits(subject_stages, seed=7, val_frac=0.15, test_frac=0.15)
    assert len(splits["val"]) == 3    # round(20 * 0.15)
    assert len(splits["test"]) == 3
    assert len(splits["train"]) == 14


def test_every_split_covers_all_stages():
    subject_stages = _subjects_all_stages(15)
    splits = assign_splits(subject_stages, seed=0, val_frac=0.2, test_frac=0.2)
    for name, subs in splits.items():
        covered = set().union(*(subject_stages[s] for s in subs))
        assert covered == set(FULL_STAGES), name


def test_missing_stage_in_a_split_raises():
    # One subject is the only source of REM (stage 4); if it lands outside a
    # split, that split cannot cover all stages.
    subject_stages = {f"SC{i:02d}": {0, 1, 2, 3} for i in range(9)}
    subject_stages["SC00"] = {0, 1, 2, 3, 4}
    with pytest.raises(ValueError, match="missing stage"):
        assign_splits(subject_stages, seed=0, val_frac=0.2, test_frac=0.2)


def test_too_few_subjects_raises():
    with pytest.raises(ValueError, match="at least 3"):
        assign_splits(_subjects_all_stages(2), seed=0, val_frac=0.2, test_frac=0.2)


def test_train_always_nonempty_with_large_fractions():
    subject_stages = _subjects_all_stages(4)
    splits = assign_splits(subject_stages, seed=0, val_frac=0.4, test_frac=0.4)
    assert len(splits["train"]) >= 1
    total = sum(len(v) for v in splits.values())
    assert total == 4
