import json

import pytest

pytest.importorskip("xgboost")

from train_tune_all_finished_winner_ranker import crossfit_fold_ids, tree_counts


def test_crossfit_assigns_every_whole_race_once():
    race_ids = list(range(1, 12))

    folds = crossfit_fold_ids(race_ids, folds=4)

    flattened = [race_id for fold in folds for race_id in fold]
    assert sorted(flattened) == race_ids
    assert len(flattened) == len(set(flattened))
    assert max(map(len, folds)) - min(map(len, folds)) <= 1


def test_crossfit_requires_at_least_two_folds():
    with pytest.raises(ValueError, match="at least two"):
        crossfit_fold_ids([1, 2], folds=1)


def test_tree_counts_reuses_validated_bundle_counts(tmp_path):
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "best_tree_counts": {"form": [10, 20, 30]},
    }))

    counts = tree_counts(bundle, "form", ensemble_size=5, fallback=99)

    assert counts == [10, 20, 30, 10, 20]


def test_tree_counts_has_deterministic_fallback(tmp_path):
    counts = tree_counts(
        tmp_path / "missing.json", "market_aware", ensemble_size=3, fallback=44
    )

    assert counts == [44, 44, 44]
