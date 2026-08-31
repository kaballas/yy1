from backtest_winner_ranker_checkpoints import cohort_key, leaderboard


def test_cohort_key_changes_when_race_order_changes():
    assert cohort_key([1, 2, 3]) == cohort_key([1, 2, 3])
    assert cohort_key([1, 2, 3]) != cohort_key([3, 2, 1])


def test_leaderboard_ranks_models_by_logloss_then_mrr_and_top1():
    records = [
        {
            "path": "a.pt", "model": "a", "checkpoint_type": "race_winner_moe",
            "cohort_id": "shared", "races": 2,
            "metrics": {
                "top1_hit_rate": 0.60, "top2_containment": 0.8,
                "top3_containment": 0.9, "mrr": 0.70,
                "race_logloss": 1.20, "average_winner_probability": 0.3,
            },
        },
        {
            "path": "b.pt", "model": "b", "checkpoint_type": "race_winner_moe",
            "cohort_id": "shared", "races": 2,
            "metrics": {
                "top1_hit_rate": 0.50, "top2_containment": 0.7,
                "top3_containment": 0.8, "mrr": 0.60,
                "race_logloss": 1.10, "average_winner_probability": 0.4,
            },
        },
    ]

    result = leaderboard(records)

    assert result["model"].tolist() == ["b", "a"]
    assert result["cohort_rank"].tolist() == [1, 2]
