import pandas as pd

from predict_moe_winner_ranker_feature_map import prediction_view


def test_default_prediction_view_only_shows_rank_number_and_name():
    result = pd.DataFrame({
        "runner_number": [4],
        "runner_name": ["Nankeen"],
        "ranking_logit": [0.42],
        "winner_probability": [0.14],
        "rank": [1],
        "expert_0_gate": [0.5],
    })

    view = prediction_view(result, diagnostics=False)

    assert view.columns.tolist() == ["rank", "runner_number", "runner_name"]
    assert view.iloc[0].tolist() == [1, 4, "Nankeen"]


def test_diagnostic_prediction_view_keeps_all_columns():
    result = pd.DataFrame({"rank": [1], "expert_0_gate": [0.5]})

    assert prediction_view(result, diagnostics=True) is result
