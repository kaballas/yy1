import pandas as pd

from analyze_sealed_market_disagreements import segment_metrics


def test_segment_metrics_counts_corrections_and_damage_per_challenger():
    races = pd.DataFrame({
        "favourite_price_segment": ["2.50-3.99", "2.50-3.99", "4.00-5.99"],
        "field_size_segment": ["8-10"] * 3,
        "distance_segment": ["<=1200m"] * 3,
        "class_segment": ["<=50"] * 3,
        "corrector_confidence_segment": ["validation_Q1"] * 3,
        "corrector_market_rank_gap_segment": ["1"] * 3,
        "gpt_pick_market_agreement": ["disagree"] * 3,
        "gpt_fluc2_market_agreement": ["disagree"] * 3,
        "market_win": [0, 1, 0],
        **{
            f"{name}_{suffix}": values
            for name in ("frozen_blend", "market_corrector", "gpt_pick", "gpt_fluc2")
            for suffix, values in {
                "win": [1, 0, 0], "changed": [1, 1, 0],
                "corrected": [1, 0, 0], "damaged": [0, 1, 0],
            }.items()
        },
    })

    result = segment_metrics(races, "sealed_test")
    row = result.loc[
        (result["dimension"] == "favourite_price")
        & (result["segment"] == "2.50-3.99")
        & (result["challenger"] == "frozen_blend")
    ].iloc[0]

    assert row["races"] == 2
    assert row["market_wins"] == 1
    assert row["model_wins"] == 1
    assert row["pick_changes"] == 2
    assert row["market_losses_corrected"] == 1
    assert row["market_winners_damaged"] == 1
    assert row["net_winners_gained"] == 0
