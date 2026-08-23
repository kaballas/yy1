import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1]))

from src.winner_ranker import current_market_free_model_labels


def test_current_market_free_labels_exclude_target_race_market_inputs():
    labels = current_market_free_model_labels({
        "form": ["draw_number", "recent_days_since_last_run"],
        "historical": ["historical_market_overperformance_weighted_3"],
        "price": ["open_price", "draw_number"],
        "movement": ["market_open_to_fluc2_move_pct"],
        "rank": ["jockey_market_rank_abs_gap"],
        "engineered": ["current_market_log_price"],
    })

    assert labels == ["form", "historical"]
