import pandas as pd
import pytest

from validate_chronological_winner_blend import validate_chronology


def cohort(race_id: int, start: str) -> pd.DataFrame:
    return pd.DataFrame({"race_id": [race_id], "start_time_iso": [start]})


def test_validate_chronology_requires_disjoint_strictly_ordered_cohorts():
    train = cohort(1, "2024-01-01T00:00:00Z")
    validation = cohort(2, "2024-02-01T00:00:00Z")
    test = cohort(3, "2024-03-01T00:00:00Z")

    validate_chronology(train, validation, test)

    with pytest.raises(ValueError, match="strictly ordered"):
        validate_chronology(train, test, validation)
    with pytest.raises(ValueError, match="overlap"):
        validate_chronology(train, validation, validation)
