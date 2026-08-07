"""Repository-relative filesystem default contracts."""

from src.config import DEFAULT_DB, ROOT


def test_default_database_is_derived_from_repository_root():
    assert DEFAULT_DB == ROOT / "db/race_runners.sqlite"
    assert DEFAULT_DB.is_absolute()
