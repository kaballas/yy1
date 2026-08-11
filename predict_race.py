#!/usr/bin/env python3
"""Predict every runner in one race with a trained TabFM model.

The script expects a tabular file containing:
  * historical labelled rows used as in-context examples
  * the target race rows identified by --race-id

Use the same feature columns, ordering, preprocessing, and label encoding that
were used during training. A saved sklearn-compatible transformer can be passed
with --preprocessor.

Example:
    python predict_race.py \
      --checkpoint checkpoints/best.pt \
      --data data/model_rows.parquet \
      --race-id 123456 \
      --race-id-column race_id \
      --runner-id-column runner_id \
      --label-column target \
      --feature-columns-file artifacts/feature_columns.json \
      --preprocessor artifacts/preprocessor.joblib \
      --date-column race_date \
      --positive-class 1 \
      --output predictions/race_123456.csv
"""

from __future__ import annotations

import argparse
import bisect
import copy
import inspect
import json
import sqlite3
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from src.config import DEFAULT_DB
from src.constants import TRAINING_ROWS_VIEW
from src.database import quote_identifier, require_training_rows_view
from src.preprocessing import transform as transform_training_features
from src.metrics import probability_metrics, race_top3_metrics
from src.prediction import market_rank_scores

try:
    from src.model.model import TabFM
except ImportError:
    try:
        from model import TabFM  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Cannot import TabFM. Run this script from the repository root, "
            "or add the repository root to PYTHONPATH."
        ) from exc


MODEL_CONFIG_KEYS = set(inspect.signature(TabFM.__init__).parameters) - {"self"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict all runners for one race using a trained TabFM checkpoint."
    )
    parser.add_argument(
        "--checkpoint", type=Path, action="append",
        help="Checkpoint to use. Repeat to select particular files instead of all models.",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("outputs"),
        help="Directory scanned for *.pt checkpoints when --checkpoint is omitted (default: outputs).",
    )
    parser.add_argument("--data", type=Path,
                        help="CSV or Parquet for generic mode. Without this, use --db and the training race context.")
    parser.add_argument("--db", type=Path,
                        default=DEFAULT_DB,
                        help="SQLite database used by train_model.py native mode.")
    parser.add_argument(
        "--race-id",
        help=(
            "Race ID to predict, or the only race to evaluate with --backtest. "
            "Compared as text to avoid integer/string mismatches."
        ),
    )
    parser.add_argument("--race-id-column", default="race_id")
    parser.add_argument("--runner-id-column", default="runner_number")
    parser.add_argument("--label-column", default="top3_mask")

    features = parser.add_mutually_exclusive_group(required=False)
    features.add_argument(
        "--feature-columns",
        help="Comma-separated feature names in exact training order.",
    )
    features.add_argument(
        "--feature-columns-file",
        type=Path,
        help="JSON list or newline-delimited feature names in exact training order.",
    )

    parser.add_argument(
        "--preprocessor",
        type=Path,
        help="Optional joblib/pickle object exposing transform(DataFrame).",
    )
    parser.add_argument(
        "--categorical-indices",
        default="",
        help="Comma-separated categorical feature indices after preprocessing.",
    )
    parser.add_argument(
        "--date-column",
        help="When set, context rows must be strictly earlier than the target race.",
    )
    parser.add_argument(
        "--max-context-rows",
        type=int,
        default=0,
        help="Maximum labelled historical rows supplied as context. Use 0 for all rows.",
    )
    parser.add_argument(
        "--positive-class",
        type=int,
        default=1,
        help="Class probability used for ranking a classification result.",
    )
    parser.add_argument(
        "--class-names",
        help="Optional comma-separated class labels in model index order.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--backtest", action="store_true",
        help="Backtest every status='finished' race separately for every checkpoint.",
    )
    parser.add_argument(
        "--backtest-max-races", type=int, default=0,
        help=(
            "Use the most recent N matching finished races for --backtest, then score "
            "them chronologically; 0 means every matching finished race."
        ),
    )
    parser.add_argument(
        "--competition-id", type=int,
        help="Limit --backtest target races to this competition_id.",
    )
    parser.add_argument(
        "--include-competition-history-context",
        action="store_true",
        help=(
            "In native single-race prediction, augment the checkpoint context "
            "with every strictly earlier finished, completely labelled race "
            "having the target race's competition_id."
        ),
    )
    parser.add_argument(
        "--strict-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require checkpoint state_dict keys to match the model exactly.",
    )
    parser.add_argument(
        "--show-warnings",
        action="store_true",
        help="Show Python library warnings (suppressed by default for batch prediction output).",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data format {suffix!r}; use CSV or Parquet.")


def load_native_query(
    db_path: Path,
    race_id: str,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Load one target race directly from the training SQLite database."""
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    selected = list(dict.fromkeys([
        "race_id", "start_time_iso", "competition_id", "race_number",
        "runner_number", *feature_columns, "top3_mask", "is_winner",
    ]))
    sql = (
        f"SELECT {', '.join(quote_identifier(column) for column in selected)} "
        f"FROM race_runners WHERE race_id = ? ORDER BY start_time_iso, race_id, runner_number"
    )
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(sql, (int(race_id),)).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError(f"Race ID {race_id!r} was not found in race_runners.")
    return pd.DataFrame(rows, columns=selected)


def load_competition_context_race_ids(
    db_path: Path, target_race_id: str
) -> tuple[list[int], int, int]:
    """Select all complete earlier races from the target's competition."""
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        target_rows = connection.execute(
            "SELECT competition_id, race_number, MIN(start_time_iso) "
            "FROM race_runners WHERE race_id = ? "
            "GROUP BY competition_id, race_number",
            (int(target_race_id),),
        ).fetchall()
        if not target_rows:
            raise ValueError(
                f"Race ID {target_race_id!r} was not found in race_runners."
            )
        if len(target_rows) != 1:
            raise ValueError(
                f"Target race {target_race_id} has inconsistent competition_id/race_number."
            )
        competition_id, target_race_number, target_time = target_rows[0]
        if competition_id is None or target_race_number is None or target_time is None:
            raise ValueError(
                f"Target race {target_race_id} has NULL competition_id, race_number, "
                "or start_time_iso."
            )
        context_rows = connection.execute(
            "SELECT race_id "
            "FROM race_runners "
            "WHERE status = 'finished' "
            "AND competition_id = ? "
            "AND race_id <> ? "
            "AND start_time_iso < ? "
            "AND top3_mask IN (0, 1) "
            "GROUP BY race_id "
            "HAVING COUNT(*) >= 4 "
            "AND SUM(CASE WHEN top3_mask = 1 THEN 1 ELSE 0 END) = 3 "
            "ORDER BY MIN(start_time_iso), race_id",
            (int(competition_id), int(target_race_id), str(target_time)),
        ).fetchall()
    finally:
        connection.close()

    context_race_ids = [int(row[0]) for row in context_rows]
    if not context_race_ids:
        raise ValueError(
            f"No finished historical races found for competition_id={int(competition_id)} "
            f"strictly before target race {int(target_race_id)}."
        )
    tqdm.write(
        "competition_context_selection "
        f"target_race_id={int(target_race_id)} "
        f"competition_id={int(competition_id)} "
        f"target_race_number={int(target_race_number)} "
        f"eligible_previous_races={len(context_race_ids)} "
        f"race_ids={context_race_ids}",
    )
    return context_race_ids, int(competition_id), int(target_race_number)


def load_context_race_metadata(
    db_path: Path, context_race_ids: Sequence[int]
) -> dict[int, tuple[str, int | None, int | None]]:
    """Load the race-level status and competition fields for display/audit."""
    if not context_race_ids:
        return {}
    placeholders = ", ".join("?" for _ in context_race_ids)
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            f"SELECT race_id, status, competition_id, race_number "
            f"FROM race_runners WHERE race_id IN ({placeholders}) "
            "GROUP BY race_id, status, competition_id, race_number",
            list(map(int, context_race_ids)),
        ).fetchall()
    finally:
        connection.close()
    metadata: dict[int, tuple[str, int | None, int | None]] = {}
    for race_id, status, competition_id, race_number in rows:
        metadata[int(race_id)] = (
            str(status),
            None if competition_id is None else int(competition_id),
            None if race_number is None else int(race_number),
        )
    return metadata


def checkpoint_context_size(metadata: Mapping[str, Any]) -> int:
    """Return the per-query context window saved by train_model.py."""
    value = metadata.get("validation_context_races_per_prediction", metadata.get("context_races_per_step"))
    if value is None:
        raise ValueError("Checkpoint has no saved context-race window.")
    size = int(value)
    if size < 1:
        raise ValueError(f"Checkpoint context-race window must be positive, got {size}.")
    return size


def load_training_context_for_target(
    db_path: Path, target_race_id: str, feature_columns: Sequence[str], metadata: Mapping[str, Any],
    prepared_context: tuple[dict[int, pd.DataFrame], list[tuple[pd.Timestamp, int]]] | None = None,
    prepared_query: pd.DataFrame | None = None,
    include_competition_history: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Use the causal same-competition context policy used for validation."""
    query = prepared_query.copy() if prepared_query is not None else load_native_query(db_path, target_race_id, feature_columns)
    target_time = pd.to_datetime(query["start_time_iso"], utc=True, errors="raise").min()
    target_competitions = pd.to_numeric(
        query["competition_id"], errors="raise"
    ).dropna().unique()
    if len(target_competitions) != 1:
        raise ValueError(
            f"Target race {target_race_id} has missing or inconsistent competition_id values."
        )
    target_competition_id = int(target_competitions[0])
    context_size = checkpoint_context_size(metadata)
    if prepared_context is not None:
        context_by_race, ordered_context = prepared_context
        cutoff = bisect.bisect_left(ordered_context, (target_time, -1))
        eligible_ids = []
        for _, race_id in ordered_context[:cutoff]:
            race_competitions = pd.to_numeric(
                context_by_race[race_id]["competition_id"], errors="raise"
            ).dropna().unique()
            if len(race_competitions) != 1:
                raise ValueError(
                    f"Context race {race_id} has missing or inconsistent competition_id values."
                )
            if int(race_competitions[0]) == target_competition_id:
                eligible_ids.append(race_id)
        selected_ids = eligible_ids[-context_size:]
        if len(selected_ids) < context_size:
            raise ValueError(
                f"Target race {target_race_id} has only {len(selected_ids)} eligible earlier "
                f"same-competition completed context races; "
                f"checkpoint requires {context_size}."
            )
        context = pd.concat(
            [context_by_race[race_id] for race_id in selected_ids], ignore_index=True
        )
        return context, query, selected_ids
    selected = list(dict.fromkeys([
        "race_id", "start_time_iso", "competition_id", "runner_number",
        "race_number", *feature_columns, "top3_mask",
    ]))
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        sql = (
            f"SELECT {', '.join(quote_identifier(column) for column in selected)} "
            "FROM race_runners "
            "WHERE status = 'finished' AND start_time_iso < ? AND race_id <> ? "
            "AND competition_id = ? AND top3_mask IN (0, 1) "
            "ORDER BY start_time_iso, race_id, runner_number"
        )
        frame = pd.read_sql_query(
            sql,
            connection,
            params=(target_time.isoformat(), int(target_race_id), target_competition_id),
        )
    finally:
        connection.close()
    minimum_race_number = metadata.get("optimizer_min_race_number")
    if minimum_race_number is not None:
        frame = frame.loc[frame["race_number"] >= int(minimum_race_number)].copy()
    summary = frame.groupby("race_id", sort=False).agg(
        start_time_iso=("start_time_iso", "min"), runners=("race_id", "size"), top3=("top3_mask", "sum")
    )
    summary = summary.loc[(summary["runners"] >= 4) & (summary["top3"] == 3)].sort_values("start_time_iso", kind="stable")
    selected_ids = [int(value) for value in summary.tail(context_size).index]
    if len(selected_ids) < context_size:
        raise ValueError(
            f"Target race {target_race_id} has only {len(selected_ids)} eligible earlier "
            f"same-competition completed context races; "
            f"checkpoint requires {context_size}."
        )
    context = frame.loc[frame["race_id"].isin(selected_ids)].copy()
    order = {race_id: index for index, race_id in enumerate(selected_ids)}
    context["__context_order"] = context["race_id"].map(order)
    context = context.sort_values(["__context_order", "runner_number"], kind="stable").drop(columns="__context_order")
    if include_competition_history:
        competition_ids, _, _ = load_competition_context_race_ids(
            db_path, target_race_id
        )
        added_ids = [race_id for race_id in competition_ids if race_id not in selected_ids]
        combined_ids = list(dict.fromkeys([*selected_ids, *competition_ids]))
        selected_columns = [
            "race_id", "start_time_iso", "runner_number", *feature_columns, "top3_mask"
        ]
        placeholders = ", ".join("?" for _ in combined_ids)
        connection = sqlite3.connect(
            f"file:{db_path.resolve()}?mode=ro", uri=True
        )
        try:
            context = pd.read_sql_query(
                f"SELECT {', '.join(quote_identifier(column) for column in selected_columns)} "
                f"FROM race_runners WHERE race_id IN ({placeholders}) "
                "ORDER BY start_time_iso, race_id, runner_number",
                connection,
                params=combined_ids,
            )
        finally:
            connection.close()
        selected_ids = [
            int(value)
            for value in context[["race_id", "start_time_iso"]]
            .drop_duplicates()
            .sort_values(["start_time_iso", "race_id"], kind="stable")["race_id"]
        ]
        tqdm.write(
            "competition_history_context_augmented "
            f"target_race_id={int(target_race_id)} "
            f"checkpoint_context_races={context_size} "
            f"competition_history_races={len(competition_ids)} "
            f"new_competition_races={len(added_ids)} "
            f"combined_context_races={len(selected_ids)} "
            f"race_ids={selected_ids}",
        )
    return context, query, selected_ids


def apply_checkpoint_preprocessing(
    values: np.ndarray,
    feature_columns: Sequence[str],
    metadata: Mapping[str, Any],
) -> np.ndarray:
    """Normalize every checkpoint feature without inference-time masking."""
    median = metadata.get("median")
    scale = metadata.get("scale")
    if median is None or scale is None:
        raise ValueError(
            "Checkpoint has no fitted median/scale preprocessing statistics; "
            "supply --preprocessor or use a native training checkpoint."
        )
    if torch.is_tensor(median):
        median = median.detach().cpu().numpy()
    if torch.is_tensor(scale):
        scale = scale.detach().cpu().numpy()
    median = np.asarray(median, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if median.shape != (len(feature_columns),) or scale.shape != (len(feature_columns),):
        raise ValueError(
            "Checkpoint preprocessing dimensions do not match its feature columns."
        )
    return transform_training_features(values, median, scale)


def torch_load_trusted(path: Path, device: torch.device) -> Any:
    """Load a trusted local checkpoint across PyTorch versions.

    weights_only=False is intentional because some training pipelines save a
    full nn.Module or configuration objects. Never use this on an untrusted file.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def first_mapping(checkpoint: Mapping[str, Any], names: Iterable[str]) -> Mapping[str, Any] | None:
    for name in names:
        value = checkpoint.get(name)
        if isinstance(value, Mapping):
            return value
    return None


def normalise_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    # Common training bundles place model arguments one level deeper.
    for nested_key in ("model", "model_config", "tabfm", "architecture"):
        nested = config.get(nested_key)
        if isinstance(nested, Mapping) and any(key in MODEL_CONFIG_KEYS for key in nested):
            config = nested
            break
    return {key: value for key, value in config.items() if key in MODEL_CONFIG_KEYS}


def load_model(checkpoint_path: Path, device: torch.device, strict: bool) -> tuple[TabFM, Mapping[str, Any]]:
    checkpoint = torch_load_trusted(checkpoint_path, device)

    if isinstance(checkpoint, TabFM):
        model = checkpoint
        metadata: Mapping[str, Any] = {}
    elif isinstance(checkpoint, nn.Module):
        if not isinstance(checkpoint, TabFM):
            raise TypeError(f"Checkpoint contains {type(checkpoint).__name__}, not TabFM.")
        model = checkpoint
        metadata = {}
    elif isinstance(checkpoint, Mapping):
        embedded_model = checkpoint.get("model")
        if isinstance(embedded_model, TabFM):
            model = embedded_model
            metadata = checkpoint
        else:
            state_dict = first_mapping(
                checkpoint,
                ("model_state_dict", "state_dict", "model_weights", "weights"),
            )
            if state_dict is None:
                # Some checkpoints are a bare state_dict.
                if checkpoint and all(isinstance(key, str) for key in checkpoint):
                    if all(torch.is_tensor(value) for value in checkpoint.values()):
                        state_dict = checkpoint
                if state_dict is None:
                    raise KeyError(
                        "Checkpoint has no TabFM model and no recognised state_dict key."
                    )

            raw_config = first_mapping(
                checkpoint,
                (
                    "model_kwargs",
                    "model_config",
                    "config",
                    "hyperparameters",
                    "hparams",
                    "args",
                ),
            )
            if raw_config is None:
                raise KeyError(
                    "Checkpoint contains only weights but no model configuration. "
                    "Save model_config beside the state_dict or load a full TabFM object."
                )
            model_config = normalise_model_config(raw_config)
            if not model_config:
                raise ValueError("No TabFM constructor values were found in checkpoint config.")
            model = TabFM(**model_config)
            incompatible = model.load_state_dict(state_dict, strict=strict)
            if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
                tqdm.write(
                    "WARNING: non-strict checkpoint load. "
                    f"Missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}",
                    file=sys.stderr,
                )
            metadata = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint).__name__}")

    model.to(device)
    model.eval()
    return model, metadata


def read_feature_columns(args: argparse.Namespace, metadata: Mapping[str, Any]) -> list[str]:
    if args.feature_columns:
        columns = [part.strip() for part in args.feature_columns.split(",") if part.strip()]
    elif args.feature_columns_file:
        text = args.feature_columns_file.read_text(encoding="utf-8").strip()
        if args.feature_columns_file.suffix.lower() == ".json":
            value = json.loads(text)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError("Feature-column JSON must be a list of strings.")
            columns = value
        else:
            columns = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        columns = []
        for key in ("feature_columns", "features", "input_columns"):
            value = metadata.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if all(isinstance(item, str) for item in value):
                    columns = list(value)
                    break
        if not columns:
            config = metadata.get("config")
            if isinstance(config, Mapping):
                for key in ("feature_columns", "features", "input_columns"):
                    value = config.get(key)
                    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                        if all(isinstance(item, str) for item in value):
                            columns = list(value)
                            break

    if not columns:
        raise ValueError(
            "Feature columns are unknown. Supply --feature-columns or "
            "--feature-columns-file in the exact order used for training."
        )
    if len(columns) != len(set(columns)):
        raise ValueError("Feature-column list contains duplicates.")
    return columns


def describe_feature_source(
    args: argparse.Namespace,
    metadata: Mapping[str, Any],
    feature_count: int,
) -> str:
    """Describe the real source of the ordered inference feature schema."""
    if args.feature_columns:
        return (
            f"source=command_line feature_count={feature_count} json_file=none "
            "zero_feature_masking=disabled"
        )
    if args.feature_columns_file:
        source = args.feature_columns_file.resolve()
        source_kind = "json_file" if source.suffix.lower() == ".json" else "text_file"
        return (
            f"source={source_kind} feature_count={feature_count} file={source} "
            "zero_feature_masking=disabled"
        )
    recorded_manifest = metadata.get("feature_manifest")
    if recorded_manifest:
        return (
            f"source=checkpoint_embedded feature_count={feature_count} "
            f"recorded_training_json={recorded_manifest} json_loaded_at_prediction=false "
            "zero_feature_masking=disabled"
        )
    return (
        f"source=checkpoint_embedded feature_count={feature_count} "
        "recorded_training_json=none json_loaded_at_prediction=false "
        "zero_feature_masking=disabled"
    )


def load_preprocessor(path: Path | None) -> Any | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import joblib
        return joblib.load(path)
    except ImportError as exc:
        raise RuntimeError("Install joblib to load --preprocessor.") from exc


def matrix_from_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    preprocessor: Any | None,
) -> np.ndarray:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Data is missing feature columns: {missing}")

    selected = frame.loc[:, feature_columns]
    if preprocessor is not None:
        values = preprocessor.transform(selected)
        if hasattr(values, "toarray"):
            values = values.toarray()
        values = np.asarray(values)
    else:
        converted = selected.apply(pd.to_numeric, errors="coerce")
        non_numeric = [
            column for column in feature_columns
            if (selected[column].notna() & converted[column].isna()).any()
        ]
        if non_numeric:
            raise ValueError(
                "Non-numeric feature values require the exact saved training preprocessor. "
                f"Affected columns: {non_numeric}"
            )
        values = converted.to_numpy(dtype=np.float32)

    if values.ndim != 2:
        raise ValueError(f"Preprocessor must return a 2-D matrix, got shape {values.shape}.")
    try:
        return values.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Preprocessed features must be numeric.") from exc


def parse_categorical_indices(text: str, feature_count: int) -> list[int]:
    if not text.strip():
        return []
    indices = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    invalid = [index for index in indices if index < 0 or index >= feature_count]
    if invalid:
        raise ValueError(
            f"Categorical indices {invalid} are outside [0, {feature_count - 1}]."
        )
    return indices


def checkpoint_cat_mask(metadata: Mapping[str, Any], feature_count: int) -> np.ndarray | None:
    for key in ("cat_mask", "categorical_mask"):
        value = metadata.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=bool)
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        if array.shape == (feature_count,):
            return array
    return None


def select_rows(
    data: pd.DataFrame,
    race_id: str,
    race_id_column: str,
    label_column: str,
    date_column: str | None,
    max_context_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [race_id_column, label_column]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(f"Data is missing required columns: {missing}")

    race_text = data[race_id_column].astype(str)
    query = data.loc[race_text == str(race_id)].copy()
    if query.empty:
        raise ValueError(f"Race ID {race_id!r} was not found in {race_id_column!r}.")

    context = data.loc[(race_text != str(race_id)) & data[label_column].notna()].copy()

    if date_column:
        if date_column not in data.columns:
            raise KeyError(f"Date column {date_column!r} is missing.")
        query_dates = pd.to_datetime(query[date_column], errors="raise", utc=True)
        context_dates = pd.to_datetime(context[date_column], errors="raise", utc=True)
        target_start = query_dates.min()
        context = context.loc[context_dates < target_start].copy()
        context["__prediction_sort_date"] = context_dates.loc[context.index]
        context = context.sort_values("__prediction_sort_date")

    if context.empty:
        raise ValueError("No labelled historical rows are available before the target race.")

    if max_context_rows < 0:
        raise ValueError("--max-context-rows must be zero or positive.")
    if max_context_rows and len(context) > max_context_rows:
        context = context.tail(max_context_rows).copy()

    return context, query


def class_names_from_args(
    args: argparse.Namespace,
    metadata: Mapping[str, Any],
    max_classes: int,
) -> list[str] | None:
    if args.class_names:
        names = [part.strip() for part in args.class_names.split(",")]
    else:
        names = []
        for key in ("class_names", "classes", "label_classes"):
            value = metadata.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                names = [str(item) for item in value]
                break
    if names and len(names) != max_classes:
        raise ValueError(
            f"Expected {max_classes} class names, received {len(names)}."
        )
    return names or None


def encode_context_labels(
    series: pd.Series,
    max_classes: int,
    class_names: Sequence[str] | None,
) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_is_complete = numeric.notna().all()
    if numeric_is_complete:
        numeric_values = numeric.to_numpy()
        if not np.equal(numeric_values, np.round(numeric_values)).all():
            raise ValueError("Classification labels must be integer class IDs.")
        labels = numeric_values.astype(np.int64)
    elif class_names:
        lookup = {name: index for index, name in enumerate(class_names)}
        encoded = series.astype(str).map(lookup)
        if encoded.isna().any():
            unknown = sorted(series.loc[encoded.isna()].astype(str).unique().tolist())
            raise ValueError(f"Context contains labels absent from --class-names: {unknown}")
        labels = encoded.to_numpy(dtype=np.int64)
    else:
        raise ValueError(
            "Classification labels are not numeric class IDs. Supply --class-names "
            "in model index order to encode them."
        )

    if labels.size and (labels.min() < 0 or labels.max() >= max_classes):
        raise ValueError(
            f"Classification labels must be in [0, {max_classes - 1}], "
            f"got [{labels.min()}, {labels.max()}]."
        )
    return labels


def make_race_group_ids(
    model: TabFM,
    context: pd.DataFrame,
    query: pd.DataFrame,
    race_id_column: str,
    device: torch.device,
) -> torch.Tensor | None:
    pre_icl_enabled = getattr(model, "pre_icl_race_encoder", None) is not None
    post_icl_enabled = getattr(model, "race_context_mode", "none") == "self_attention"
    if not pre_icl_enabled and not post_icl_enabled:
        return None

    if pre_icl_enabled:
        # Every historical race gets a separate non-negative group. The target
        # race receives a new group that cannot overlap with context groups.
        context_codes, uniques = pd.factorize(
            context[race_id_column].astype(str), sort=False
        )
        if np.any(context_codes < 0):
            raise ValueError("Historical race IDs cannot be missing in pre-ICL race mode.")
        query_group = len(uniques)
        combined = np.concatenate(
            [context_codes.astype(np.int64), np.full(len(query), query_group, dtype=np.int64)]
        )
    else:
        # Post-ICL race conditioning applies only to target/query rows.
        combined = np.concatenate(
            [np.full(len(context), -1, dtype=np.int64), np.zeros(len(query), dtype=np.int64)]
        )

    return torch.from_numpy(combined).unsqueeze(0).to(device=device)


def predict_one(
    args: argparse.Namespace,
    checkpoint: Path,
    model: TabFM | None = None,
    metadata: Mapping[str, Any] | None = None,
    prepared_context: tuple[dict[int, pd.DataFrame], list[tuple[pd.Timestamp, int]]] | None = None,
    prepared_query: pd.DataFrame | None = None,
) -> pd.DataFrame:
    device = torch.device(args.device)
    if model is None or metadata is None:
        model, metadata = load_model(checkpoint, device, args.strict_load)
    feature_columns = read_feature_columns(args, metadata)

    native_mode = args.data is None
    if native_mode:
        if args.preprocessor is not None:
            raise ValueError("--preprocessor is only supported with --data generic mode.")
        context, query, ordered_context_race_ids = load_training_context_for_target(
            args.db,
            str(args.race_id),
            feature_columns,
            metadata,
            prepared_context,
            prepared_query,
            include_competition_history=getattr(
                args, "include_competition_history_context", False
            ),
        )
        if not args.backtest:
            context_strategy = (
                "most_recent_earlier_same_competition_training_races_plus_all_earlier_same_competition"
                if getattr(args, "include_competition_history_context", False)
                else "most_recent_earlier_same_competition_training_races"
            )
            tqdm.write(
                "PREDICTION CONTEXT "
                f"checkpoint={checkpoint.name} strategy={context_strategy} "
                f"races={len(ordered_context_race_ids)} rows={len(context)} "
                f"race_ids={ordered_context_race_ids}",
            )
            if len(ordered_context_race_ids) > checkpoint_context_size(metadata):
                tqdm.write(
                    "WARNING context_window_out_of_distribution "
                    f"checkpoint={checkpoint.name} "
                    f"trained_context_races={checkpoint_context_size(metadata)} "
                    f"prediction_context_races={len(ordered_context_race_ids)} "
                    "reason=include_competition_history_context",
                    file=sys.stderr,
                )
        context_values = matrix_from_frame(context, feature_columns, None)
    else:
        data = load_table(args.data)
        context, query = select_rows(
            data=data,
            race_id=str(args.race_id),
            race_id_column=args.race_id_column,
            label_column=args.label_column,
            date_column=args.date_column,
            max_context_rows=args.max_context_rows,
        )
        context_values = None

    if args.runner_id_column not in query.columns:
        raise KeyError(f"Runner ID column {args.runner_id_column!r} is missing.")

    preprocessor = load_preprocessor(args.preprocessor)
    if native_mode:
        query_values = matrix_from_frame(query, feature_columns, None)
        context_values = apply_checkpoint_preprocessing(
            context_values, feature_columns, metadata
        )
        query_values = apply_checkpoint_preprocessing(
            query_values, feature_columns, metadata
        )
    else:
        context_values = matrix_from_frame(context, feature_columns, preprocessor)
        query_values = matrix_from_frame(query, feature_columns, preprocessor)
    if context_values.shape[1] != query_values.shape[1]:
        raise ValueError("Context and query preprocessing produced different feature counts.")

    feature_count = context_values.shape[1]
    categorical_indices = parse_categorical_indices(args.categorical_indices, feature_count)
    cat_array = checkpoint_cat_mask(metadata, feature_count)
    if categorical_indices:
        cat_array = np.zeros(feature_count, dtype=bool)
        cat_array[categorical_indices] = True

    x_values = np.concatenate([context_values, query_values], axis=0)
    x = torch.from_numpy(x_values).unsqueeze(0).to(device=device)
    train_size = torch.tensor([len(context)], dtype=torch.long, device=device)
    d = torch.tensor([feature_count], dtype=torch.long, device=device)
    cat_mask = None
    if cat_array is not None:
        cat_mask = torch.from_numpy(cat_array).unsqueeze(0).to(device=device, dtype=torch.bool)

    class_names: list[str] | None = None
    if model.is_classifier:
        class_names = class_names_from_args(args, metadata, model.max_classes)
        context_labels = encode_context_labels(
            context[args.label_column], model.max_classes, class_names
        ).astype(np.float32)
    else:
        context_labels = pd.to_numeric(
            context[args.label_column], errors="raise"
        ).to_numpy(dtype=np.float32)
        if not np.isfinite(context_labels).all():
            raise ValueError("Regression context labels must be finite.")

    y_values = np.concatenate(
        [context_labels, np.full(len(query), -100.0, dtype=np.float32)]
    )
    y = torch.from_numpy(y_values).unsqueeze(0).to(device=device)
    race_group_ids = make_race_group_ids(
        model, context, query, args.race_id_column, device
    )
    valid_row_mask = torch.ones(
        (1, len(context) + len(query)), dtype=torch.bool, device=device
    )

    with torch.inference_mode():
        logits = model(
            x,
            y,
            train_size,
            cat_mask=cat_mask,
            d=d,
            race_group_ids=race_group_ids,
            valid_row_mask=valid_row_mask,
        )
        query_logits = logits[0, len(context): len(context) + len(query)]

    result = query.loc[:, [args.race_id_column, args.runner_id_column]].reset_index(drop=True)

    # Keep the market prices alongside the model ranking so live predictions
    # can be compared directly with the available market.  These columns are
    # optional in generic input files, but are present in the native SQLite
    # race-runner data.
    for market_column in ("open_price", "fluc1", "fluc2"):
        if market_column in query.columns:
            result[market_column] = query[market_column].reset_index(drop=True)
        else:
            result[market_column] = "-"

    for outcome_column in ("is_winner", "top3_mask"):
        if outcome_column in query.columns:
            result[outcome_column] = query[outcome_column].map(
                lambda value: "-" if pd.isna(value) else int(value)
            ).to_numpy()
        else:
            # Keep the output schema stable for live/generic inputs without
            # post-race outcome columns.
            result[outcome_column] = "-"

    if model.is_classifier:
        probabilities = torch.softmax(query_logits.float(), dim=-1).cpu().numpy()
        if args.positive_class < 0 or args.positive_class >= probabilities.shape[1]:
            raise ValueError(
                f"--positive-class must be in [0, {probabilities.shape[1] - 1}]."
            )
        predicted_indices = probabilities.argmax(axis=1)
        for index in range(probabilities.shape[1]):
            label = class_names[index] if class_names else str(index)
            result[f"probability_{label}"] = probabilities[:, index]
        result["predicted_class"] = predicted_indices
        if class_names:
            result["predicted_label"] = [class_names[index] for index in predicted_indices]
        result["ranking_probability"] = probabilities[:, args.positive_class]
        result = result.sort_values(
            ["ranking_probability", args.runner_id_column],
            ascending=[False, True],
        ).reset_index(drop=True)
        result.insert(0, "predicted_rank", np.arange(1, len(result) + 1))
    else:
        predictions = query_logits.squeeze(-1).float().cpu().numpy()
        result["prediction"] = predictions
        result = result.sort_values(
            ["prediction", args.runner_id_column], ascending=[False, True]
        ).reset_index(drop=True)
        result.insert(0, "predicted_rank", np.arange(1, len(result) + 1))

    return result


def checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    """Return explicitly selected checkpoints or every .pt file in models-dir."""
    paths = args.checkpoint or sorted(args.models_dir.glob("*.pt"))
    paths = [path.resolve() for path in paths if path.is_file()]
    if not paths:
        raise ValueError(
            f"No .pt checkpoints found in {args.models_dir} (or supplied with --checkpoint)."
        )
    return paths


def checkpoint_column_name(path: Path, used: set[str]) -> str:
    stem = "".join(char if char.isalnum() else "_" for char in path.stem).strip("_") or "model"
    column = f"model_{stem}_score"
    suffix = 2
    while column in used:
        column = f"model_{stem}_{suffix}_score"
        suffix += 1
    used.add(column)
    return column


def predict(args: argparse.Namespace) -> pd.DataFrame:
    """Score the race with every requested checkpoint and average their scores."""
    combined: pd.DataFrame | None = None
    score_columns: list[str] = []
    used_columns: set[str] = set()
    skipped: list[tuple[Path, str]] = []
    key_columns = [args.race_id_column, args.runner_id_column]
    paths = checkpoint_paths(args)
    checkpoint_progress = tqdm(
        paths,
        desc="Predicting checkpoints",
        unit="model",
        dynamic_ncols=True,
    )
    for path in checkpoint_progress:
        checkpoint_progress.set_description(f"Predicting {path.name}")
        try:
            model, metadata = load_model(path, torch.device(args.device), args.strict_load)
            feature_columns = read_feature_columns(args, metadata)
            tqdm.write(
                f"FEATURES checkpoint={path.name} "
                + describe_feature_source(args, metadata, len(feature_columns))
            )
            result = predict_one(args, path, model, metadata)
        except Exception as exc:
            skipped.append((path, str(exc)))
            tqdm.write(
                f"WARNING skipped_checkpoint={path.name} reason={exc}",
                file=sys.stderr,
            )
            continue
        score_source = "ranking_probability" if "ranking_probability" in result else "prediction"
        score_column = checkpoint_column_name(path, used_columns)
        score_columns.append(score_column)
        keep = [*key_columns, score_source]
        model_scores = result.loc[:, keep].rename(columns={score_source: score_column})
        if combined is None:
            base_columns = [
                column for column in result.columns
                if column not in {"predicted_rank", "ranking_probability", "prediction", "predicted_class", "predicted_label"}
                and not column.startswith("probability_")
            ]
            combined = result.loc[:, base_columns].copy()
            combined = combined.merge(model_scores, on=key_columns, how="left", validate="one_to_one")
        else:
            combined = combined.merge(model_scores, on=key_columns, how="left", validate="one_to_one")
    if combined is None:
        details = "; ".join(f"{path.name}: {reason}" for path, reason in skipped)
        raise ValueError(f"No checkpoint could score race {args.race_id}. {details}")
    if skipped:
        tqdm.write(
            f"PREDICTION SUMMARY compatible_models={len(score_columns)} "
            f"skipped_models={len(skipped)}",
        )
    combined["ensemble_score"] = combined[score_columns].mean(axis=1)
    combined = combined.sort_values(
        ["ensemble_score", args.runner_id_column], ascending=[False, True]
    ).reset_index(drop=True)
    combined.insert(0, "predicted_rank", np.arange(1, len(combined) + 1))
    return combined


def print_model_rankings(result: pd.DataFrame, runner_id_column: str) -> None:
    """Print an independently ranked runner table for every compatible model."""
    score_columns = [
        column for column in result.columns
        if column.startswith("model_") and column.endswith("_score")
    ]
    display_columns = ["race_id", runner_id_column]
    for market_column in ("open_price", "fluc1", "fluc2", "is_winner", "top3_mask"):
        if market_column in result.columns:
            display_columns.append(market_column)
    for score_column in score_columns:
        ranking = result.loc[:, [*display_columns, score_column]].sort_values(
            [score_column, runner_id_column], ascending=[False, True]
        ).reset_index(drop=True)
        ranking.insert(0, "predicted_rank", np.arange(1, len(ranking) + 1))
        print(f"\nMODEL RANKING checkpoint={score_column.removeprefix('model_').removesuffix('_score')}")
        print(ranking.to_string(index=False))


def finished_race_ids(
    db_path: Path, maximum: int, competition_id: int | None = None
) -> list[int]:
    """Return matching finished targets chronologically, capped to the latest N."""
    if maximum < 0:
        raise ValueError("--backtest-max-races must be zero or positive.")
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        where = "status = 'finished'"
        params: tuple[int, ...] = ()
        if competition_id is not None:
            where += " AND competition_id = ?"
            params = (competition_id,)
        rows = connection.execute(
            f"SELECT race_id FROM race_runners WHERE {where} "
            "GROUP BY race_id ORDER BY MIN(start_time_iso), race_id",
            params,
        ).fetchall()
    finally:
        connection.close()
    race_ids = [int(row[0]) for row in rows]
    return race_ids if maximum == 0 else race_ids[-maximum:]


def prepare_backtest_native_data(
    db_path: Path,
    feature_columns: Sequence[str],
    metadata: Mapping[str, Any],
    maximum: int,
    competition_id: int | None = None,
    target_race_id: str | None = None,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], list[tuple[pd.Timestamp, int]]]:
    """Load finished targets and the eligible causal context pool once."""
    if maximum < 0:
        raise ValueError("--backtest-max-races must be zero or positive.")
    selected = list(dict.fromkeys([
        "race_id", "start_time_iso", "competition_id", "runner_number",
        "race_number", *feature_columns, "top3_mask", "is_winner",
    ]))
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        target_where = "status = 'finished'"
        target_params: list[int] = []
        if competition_id is not None:
            target_where += " AND competition_id = ?"
            target_params.append(competition_id)
        if target_race_id is not None:
            try:
                selected_race_id = int(target_race_id)
            except ValueError as exc:
                raise ValueError("--race-id must be an integer in native backtest mode") from exc
            target_where += " AND race_id = ?"
            target_params.append(selected_race_id)
        targets = pd.read_sql_query(
            f"SELECT {', '.join(quote_identifier(column) for column in selected)} "
            f"FROM race_runners WHERE {target_where} "
            "ORDER BY start_time_iso, race_id, runner_number",
            connection,
            params=target_params,
        )
        pool = pd.read_sql_query(
            f"SELECT {', '.join(quote_identifier(column) for column in selected[:-1])} "
            "FROM race_runners WHERE status = 'finished' AND top3_mask IN (0, 1) "
            "ORDER BY start_time_iso, race_id, runner_number",
            connection,
        )
    finally:
        connection.close()
    if maximum and target_race_id is None:
        # A cap is operationally useful for testing current performance. Keep
        # the latest matching races, while retaining chronological score order.
        wanted = targets.groupby("race_id", sort=False).size().index[-maximum:]
        targets = targets.loc[targets["race_id"].isin(wanted)].copy()
    minimum = metadata.get("optimizer_min_race_number")
    if minimum is not None:
        pool = pool.loc[pool["race_number"] >= int(minimum)].copy()
    summary = pool.groupby("race_id", sort=False).agg(
        start_time_iso=("start_time_iso", "min"), runners=("race_id", "size"), top3=("top3_mask", "sum")
    )
    summary = summary.loc[(summary["runners"] >= 4) & (summary["top3"] == 3)]
    context_ids = set(map(int, summary.index))
    pool = pool.loc[pool["race_id"].isin(context_ids)].copy()
    context_by_race = {int(race_id): group.copy() for race_id, group in pool.groupby("race_id", sort=False)}
    target_by_race = {int(race_id): group.copy() for race_id, group in targets.groupby("race_id", sort=False)}
    ordered_context = sorted(
        (pd.to_datetime(summary.loc[race_id, "start_time_iso"], utc=True), int(race_id))
        for race_id in summary.index
    )
    return context_by_race, target_by_race, ordered_context


def backtest(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest each checkpoint independently on every completed target race."""
    if args.data is not None:
        raise ValueError("--backtest requires native SQLite mode; omit --data.")
    all_predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    device = torch.device(args.device)
    checkpoints = checkpoint_paths(args)
    checkpoint_progress = tqdm(
        checkpoints,
        desc="Backtesting checkpoints",
        unit="model",
        dynamic_ncols=True,
        position=0,
    )
    for checkpoint in checkpoint_progress:
        checkpoint_progress.set_description(f"Backtesting {checkpoint.name}")
        checkpoint_started_at = time.perf_counter()
        try:
            model, metadata = load_model(checkpoint, device, args.strict_load)
        except Exception as exc:
            tqdm.write(
                f"WARNING skipped_checkpoint={checkpoint.name} reason={exc}",
                file=sys.stderr,
            )
            continue
        feature_columns = read_feature_columns(args, metadata)
        tqdm.write(
            f"FEATURES checkpoint={checkpoint.name} "
            + describe_feature_source(args, metadata, len(feature_columns))
        )
        context_by_race, target_by_race, ordered_context = prepare_backtest_native_data(
            args.db,
            feature_columns,
            metadata,
            args.backtest_max_races,
            args.competition_id,
            args.race_id,
        )
        race_ids = list(target_by_race)
        if not race_ids:
            detail = f" for race_id={args.race_id}" if args.race_id else ""
            if args.competition_id is not None:
                detail += f" competition_id={args.competition_id}"
            raise ValueError(f"No status='finished' race was found{detail}.")
        model_predictions: list[pd.DataFrame] = []
        skipped = 0
        top1_hits = 0
        top1_races = 0
        skip_examples: list[str] = []
        inference_started_at = time.perf_counter()
        race_progress = tqdm(
            race_ids,
            desc=f"Backtesting {checkpoint.name}",
            unit="race",
            dynamic_ncols=True,
            position=1,
            leave=False,
        )
        for race_id in race_progress:
            race_args = copy.copy(args)
            race_args.race_id = str(race_id)
            try:
                prediction = predict_one(
                    race_args, checkpoint, model, metadata,
                    (context_by_race, ordered_context), target_by_race[race_id],
                )
            except ValueError as exc:
                # Earliest races may not yet have enough historical context.
                skipped += 1
                if len(skip_examples) < 3:
                    skip_examples.append(f"race_id={race_id}: {exc}")
            else:
                score_column = "ranking_probability" if "ranking_probability" in prediction else "prediction"
                prediction = prediction.rename(columns={score_column: "model_score"})
                prediction["checkpoint"] = checkpoint.name
                model_predictions.append(prediction)
                winners = pd.to_numeric(prediction["is_winner"], errors="coerce")
                scores = pd.to_numeric(prediction["model_score"], errors="coerce")
                if int(winners.eq(1).sum()) == 1 and scores.notna().any():
                    top1_hits += int(winners.loc[scores.idxmax()] == 1)
                    top1_races += 1
            race_progress.set_postfix(
                scored=len(model_predictions),
                skipped=skipped,
                top1_hits=top1_hits,
                top1=(f"{top1_hits / top1_races:.3f}" if top1_races else "-"),
                refresh=False,
            )
        race_progress.close()
        inference_seconds = time.perf_counter() - inference_started_at
        if not model_predictions:
            detail = " | ".join(skip_examples)
            tqdm.write(
                f"WARNING skipped_checkpoint={checkpoint.name} reason=no scoreable races"
                + (f" examples={detail}" if detail else ""),
                file=sys.stderr,
            )
            continue
        model_frame = pd.concat(model_predictions, ignore_index=True)
        labels = pd.to_numeric(model_frame["top3_mask"], errors="coerce")
        valid = labels.isin([0, 1])
        if valid.any():
            metrics = probability_metrics(
                labels.loc[valid].to_numpy(dtype=np.int64),
                model_frame.loc[valid, "model_score"].to_numpy(dtype=np.float64),
                model_frame.loc[valid, args.race_id_column].to_numpy(dtype=np.int64),
            )
            metrics["exact_top1_rate"] = exact_top1_rate(
                model_frame.loc[valid],
                "model_score",
                args.race_id_column,
            )
        else:
            metrics = {"complete_races": 0}
        checkpoint_seconds = time.perf_counter() - checkpoint_started_at
        seconds_per_target = inference_seconds / len(race_ids)
        metric_rows.append({
            "checkpoint": checkpoint.name,
            "targets_requested": len(race_ids),
            "targets_scored": int(model_frame[args.race_id_column].nunique()),
            "targets_skipped": skipped,
            "inference_seconds": inference_seconds,
            "seconds_per_target": seconds_per_target,
            "checkpoint_seconds": checkpoint_seconds,
            **metrics,
        })
        all_predictions.append(model_frame)
    if not all_predictions:
        raise ValueError("No checkpoint produced any backtest predictions.")
    market_frame = all_predictions[0]
    market_labels = pd.to_numeric(market_frame["top3_mask"], errors="coerce")
    market_prices = pd.to_numeric(market_frame["fluc2"], errors="coerce")
    market_rows_valid = market_labels.isin([0, 1]) & market_prices.gt(0)
    fully_covered_races = market_frame.loc[market_rows_valid].groupby(
        args.race_id_column
    ).size()
    all_race_sizes = market_frame.groupby(args.race_id_column).size()
    fully_covered_races = fully_covered_races.index[
        fully_covered_races.eq(all_race_sizes.reindex(fully_covered_races.index))
    ]
    market_mask = market_rows_valid & market_frame[args.race_id_column].isin(
        fully_covered_races
    )
    if market_mask.any():
        market_metrics = race_top3_metrics(
            market_labels.loc[market_mask].to_numpy(dtype=np.int64),
            market_rank_scores(
                market_prices.loc[market_mask].to_numpy(dtype=np.float64)
            ),
            market_frame.loc[market_mask, args.race_id_column].to_numpy(
                dtype=np.int64
            ),
        )
        market_scored_frame = market_frame.loc[market_mask].copy()
        market_scored_frame["market_score"] = market_rank_scores(
            market_prices.loc[market_mask].to_numpy(dtype=np.float64)
        )
        market_metrics["exact_top1_rate"] = exact_top1_rate(
            market_scored_frame,
            "market_score",
            args.race_id_column,
        )
        metric_rows.append(
            {
                "checkpoint": "fluc2_market",
                "targets_requested": int(
                    market_frame[args.race_id_column].nunique()
                ),
                "targets_scored": int(len(fully_covered_races)),
                "targets_skipped": int(
                    market_frame[args.race_id_column].nunique()
                    - len(fully_covered_races)
                ),
                "inference_seconds": 0.0,
                "seconds_per_target": 0.0,
                "checkpoint_seconds": 0.0,
                "roc_auc": float("nan"),
                "logloss": float("nan"),
                **market_metrics,
            }
        )
    return pd.concat(all_predictions, ignore_index=True), pd.DataFrame(metric_rows)


def exact_top1_rate(
    frame: pd.DataFrame, score_column: str, race_id_column: str
) -> float:
    """Return winner accuracy among races with exactly one known winner."""
    correct = 0
    complete = 0
    for _, race in frame.groupby(race_id_column, sort=False):
        winners = pd.to_numeric(race["is_winner"], errors="coerce")
        scores = pd.to_numeric(race[score_column], errors="coerce")
        if int(winners.eq(1).sum()) != 1 or not scores.notna().any():
            continue
        predicted_row = scores.idxmax()
        correct += int(winners.loc[predicted_row] == 1)
        complete += 1
    return correct / complete if complete else float("nan")


def ranked_backtest_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return the compact model leaderboard shown after a backtest."""
    ranking = metrics.sort_values(
        by=[
            "top3_recall",
            "contained_top5_rate",
            "contained_top4_rate",
            "roc_auc",
            "logloss",
        ],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    columns = [
        "rank",
        "checkpoint",
        "exact_top1_rate",
        "top3_recall",
        "exact_top3_set_rate",
        "contained_top4_rate",
        "contained_top5_rate",
        "contained_top6_rate",
        "roc_auc",
        "logloss",
        "targets_scored",
    ]
    return ranking.loc[:, [column for column in columns if column in ranking]]


def main() -> int:
    args = parse_args()
    if not args.show_warnings:
        warnings.filterwarnings("ignore")
    try:
        if getattr(args, "include_competition_history_context", False) and args.backtest:
            raise ValueError(
                "--include-competition-history-context currently supports "
                "single-race prediction only; omit --backtest."
            )
        if (
            getattr(args, "include_competition_history_context", False)
            and args.data is not None
        ):
            raise ValueError(
                "--include-competition-history-context requires native SQLite mode; "
                "omit --data."
            )
        if args.backtest:
            result, metrics = backtest(args)
        else:
            if not args.race_id:
                raise ValueError("--race-id is required unless --backtest is used.")
            result = predict(args)
            metrics = None
    except Exception as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 1

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        if metrics is not None:
            leaderboard = ranked_backtest_metrics(metrics)
            print("MODEL RANKING")
            print(
                leaderboard.to_string(
                    index=False,
                    float_format=lambda value: f"{value:.4f}",
                )
            )
        else:
            print(result.to_string(index=False))
            print_model_rankings(result, args.runner_id_column)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"\nSaved {len(result)} predictions to {args.output}")
        if metrics is not None:
            metrics_path = args.output.with_name(f"{args.output.stem}_metrics.csv")
            metrics.to_csv(metrics_path, index=False)
            print(f"Saved {len(metrics)} model metric rows to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
