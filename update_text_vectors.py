#!/usr/bin/env python3
"""Train Word2Vec on text features and populate their ``*_vec`` BLOB columns."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from add_text_vector_columns import planned_vector_columns
from list_database_features import DEFAULT_DATABASE, inspect_columns, quote_identifier


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "outputs" / "race_runner_text_word2vec.model"
TOKEN_SEPARATOR = "\x1f"


class EpochProgress:
    """Gensim callback that reports useful progress after every epoch."""

    def __init__(self, epochs: int):
        self.epochs = epochs
        self.epoch = 0
        self.started = 0.0
        self.previous_loss = 0.0

    def on_train_begin(self, model) -> None:
        self.started = time.monotonic()
        print("word2vec_training_started", flush=True)

    def on_epoch_begin(self, model) -> None:
        pass

    def on_epoch_end(self, model) -> None:
        self.epoch += 1
        cumulative_loss = float(model.get_latest_training_loss())
        epoch_loss = cumulative_loss - self.previous_loss
        self.previous_loss = cumulative_loss
        elapsed = time.monotonic() - self.started
        average = elapsed / self.epoch
        eta = average * max(0, self.epochs - self.epoch)
        print(
            f"epoch={self.epoch}/{self.epochs} loss={epoch_loss:.6f} "
            f"cumulative_loss={cumulative_loss:.6f} elapsed={elapsed:.1f}s "
            f"eta={eta:.1f}s",
            flush=True,
        )

    def on_train_end(self, model) -> None:
        print(
            f"word2vec_training_complete epochs={self.epoch} "
            f"elapsed={time.monotonic() - self.started:.1f}s",
            flush=True,
        )


def text_vector_pairs(database: Path, table: str) -> list[tuple[str, str]]:
    columns = inspect_columns(database, table)
    names = {column.name for column in columns}
    pairs = [
        (column.name, f"{column.name}_vec")
        for column in columns
        if column.category == "text" and column.is_feature
        and f"{column.name}_vec" in names
    ]
    missing, _existing = planned_vector_columns(database, table)
    if missing:
        names = ", ".join(vector for _source, vector in missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(
            f"Missing {len(missing)} vector columns ({names}{suffix}); run "
            "add_text_vector_columns.py --apply first"
        )
    if not pairs:
        raise ValueError("No text feature/vector column pairs were found")
    return pairs


def value_token(column: str, value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return f"{column}{TOKEN_SEPARATOR}{text}"


class SQLiteSentences:
    """Restartable corpus yielding one sentence of categorical values per row."""

    def __init__(self, database: Path, table: str, columns: Sequence[str]):
        self.database = database
        self.table = table
        self.columns = list(columns)

    def __iter__(self) -> Iterator[list[str]]:
        selected = ", ".join(quote_identifier(name) for name in self.columns)
        uri = f"file:{self.database.resolve()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            cursor = connection.execute(
                f"SELECT {selected} FROM {quote_identifier(self.table)}"
            )
            for row in cursor:
                sentence = [
                    token
                    for name, value in zip(self.columns, row)
                    if (token := value_token(name, value)) is not None
                ]
                if sentence:
                    yield sentence


def train_model(
    database: Path,
    table: str,
    columns: Sequence[str],
    model_path: Path,
    dimensions: int,
    window: int,
    epochs: int,
    min_count: int,
    workers: int,
    seed: int,
):
    from gensim.models import Word2Vec

    if dimensions < 1 or window < 1 or epochs < 1 or min_count < 1 or workers < 1:
        raise ValueError("dimensions, window, epochs, min_count, and workers must be positive")
    corpus = SQLiteSentences(database, table, columns)
    model = Word2Vec(
        vector_size=dimensions,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,
        seed=seed,
    )
    print("building_word2vec_vocabulary", flush=True)
    model.build_vocab(corpus)
    if not model.wv:
        raise ValueError("Word2Vec vocabulary is empty")
    print(
        f"vocabulary_ready tokens={len(model.wv):,} "
        f"training_rows={model.corpus_count:,}",
        flush=True,
    )
    model.train(
        corpus,
        total_examples=model.corpus_count,
        epochs=epochs,
        compute_loss=True,
        callbacks=[EpochProgress(epochs)],
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    metadata = {
        "database": str(database.resolve()),
        "table": table,
        "source_columns": list(columns),
        "vector_size": dimensions,
        "window": window,
        "epochs": epochs,
        "min_count": min_count,
        "sg": 1,
        "seed": seed,
        "token_format": "<column>\\x1f<complete stripped value>",
        "vocabulary_size": len(model.wv),
    }
    model_path.with_suffix(model_path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return model


def populate_vectors(
    database: Path,
    table: str,
    pairs: Sequence[tuple[str, str]],
    keyed_vectors,
    batch_size: int = 500,
) -> tuple[int, int, int]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    sources = [source for source, _vector in pairs]
    source_sql = ", ".join(quote_identifier(name) for name in sources)
    assignments = ", ".join(
        f"{quote_identifier(vector)} = ?" for _source, vector in pairs
    )
    rows_updated = 0
    vectors_written = 0
    missing_vocabulary = 0
    last_rowid = -1
    with sqlite3.connect(database) as connection:
        while True:
            rows = connection.execute(
                f"SELECT rowid, {source_sql} FROM {quote_identifier(table)} "
                "WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (last_rowid, batch_size),
            ).fetchall()
            if not rows:
                break
            updates = []
            for row in rows:
                blobs: list[bytes | None] = []
                for source, value in zip(sources, row[1:]):
                    token = value_token(source, value)
                    if token is None:
                        blobs.append(None)
                    elif token not in keyed_vectors:
                        blobs.append(None)
                        missing_vocabulary += 1
                    else:
                        vector = np.asarray(keyed_vectors[token], dtype="<f4")
                        blobs.append(vector.tobytes(order="C"))
                        vectors_written += 1
                updates.append((*blobs, int(row[0])))
            connection.executemany(
                f"UPDATE {quote_identifier(table)} SET {assignments} WHERE rowid = ?",
                updates,
            )
            connection.commit()
            rows_updated += len(rows)
            last_rowid = int(rows[-1][0])
            print(
                f"updated_rows={rows_updated:,} vectors_written={vectors_written:,}",
                flush=True,
            )
    return rows_updated, vectors_written, missing_vocabulary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--table", default="race_runners")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dimensions", type=int, default=32)
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Word2Vec context window; 100 covers all text values on a runner row.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument(
        "--update-only",
        action="store_true",
        help="Load --model and populate the database without retraining.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.train_only and args.update_only:
        raise SystemExit("--train-only and --update-only cannot be used together")
    try:
        pairs = text_vector_pairs(args.db, args.table)
        sources = [source for source, _vector in pairs]
        if args.update_only:
            from gensim.models import Word2Vec

            if not args.model.is_file():
                raise ValueError(f"Word2Vec model does not exist: {args.model}")
            model = Word2Vec.load(str(args.model))
        else:
            print(
                f"training rows_table={args.table} text_features={len(sources)} "
                f"dimensions={args.dimensions} epochs={args.epochs}"
            )
            model = train_model(
                args.db, args.table, sources, args.model, args.dimensions,
                args.window, args.epochs, args.min_count, args.workers, args.seed,
            )
            print(f"model_saved={args.model} vocabulary={len(model.wv):,}")
        if not args.train_only:
            rows, written, missing = populate_vectors(
                args.db, args.table, pairs, model.wv, args.batch_size
            )
            print(
                f"COMPLETE rows_updated={rows:,} vectors_written={written:,} "
                f"missing_vocabulary={missing:,} vector_dimensions={model.vector_size}"
            )
    except (OSError, sqlite3.Error, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
