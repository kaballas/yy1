#!/usr/bin/env python3
"""Populate text ``*_vec`` columns with deterministic hash-based vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from list_database_features import DEFAULT_DATABASE, quote_identifier
from update_text_vectors import ROOT, text_vector_pairs, value_token


DEFAULT_METADATA = ROOT / "outputs" / "race_runner_text_hash_vectors.json"


def hash_vector(token: str, dimensions: int, seed: str = "race-text-v1") -> np.ndarray:
    """Map a token to a stable, unit-length float32 vector."""
    if dimensions < 1:
        raise ValueError("dimensions must be positive")
    material = (seed + "\x00" + token).encode("utf-8")
    required = dimensions * 4
    output = bytearray()
    counter = 0
    while len(output) < required:
        output.extend(
            hashlib.sha256(material + counter.to_bytes(8, "little")).digest()
        )
        counter += 1
    integers = np.frombuffer(bytes(output[:required]), dtype="<u4").astype(np.float64)
    vector = integers / np.float64(2**32 - 1) * 2.0 - 1.0
    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    return vector.astype("<f4")


def populate_hash_vectors(
    database: Path,
    table: str,
    pairs: Sequence[tuple[str, str]],
    dimensions: int = 32,
    seed: str = "race-text-v1",
    batch_size: int = 500,
) -> tuple[int, int]:
    if dimensions < 1 or batch_size < 1:
        raise ValueError("dimensions and batch_size must be positive")
    sources = [source for source, _vector in pairs]
    source_sql = ", ".join(quote_identifier(name) for name in sources)
    assignments = ", ".join(
        f"{quote_identifier(vector)} = ?" for _source, vector in pairs
    )
    cache: dict[str, bytes] = {}
    rows_updated = 0
    vectors_written = 0
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
                        continue
                    blob = cache.get(token)
                    if blob is None:
                        blob = hash_vector(token, dimensions, seed).tobytes(order="C")
                        cache[token] = blob
                    blobs.append(blob)
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
                f"updated_rows={rows_updated:,} vectors_written={vectors_written:,} "
                f"unique_values={len(cache):,}",
                flush=True,
            )
    return rows_updated, vectors_written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--table", default="race_runners")
    parser.add_argument("--dimensions", type=int, default=32)
    parser.add_argument("--seed", default="race-text-v1")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        pairs = text_vector_pairs(args.db, args.table)
        print(
            f"hash_vector_update table={args.table} text_features={len(pairs)} "
            f"dimensions={args.dimensions} seed={args.seed!r}",
            flush=True,
        )
        rows, written = populate_hash_vectors(
            args.db, args.table, pairs, args.dimensions, args.seed, args.batch_size
        )
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(
            json.dumps(
                {
                    "method": "sha256_counter_unit_float32",
                    "database": str(args.db.resolve()),
                    "table": args.table,
                    "dimensions": args.dimensions,
                    "seed": args.seed,
                    "token_format": "<column>\\x1f<complete stripped value>",
                    "source_columns": [source for source, _vector in pairs],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"COMPLETE rows_updated={rows:,} vectors_written={written:,} "
            f"metadata={args.metadata}",
            flush=True,
        )
    except (OSError, sqlite3.Error, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
