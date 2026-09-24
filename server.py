#!/usr/bin/env python3
"""Receive AirPods motion batches and append them to per-session CSV files."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from websockets.asyncio.server import serve

HOST = os.environ.get("MOTION_HOST", "127.0.0.1")
PORT = int(os.environ.get("MOTION_PORT", "8765"))
DATA_DIR = Path(os.environ.get("MOTION_DATA_DIR", "data"))
SCHEMA_VERSION = 2
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,96}$")

FIELDS = (
    "seq",
    "coremotion_timestamp",
    "host_timestamp_utc",
    "sensor_location",
    "user_accel_x",
    "user_accel_y",
    "user_accel_z",
    "gravity_x",
    "gravity_y",
    "gravity_z",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
)
FLOAT_FIELDS = tuple(field for field in FIELDS if field not in {"seq", "sensor_location"})


def validate_message(message: str | bytes) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(message)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION}")

    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not SESSION_RE.fullmatch(session_id):
        raise ValueError("Invalid session_id")

    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError('Expected a "samples" array')

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"samples[{index}] must be an object")

        seq = sample.get("seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            raise ValueError(f"samples[{index}].seq must be a non-negative integer")

        sensor_location = sample.get("sensor_location")
        if sensor_location not in {"left", "right", "unknown"}:
            raise ValueError(
                f"samples[{index}].sensor_location must be left, right, or unknown"
            )

        row: dict[str, Any] = {"seq": seq, "sensor_location": sensor_location}
        for field in FLOAT_FIELDS:
            value = sample.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"samples[{index}].{field} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"samples[{index}].{field} must be finite")
            row[field] = value
        rows.append(row)

    return session_id, rows


def recording_path(session_id: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{session_id}.csv"


def open_session_file(session_id: str) -> tuple[Path, Any, csv.DictWriter]:
    path = recording_path(session_id)
    needs_header = not path.exists() or path.stat().st_size == 0
    output = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(output, fieldnames=FIELDS)
    if needs_header:
        writer.writeheader()
        output.flush()
    return path, output, writer


async def receive_recording(websocket: Any) -> None:
    session_id: str | None = None
    path: Path | None = None
    output = None
    writer: csv.DictWriter | None = None
    count = 0

    print("Client connected; waiting for first motion batch", flush=True)
    try:
        async for message in websocket:
            try:
                incoming_session_id, rows = validate_message(message)
            except (json.JSONDecodeError, ValueError) as exc:
                await websocket.send(json.dumps({"error": str(exc)}))
                continue

            if session_id is None:
                session_id = incoming_session_id
                path, output, writer = open_session_file(session_id)
                print(f"Session {session_id}; recording to {path}", flush=True)
            elif incoming_session_id != session_id:
                await websocket.send(
                    json.dumps({"error": "session_id changed within one WebSocket connection"})
                )
                continue

            if not rows:
                continue
            assert output is not None and writer is not None
            writer.writerows(rows)
            output.flush()
            count += len(rows)
            print(f"Received {len(rows)} samples ({count} this connection)", flush=True)
    finally:
        if output is not None:
            output.flush()
            output.close()
        if session_id is None:
            print("Client disconnected before sending motion data", flush=True)
        else:
            print(
                f"Client disconnected; appended {count} samples to {path}",
                flush=True,
            )


async def main() -> None:
    print(f"Listening on ws://{HOST}:{PORT}; CSV directory: {DATA_DIR}", flush=True)
    async with serve(receive_recording, HOST, PORT, max_size=2**20):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped", flush=True)
