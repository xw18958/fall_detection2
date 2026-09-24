#!/usr/bin/env python3
"""Receive batched AirPods motion samples and append them to per-session CSVs."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from websockets.asyncio.server import serve

HOST = os.environ.get("MOTION_HOST", "127.0.0.1")
PORT = int(os.environ.get("MOTION_PORT", "8765"))
DATA_DIR = Path(os.environ.get("MOTION_DATA_DIR", "data"))
FIELDS = ("timestamp", "ax", "ay", "az", "gx", "gy", "gz")


def validate_samples(message: str) -> list[dict[str, float]]:
    payload = json.loads(message)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError('Expected a JSON object with a "samples" array')

    rows: list[dict[str, float]] = []
    for index, sample in enumerate(payload["samples"]):
        if not isinstance(sample, dict):
            raise ValueError(f"samples[{index}] must be an object")
        row: dict[str, float] = {}
        for field in FIELDS:
            value = sample.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"samples[{index}].{field} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"samples[{index}].{field} must be finite")
            row[field] = value
        rows.append(row)
    return rows


def new_recording_path() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
    path = DATA_DIR / f"recording_{stamp}.csv"
    suffix = 1
    while path.exists():
        path = DATA_DIR / f"recording_{stamp}_{suffix:02d}.csv"
        suffix += 1
    return path


async def receive_recording(websocket: Any) -> None:
    path = new_recording_path()
    count = 0
    print(f"Client connected; recording to {path}", flush=True)
    try:
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            output.flush()
            async for message in websocket:
                try:
                    rows = validate_samples(message)
                except (json.JSONDecodeError, ValueError) as exc:
                    await websocket.send(json.dumps({"error": str(exc)}))
                    continue
                writer.writerows(rows)
                output.flush()
                count += len(rows)
                print(f"Received {len(rows)} samples ({count} total)", flush=True)
    finally:
        print(f"Client disconnected; saved {count} samples to {path}", flush=True)


async def main() -> None:
    print(f"Listening on ws://{HOST}:{PORT}; CSV directory: {DATA_DIR}", flush=True)
    async with serve(receive_recording, HOST, PORT, max_size=2**20):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped", flush=True)
