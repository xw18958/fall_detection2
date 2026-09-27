#!/usr/bin/env python3
"""Build the public GitHub OTA artifact from a local PlatformIO firmware.bin.

This script never uploads anything. It encrypts firmware.bin with the local OTA
public key, round-trip decrypts it with the local private key as a sanity check,
and writes a stable.json manifest containing the SHA-256 of the encrypted asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_HEADER = ROOT / "main" / "version.h"
FIRMWARE_BIN = ROOT / ".pio" / "build" / "m5stickc-plus2" / "firmware.bin"
PRIVATE_KEY = ROOT / "security" / "ota_private.pem"
PUBLIC_KEY = ROOT / "security" / "ota_public.pem"
DEFAULT_TOOL = (
    ROOT
    / "managed_components"
    / "espressif__esp_encrypted_img"
    / "tools"
    / "esp_enc_img_gen.py"
)
VERSION_RE = re.compile(r'^#define\s+FALL_FIRMWARE_VERSION\s+"([0-9]+\.[0-9]+\.[0-9]+)"\s*$')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_version() -> str:
    for line in VERSION_HEADER.read_text(encoding="utf-8").splitlines():
        match = VERSION_RE.match(line.strip())
        if match:
            return match.group(1)
    raise SystemExit(f"Could not read FALL_FIRMWARE_VERSION from {VERSION_HEADER}")


def find_encrypt_tool() -> Path:
    if DEFAULT_TOOL.exists():
        return DEFAULT_TOOL
    matches = sorted(ROOT.glob("managed_components/**/tools/esp_enc_img_gen.py"))
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(
        "Could not find esp_enc_img_gen.py. Run `pio run` once so ESP-IDF "
        "downloads the esp_encrypted_img managed component."
    )


def run(*args: str | Path) -> None:
    command = [str(arg) for arg in args]
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--firmware",
        type=Path,
        default=FIRMWARE_BIN,
        help="Path to plaintext PlatformIO firmware.bin",
    )
    args = parser.parse_args()

    version = read_version()
    firmware = args.firmware.resolve()
    if not firmware.is_file():
        raise SystemExit(f"Missing firmware image: {firmware}. Run `pio run` first.")
    if not PRIVATE_KEY.is_file() or not PUBLIC_KEY.is_file():
        raise SystemExit(
            "Missing OTA key pair. Run `bash tools/generate_ota_keys.sh` first."
        )

    tool = find_encrypt_tool()
    release_dir = ROOT / "dist" / f"v{version}"
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True)
    encrypted = release_dir / "firmware.enc"

    run(sys.executable, tool, "encrypt", firmware, PUBLIC_KEY, encrypted)

    # Verify the packaging step before anything can be uploaded publicly.
    with tempfile.TemporaryDirectory(prefix="fall-ota-check-") as tmp:
        decrypted = Path(tmp) / "firmware.bin"
        run(sys.executable, tool, "decrypt", encrypted, PRIVATE_KEY, decrypted)
        original_hash = sha256(firmware)
        decrypted_hash = sha256(decrypted)
        if original_hash != decrypted_hash:
            raise SystemExit(
                "Encrypted OTA round-trip verification FAILED: decrypted firmware "
                "does not match the original firmware.bin"
            )

    encrypted_hash = sha256(encrypted)
    manifest = {
        "enabled": True,
        "version": version,
        "sha256": encrypted_hash,
    }
    manifest_path = release_dir / "stable.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print()
    print("Release package verified successfully")
    print(f"  version:          {version}")
    print(f"  plaintext sha256: {sha256(firmware)}")
    print(f"  encrypted sha256: {encrypted_hash}")
    print(f"  asset:            {encrypted}")
    print(f"  manifest:         {manifest_path}")
    print()
    print("Publish in this order:")
    print(f"  1. Create GitHub Release tag v{version} and attach firmware.enc")
    print("  2. Copy the generated stable.json to ota/stable.json and push it")
    print("Never upload plaintext firmware.bin or security/ota_private.pem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
