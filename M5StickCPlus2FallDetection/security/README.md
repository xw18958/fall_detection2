# Development OTA keys

This directory intentionally contains **no real key material** in Git.

For the current GitHub-hosted development OTA flow, generate one RSA-3072 key pair locally:

```bash
bash tools/generate_ota_keys.sh
```

This creates gitignored files:

```text
security/ota_private.pem
security/ota_public.pem
```

`ota_private.pem` is embedded into firmware builds so the device can decrypt the public GitHub release artifact. `ota_public.pem` is used by `tools/prepare_release.py` to encrypt `firmware.bin` before upload.

This is deliberately a **development-stage** confidentiality measure. It protects the model from being trivially downloaded from a public GitHub Release, but it does not stop a skilled attacker with physical access from dumping an unprotected ESP32 and recovering firmware/key material. Before commercial production, add ESP32 Flash Encryption + Secure Boot and use a production key-provisioning process.

Never commit either generated key. Back up the private key before deploying it to devices; a device built with one private key cannot decrypt releases made for a different key.
