# Development OTA keys

This directory intentionally contains **no real private key material** in Git.

For the current GitHub-hosted development OTA flow, generate one RSA-3072 key pair locally:

```bash
bash tools/generate_ota_keys.sh
```

This creates three gitignored local build inputs:

```text
main/ota_private.pem          # RSA private key
main/ota_private_blob.S       # generated assembly containing the same private key
security/ota_public.pem       # matching public key
```

`main/ota_private_blob.S` is compiled into development firmware so the device can decrypt `firmware.enc`. The assembly file is only a PlatformIO/ESP-IDF build workaround; it contains the same secret material as `main/ota_private.pem` and must be protected the same way.

`security/ota_public.pem` is used by `tools/prepare_release.py` to encrypt the locally built `firmware.bin`. The script then decrypts the package again locally and checks that it exactly matches the original firmware before producing the release manifest.

This is deliberately a **development-stage confidentiality measure**. It prevents the model from being trivially recovered by downloading a public GitHub Release, but it does not stop a skilled attacker with physical access from dumping an unprotected ESP32 and recovering firmware/key material. Secure Boot and Flash Encryption are intentionally postponed until the hardware/software design is stable.

Never commit any of the generated key files. Back up `main/ota_private.pem` before deploying it to a device you care about; a device built with one private key cannot decrypt releases encrypted for a different key.
