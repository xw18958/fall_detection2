#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECURITY_DIR="$ROOT_DIR/security"
PRIVATE_KEY="$ROOT_DIR/main/ota_private.pem"
PUBLIC_KEY="$SECURITY_DIR/ota_public.pem"

mkdir -p "$SECURITY_DIR"

if [[ -e "$PRIVATE_KEY" || -e "$PUBLIC_KEY" ]]; then
  echo "Refusing to overwrite existing OTA key material." >&2
  echo "  private: $PRIVATE_KEY" >&2
  echo "  public:  $PUBLIC_KEY" >&2
  echo "Move/delete the existing files deliberately if you want to rotate the key." >&2
  exit 1
fi

command -v openssl >/dev/null 2>&1 || {
  echo "openssl is required but was not found." >&2
  exit 1
}

openssl genpkey -algorithm RSA \
  -pkeyopt rsa_keygen_bits:3072 \
  -out "$PRIVATE_KEY"
openssl pkey -in "$PRIVATE_KEY" -pubout -out "$PUBLIC_KEY"
chmod 600 "$PRIVATE_KEY"
chmod 644 "$PUBLIC_KEY"

cat <<EOF
Generated development OTA keys:
  private: $PRIVATE_KEY
  public:  $PUBLIC_KEY

Both files are gitignored. Back up the private key somewhere safe before using it
on devices you care about. Rebuilding a device with a different private key means
it cannot decrypt releases encrypted for the old key.
EOF
