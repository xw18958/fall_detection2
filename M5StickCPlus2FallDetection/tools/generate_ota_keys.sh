#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECURITY_DIR="$ROOT_DIR/security"
PRIVATE_KEY="$ROOT_DIR/main/ota_private.pem"
PUBLIC_KEY="$SECURITY_DIR/ota_public.pem"
PRIVATE_BLOB="$ROOT_DIR/main/ota_private_blob.S"

mkdir -p "$SECURITY_DIR"

if [[ -e "$PRIVATE_KEY" || -e "$PUBLIC_KEY" || -e "$PRIVATE_BLOB" ]]; then
  echo "Refusing to overwrite existing OTA key material." >&2
  echo "  private: $PRIVATE_KEY" >&2
  echo "  public:  $PUBLIC_KEY" >&2
  echo "  blob:    $PRIVATE_BLOB" >&2
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

# PlatformIO's ESP-IDF integration has been unreliable when asked to embed this
# local text file through target_add_binary_data/EMBED_TXTFILES. Generate a tiny
# assembly source instead. It exports the same symbols that ESP-IDF's embedding
# helper would normally create, but it is compiled as an ordinary source file.
{
  echo '.section .rodata'
  echo '.balign 4'
  echo '.global _binary_ota_private_pem_start'
  echo '.global _binary_ota_private_pem_end'
  echo '_binary_ota_private_pem_start:'
  while IFS= read -r line || [[ -n "$line" ]]; do
    printf '  .ascii "%s\\n"\n' "$line"
  done < "$PRIVATE_KEY"
  echo '  .byte 0'
  echo '_binary_ota_private_pem_end:'
} > "$PRIVATE_BLOB"
chmod 600 "$PRIVATE_BLOB"

cat <<EOF
Generated development OTA key material:
  private: $PRIVATE_KEY
  public:  $PUBLIC_KEY
  blob:    $PRIVATE_BLOB

All three files are gitignored. Back up the private key somewhere safe before
using it on devices you care about. Rebuilding a device with a different private
key means it cannot decrypt releases encrypted for the old key.
EOF
