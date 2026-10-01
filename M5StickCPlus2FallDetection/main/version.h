#pragma once

// Single source of truth for the detector firmware version used by Internet OTA.
// Bump this before building a release. tools/prepare_release.py reads the same file.
#define FALL_FIRMWARE_VERSION "1.0.0"
