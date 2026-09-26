#pragma once

namespace fall_ota {

// Starts a local Wi-Fi access point and browser-based OTA update server.
// Connect to SSID "FallDetector-OTA" with password "fallupdate", then open
// http://192.168.4.1/ and upload the PlatformIO firmware.bin file.
bool Start();

// True only while a firmware image is actively being written to the inactive
// OTA slot. The sensor/inference loop can pause during this short period.
bool InProgress();

// If the current image was booted as a pending OTA image, mark it valid after
// the application has completed its startup self-tests.
bool MarkRunningImageValid();

// If the current image is pending verification, mark it invalid and reboot
// into the previous known-good OTA slot. Returns false when no rollback was
// needed or rollback could not be started.
bool RollbackPendingImageAndReboot();

}  // namespace fall_ota
