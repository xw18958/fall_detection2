#pragma once

#include <cstddef>

namespace fall_ota {

// Starts the local recovery/provisioning access point and browser UI.
// Connect to SSID "FallDetector-OTA" with password "fallupdate", then open
// http://192.168.4.1/. The page supports both Wi-Fi provisioning and local OTA.
bool Start();

// Load Wi-Fi credentials previously provisioned through the local web page.
// Returns false when no complete credential pair has been stored yet.
bool LoadStoredWifiCredentials(char* ssid, size_t ssid_size,
                               char* password, size_t password_size);

// Remove provisioned Wi-Fi credentials from NVS.
bool ClearStoredWifiCredentials();

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
