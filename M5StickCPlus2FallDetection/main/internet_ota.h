#pragma once

namespace fall_internet_ota {

// Makes one bounded boot-time version check and installs a newer image if found.
// Internet failures are logged and treated as a no-update result.
void CheckAndUpdate();

}  // namespace fall_internet_ota
