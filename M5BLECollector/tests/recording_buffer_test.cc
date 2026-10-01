#include <cassert>
#include <cstdint>
#include <vector>
#include "../firmware/main/recording_buffer.h"

using m5ble::RecordState;
using m5ble::RecordingBuffer;

int main() {
  std::vector<uint8_t> storage(8 * m5ble::kRecordBytes);
  RecordingBuffer b(storage.data(), 8);
  const int16_t sample[6] = {1, 2, 3, 4, 5, 6};

  assert(b.Start(11));
  assert(b.State() == RecordState::Recording);
  assert(b.Append(1000, sample, 0));
  assert(b.Append(2000, sample, m5ble::TimingGap));
  assert(b.Pending() == 2);

  // STOP must not expose data for transfer. A trial stays local until KEEP.
  b.Stop();
  assert(b.State() == RecordState::Review);
  uint8_t packet[8 * m5ble::kRecordBytes]{};
  assert(b.Peek(packet, 8) == 0);
  assert(!b.CanLeave());

  // DISCARD clears the complete local trial without ever entering Saving.
  assert(b.Discard());
  assert(b.State() == RecordState::Ready);
  assert(b.Pending() == 0);
  assert(b.Session() == 0);
  assert(b.CanLeave());

  // KEEP is the only transition that permits transfer/ACK.
  assert(b.Start(22));
  for (int i = 0; i < 4; ++i) assert(b.Append(3000 + i * 1000, sample, 0));
  b.Stop();
  assert(b.Keep());
  assert(b.State() == RecordState::Saving);
  assert(b.Peek(packet, 8) == 4);
  const uint32_t exclusive = m5ble::Get32(packet + 3 * m5ble::kRecordBytes) + 1;
  b.Offered(exclusive);
  assert(b.Ack(22, exclusive));
  assert(b.State() == RecordState::Complete);
  assert(b.Pending() == 0);
  assert(b.CanLeave());

  // A full buffer is also reviewable: it can be discarded or kept.
  std::vector<uint8_t> small_storage(2 * m5ble::kRecordBytes);
  RecordingBuffer full(small_storage.data(), 2);
  assert(full.Start(33));
  assert(full.Append(1, sample, 0));
  assert(full.Append(2, sample, 0));
  assert(!full.Append(3, sample, 0));
  assert(full.State() == RecordState::Full);
  assert(full.Overflowed());
  assert(full.Discard());
  assert(full.State() == RecordState::Ready);
  assert(!full.Overflowed());

  return 0;
}
