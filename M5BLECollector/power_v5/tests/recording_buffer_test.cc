#include "../firmware/main/recording_buffer.h"
#include <cassert>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
  using namespace m5ble;
  std::vector<uint8_t> storage(8 * kRecordBytes);
  uint8_t packet[8 * kRecordBytes]{};
  const int16_t sample[6] = {-32768, 32767, -123, 123, 0, -1};
  RecordingBuffer b(storage.data(), 8);

  assert(b.CanLeave());
  assert(b.Start(11));
  assert(b.State() == RecordState::Recording);
  assert(b.Append(1000000, sample, 0));
  assert(b.Append(1033333, sample, TimingGap));
  assert(b.Pending() == 2);

  // STOP moves to REVIEW and must not expose any record for BLE transfer.
  b.Stop();
  assert(b.State() == RecordState::Review);
  assert(b.Peek(packet, 8) == 0);
  assert(!b.CanLeave());

  // DISCARD deletes the local trial without entering Saving.
  assert(b.Discard());
  assert(b.State() == RecordState::Ready);
  assert(b.Pending() == 0);
  assert(b.Session() == 0);
  assert(b.CanLeave());

  // KEEP is the only transition that exposes records to the transfer path.
  assert(b.Start(22));
  for (int i = 0; i < 4; ++i) assert(b.Append(2000000 + uint64_t(i) * 33333, sample, 0));
  b.Stop();
  assert(b.State() == RecordState::Review);
  assert(b.Keep());
  assert(b.State() == RecordState::Saving);
  assert(b.Peek(packet, 8) == 4);
  assert(!b.Ack(22, 4));  // Not yet offered over BLE.
  b.Offered(4);
  assert(!b.Ack(99, 4));  // Wrong session.
  assert(!b.Ack(22, 5));  // Future ACK.
  assert(b.Ack(22, 2));
  assert(b.Pending() == 2);
  assert(b.Peek(packet, 8) == 2);
  b.Offered(4);
  assert(b.Ack(22, 4));
  assert(b.State() == RecordState::Complete);
  assert(b.Pending() == 0);
  assert(b.CanLeave());

  // Full-buffer trials are still explicitly KEEP/DISCARD decisions.
  std::vector<uint8_t> small_storage(2 * kRecordBytes);
  RecordingBuffer full(small_storage.data(), 2);
  assert(full.Start(33));
  assert(full.Append(1, sample, 0));
  assert(full.Append(2, sample, 0));
  assert(!full.Append(3, sample, 0));
  assert(full.State() == RecordState::Full);
  assert(full.Overflowed());
  assert(full.Peek(packet, 8) == 0);
  assert(full.Discard());
  assert(full.State() == RecordState::Ready);
  assert(!full.Overflowed());

  // Produce the cross-language wire fixture only after KEEP.
  if (argc == 2) {
    uint8_t fixture_storage[3 * kRecordBytes];
    RecordingBuffer fixture(fixture_storage, 3);
    assert(fixture.Start(42));
    for (uint32_t i = 0; i < 3; ++i)
      assert(fixture.Append(1000000 + uint64_t(i) * 33333, sample, 0));
    fixture.Stop();
    assert(fixture.Peek(packet, 3) == 0);
    assert(fixture.Keep());
    assert(fixture.Peek(packet, 3) == 3);
    FILE* file = std::fopen(argv[1], "wb");
    assert(file);
    assert(std::fwrite(packet, 1, 3 * kRecordBytes, file) == 3 * kRecordBytes);
    assert(std::fclose(file) == 0);
  }

  std::puts("PASS: local-first review/keep/discard, ACK guards, full buffer, wire fixture.");
  return 0;
}
