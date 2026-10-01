#include "../firmware/main/recording_buffer.h"
#include <cassert>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
  using namespace m5ble;
  uint8_t storage[3*kRecordBytes], batch[3*kRecordBytes];
  RecordingBuffer buffer(storage,3);
  int16_t raw[6]={-32768,32767,-123,123,0,-1};
  assert(buffer.CanLeave()); assert(!buffer.Start(0)); assert(buffer.Start(0x1234));
  assert(!buffer.CanLeave());
  assert(buffer.Append(1000000,raw,Marker,42));
  assert(buffer.Append(1033333,raw,ReadError));
  assert(buffer.Peek(batch,3)==2);
  assert(Get32(batch)==0 && Get64(batch+4)==1000000);
  assert(Get16(batch+12)==32768 && Get16(batch+14)==32767 && Get16(batch+26)==42);
  assert(!buffer.Ack(0x1234,1)); // Never release samples that were not fully transmitted.
  buffer.Offered(2);
  assert(!buffer.Ack(0x5678,2)); assert(!buffer.Ack(0x1234,3));
  assert(buffer.Ack(0x1234,1)); assert(buffer.Pending()==1);
  assert(buffer.Append(1066666,raw,0)); assert(buffer.Append(1100000,raw,0)); // Ring wraps.
  assert(!buffer.Append(1133333,raw,0)); assert(buffer.State()==RecordState::Full);
  assert(buffer.Overflowed()); assert(!buffer.CanLeave()); assert(!buffer.Start(0x5678));
  assert(buffer.Peek(batch,3)==3 && Get32(batch)==1 && Get32(batch+64)==3);
  buffer.Offered(4); assert(buffer.Ack(0x1234,4));
  assert(buffer.State()==RecordState::Complete && buffer.CanLeave());
  assert(buffer.Ack(0x1234,4)); // Duplicate ACK.
  assert(buffer.Start(0x5678)); assert(!buffer.Ack(0x1234,0));
  assert(buffer.Append(2000000,raw,TimingGap)); buffer.Stop();
  assert(buffer.State()==RecordState::Saving); assert(!buffer.Append(2033333,raw,0));
  buffer.Offered(1); assert(buffer.Ack(0x5678,1)); assert(buffer.CanLeave());
  assert(buffer.Start(0x6789)); buffer.Stop(); assert(buffer.State()==RecordState::Complete);
  if (argc == 2) {
    uint8_t fixture_storage[3*kRecordBytes];
    RecordingBuffer fixture(fixture_storage,3); assert(fixture.Start(42));
    for (uint32_t i=0;i<3;++i) assert(fixture.Append(1000000+uint64_t(i)*33333,raw,0));
    assert(fixture.Peek(batch,3)==3);
    FILE* file=std::fopen(argv[1],"wb"); assert(file);
    assert(std::fwrite(batch,1,sizeof(batch),file)==sizeof(batch)); assert(std::fclose(file)==0);
  }
  std::puts("Buffer overflow, wraparound, stale/future ACKs, reconnect replay and mode-exit guards passed.");
}
