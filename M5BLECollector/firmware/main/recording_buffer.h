#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace m5ble {
constexpr size_t kRecordBytes = 32;
constexpr size_t kMaxBatchRecords = 6;
enum class RecordState : uint8_t { Ready = 0, Recording = 1, Saving = 2, Complete = 3, Full = 4 };
enum SampleFlags : uint16_t { ReadError = 1, TimingGap = 2, Saturated = 4, Marker = 8 };

inline void Put16(uint8_t* p, uint16_t v) { p[0] = v; p[1] = v >> 8; }
inline void Put32(uint8_t* p, uint32_t v) { for (int i=0;i<4;++i) p[i]=v>>(8*i); }
inline void Put64(uint8_t* p, uint64_t v) { for (int i=0;i<8;++i) p[i]=v>>(8*i); }
inline uint16_t Get16(const uint8_t* p) { return p[0] | (uint16_t(p[1]) << 8); }
inline uint32_t Get32(const uint8_t* p) { uint32_t v=0; for(int i=0;i<4;++i) v |= uint32_t(p[i])<<(8*i); return v; }
inline uint64_t Get64(const uint8_t* p) { uint64_t v=0; for(int i=0;i<8;++i) v |= uint64_t(p[i])<<(8*i); return v; }

// Synchronization is supplied by the caller. No allocation, I/O, or model state.
class RecordingBuffer {
 public:
  RecordingBuffer(uint8_t* storage, size_t records) : storage_(storage), capacity_(records) {}
  bool Start(uint64_t session) {
    if (!session || state_ == RecordState::Recording || count_ || !capacity_) return false;
    session_ = session; head_ = count_ = 0; next_ = offered_ = 0;
    state_ = RecordState::Recording; overflow_ = false; return true;
  }
  void Stop() {
    if (state_ == RecordState::Recording) state_ = count_ ? RecordState::Saving : RecordState::Complete;
  }
  bool Append(uint64_t time_us, const int16_t raw[6], uint16_t flags, uint16_t marker = 0) {
    if (state_ != RecordState::Recording) return false;
    if (count_ == capacity_ || next_ == UINT32_MAX) {
      state_ = RecordState::Full; overflow_ = true; return false;
    }
    uint8_t* p = storage_ + ((head_ + count_) % capacity_) * kRecordBytes;
    std::memset(p, 0, kRecordBytes);
    Put32(p, next_++); Put64(p+4, time_us);
    for (int i=0;i<6;++i) Put16(p+12+2*i, static_cast<uint16_t>(raw[i]));
    Put16(p+24, flags); Put16(p+26, marker); ++count_; return true;
  }
  size_t Peek(uint8_t* output, size_t limit) const {
    size_t n = count_ < limit ? count_ : limit;
    for (size_t i=0;i<n;++i)
      std::memcpy(output+i*kRecordBytes, storage_+((head_+i)%capacity_)*kRecordBytes, kRecordBytes);
    return n;
  }
  // Exclusive high-watermark: only fully offered records can be acknowledged.
  void Offered(uint32_t exclusive) { if (exclusive <= next_ && exclusive > offered_) offered_ = exclusive; }
  bool Ack(uint64_t session, uint32_t exclusive) {
    if (session != session_ || exclusive > offered_) return false;
    while (count_ && Get32(storage_+head_*kRecordBytes) < exclusive) {
      head_ = (head_ + 1) % capacity_; --count_;
    }
    if (!count_ && (state_ == RecordState::Saving || state_ == RecordState::Full)) state_ = RecordState::Complete;
    return true;
  }
  bool CanLeave() const { return state_ != RecordState::Recording && count_ == 0; }
  uint64_t Session() const { return session_; }
  uint32_t Next() const { return next_; }
  size_t Pending() const { return count_; }
  RecordState State() const { return state_; }
  bool Overflowed() const { return overflow_; }
 private:
  uint8_t* storage_; size_t capacity_, head_=0, count_=0;
  uint64_t session_=0; uint32_t next_=0, offered_=0;
  RecordState state_=RecordState::Ready; bool overflow_=false;
};
}  // namespace m5ble
