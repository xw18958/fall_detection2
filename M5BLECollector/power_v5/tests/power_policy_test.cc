#include "../firmware/main/power_policy.h"
#include <cassert>
#include <cstring>
#include <iostream>

int main() {
  using namespace fall_power;
  // One hour ends exactly on the rational sample and quarter-second grids.
  assert(SampleDeadline(123, 108000) == 3600000123LL);
  assert(StepDeadline(123, 14400) == 3600000123LL);
  uint64_t previous = 0;
  FallbackCadence fallback(123);
  unsigned full_evaluations=0;
  for (uint64_t step = 1; step <= 14400; ++step) {
    const uint64_t samples = step * 30 / 4;
    assert(samples - previous == (step % 2 ? 7 : 8));
    assert(SampleDeadline(0, samples) <= StepDeadline(0, step));
    assert(SampleDeadline(0, samples + 1) > StepDeadline(0, step));
    previous = samples;
    full_evaluations+=fallback.Step(StepDeadline(123,step));
  }
  assert(full_evaluations==4800); // Exactly 4/3 Hz for a full hour.
  FallbackCadence recovery(0);
  assert(recovery.Step(3000000)); // First complete 3-second history.
  assert(!recovery.Step(3250000));
  assert(recovery.Step(3750000));
  assert(recovery.Step(5000000)); // Skipped grid deadlines do not backlog replay.
  Policy p;
  assert(!p.Step(0, false).run_full);
  auto d = p.Step(250000, true);
  assert(d.woke && d.run_full); // W0 runs immediately.
  for (int i = 2; i <= 4; ++i) assert(p.Step(i * kStrideUs, false).run_full);
  assert(!p.Step(1250000, false).run_full);
  p.Reset();
  unsigned wakes = 0, caps = 0;
  for (int i = 0; i < 100; ++i) {
    d = p.Step(i * kStrideUs, true);
    assert(d.run_full); wakes += d.woke; caps += d.capped;
  }
  assert(wakes == 1 && caps == 3); // Persistent suspicion has no blind interval.
  p.Reset(); assert(!p.active());
  assert(p.Step(25000000, true).woke);
  Window ring{}, queued{};
  ring.x[0][0] = 3.25f; ring.last_sample = 90; ring.generation = 4;
  queued = ring; // The queue owns a value copy, not the live rolling buffer.
  ring.x[0][0] = -99; ++ring.last_sample; ++ring.generation;
  assert(queued.x[0][0] == 3.25f && queued.last_sample == 90 && queued.generation == 4);
  std::cout << "PASS rational cadence, immediate W0, quiet timeout, persistent episodes, frozen queue ownership\n";
}
