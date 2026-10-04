import json,tempfile,unittest
from pathlib import Path
from power_measurements import summarize
class Measurements(unittest.TestCase):
    def test_interval_and_external_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a={k:0 for k in ('samples','gaps','trigger_n','wake_n','full_n','queue_drops','trigger_deadline_misses','full_deadline_misses','sensor_us','trigger_us','full_us','active_us','display_us','wifi_us')}
            a.update(uptime_us=1_000_000,full_arena_bytes=80000,trigger_arena_bytes=4000,free_internal=1234,free_psram=5000)
            b=dict(a,uptime_us=11_000_000,samples=300,trigger_n=40,trigger_us=80000,full_n=2,full_us=100000,active_us=1000000)
            log=root/'serial.log';log.write_text('\n'.join('POWER_JSON '+json.dumps(v) for v in (a,b)))
            csv=root/'current.csv';csv.write_text('timestamp_s,current_ma\n0,100\n10,100\n')
            r=summarize(log,csv,200);self.assertEqual(r['counts']['samples'],300);self.assertEqual(r['mean_latency_us']['trigger'],2000)
            self.assertAlmostEqual(r['energy']['measured_mah'],1000/3600);self.assertAlmostEqual(r['energy']['battery_hours'],2);self.assertIsNone(r['false_wakeups_per_hour'])
    def test_no_current_is_not_zero_current(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'log';p.write_text('POWER_JSON {}')
            with self.assertRaises(ValueError):summarize(p)
if __name__=='__main__':unittest.main()
