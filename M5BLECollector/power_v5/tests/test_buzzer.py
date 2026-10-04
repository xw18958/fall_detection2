"""Execute the actual buzzer implementation against fault-injecting IDF shims."""
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIM = r'''#include <cassert>
#include <cstdint>
#include <string>
#include <algorithm>
using esp_err_t = int;
constexpr int ESP_OK=0, ESP_ERR_NO_MEM=1, ESP_ERR_INVALID_STATE=2, ESP_ERR_TIMEOUT=3;
constexpr int GPIO_NUM_2=2, GPIO_MODE_OUTPUT=1;
constexpr int LEDC_LOW_SPEED_MODE=0, LEDC_TIMER_0=0, LEDC_CHANNEL_0=0;
constexpr int LEDC_TIMER_8_BIT=8, LEDC_USE_REF_TICK=1, ESP_PM_NO_LIGHT_SLEEP=1;
using gpio_num_t=int;
using TickType_t=unsigned;
using SemaphoreHandle_t=void*;
using esp_pm_lock_handle_t=void*;
constexpr unsigned portMAX_DELAY=~0u;
constexpr int pdTRUE=1;
#define pdMS_TO_TICKS(x) (static_cast<unsigned>(x))
struct gpio_config_t {uint64_t pin_bit_mask; int mode;};
struct ledc_timer_config_t {int speed_mode,timer_num,duty_resolution,freq_hz,clk_cfg;};
struct ledc_channel_config_t {int gpio_num,speed_mode,channel,timer_sel,duty;};
std::string fail;
bool output=false, paused=true, held=false;
int locks=0, delays=0, duty=0;
int op(const char* name) {return fail==name ? 10 : ESP_OK;}
int gpio_config(const gpio_config_t* c) {assert(c->pin_bit_mask==(1ULL<<2));return op("gpio");}
int gpio_set_level(int pin,int level) {assert(pin==2 && level==0);return ESP_OK;}
void* xSemaphoreCreateMutex() {return reinterpret_cast<void*>(1);}
void vSemaphoreDelete(void*) {assert(!held);}
int xSemaphoreTake(void*,unsigned) {assert(!held);held=true;return pdTRUE;}
int xSemaphoreGive(void*) {assert(held);held=false;return pdTRUE;}
int esp_pm_lock_create(int type,int,const char*,void** p) {
 assert(type==ESP_PM_NO_LIGHT_SLEEP);int e=op("create");if(!e)*p=reinterpret_cast<void*>(2);return e;
}
int esp_pm_lock_delete(void*) {assert(locks==0);return ESP_OK;}
int esp_pm_lock_acquire(void*) {int e=op("acquire");if(!e)++locks;return e;}
int esp_pm_lock_release(void*) {assert(locks==1);--locks;return ESP_OK;}
int ledc_timer_config(const ledc_timer_config_t* t) {
 assert(t->clk_cfg==LEDC_USE_REF_TICK && t->freq_hz==2000);
 // Catch the original invalid 1 MHz / (2 kHz * 1024) divider.
 assert(1000000.0/(t->freq_hz*(1u<<t->duty_resolution))>=1.0);
 assert(t->duty_resolution==8);return op("timer");
}
int ledc_channel_config(const ledc_channel_config_t* c) {assert(c->duty==0);return op("channel");}
int ledc_stop(int,int,int idle) {assert(idle==0);output=false;duty=0;return ESP_OK;}
int ledc_timer_pause(int,int) {paused=true;return ESP_OK;}
int ledc_timer_resume(int,int) {assert(locks==1);int e=op("resume");if(!e)paused=false;return e;}
int ledc_set_duty(int,int,int d) {assert(d==128);int e=op("duty");if(!e)duty=d;return e;}
int ledc_update_duty(int,int) {int e=op("update");if(!e)output=duty>0;return e;}
void vTaskDelay(unsigned ticks) {assert(ticks>0 && output && !paused && locks==1 && held);++delays;}
'''
MAIN = r'''
int main(int argc,char** argv) {
 const std::string scenario=argc>1 ? argv[1] : "nominal";
 if(scenario.rfind("init:",0)==0) {
  fail=scenario.substr(5);assert(fall_buzzer::Init()!=ESP_OK);
  assert(fall_buzzer::Beep(100)==ESP_ERR_INVALID_STATE);
 } else {
  assert(fall_buzzer::Init()==ESP_OK);
  assert(!output && paused && locks==0);
  if(scenario.rfind("beep:",0)==0) {
   fail=scenario.substr(5);assert(fall_buzzer::Beep(100)!=ESP_OK);
  } else {
   assert(fall_buzzer::Beep(100)==ESP_OK);
   assert(fall_buzzer::Beep(30)==ESP_OK);
   assert(fall_buzzer::Beep(0)==ESP_OK);
   assert(fall_buzzer::Beep(-1)==ESP_OK);
   assert(delays==2);
  }
 }
 assert(!output && paused && locks==0 && !held);
}
'''

class BuzzerTest(unittest.TestCase):
    def test_pwm_and_idle_cleanup_in_normal_and_fault_paths(self):
        body = (ROOT / "firmware/main/buzzer.cc").read_text()
        body = "\n".join(line for line in body.splitlines()
                         if not line.startswith("#include"))
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "buzzer_test.cc"
            binary = Path(folder) / "buzzer_test"
            source.write_text(SHIM + body + MAIN)
            subprocess.run(["clang++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", str(source), "-o", str(binary)], check=True)
            for scenario in ["nominal", "init:gpio", "init:create", "init:timer", "init:channel",
                             "beep:acquire", "beep:resume", "beep:duty", "beep:update"]:
                with self.subTest(scenario=scenario):
                    subprocess.run([str(binary), scenario], check=True)

if __name__ == "__main__":
    unittest.main()
