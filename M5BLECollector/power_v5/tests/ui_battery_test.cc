#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include "../firmware/main/device_ui.h"
#include "../firmware/main/battery_level.h"

int test_data_mode = 0;
static std::array<uint16_t, 135*240> pixels{};
static int command=0, window_x_first=0, window_x_last=0, window_y_first=0, window_y_last=0, cursor=0;
static size_t transfers=0;
int test_transmit(const uint8_t* data, size_t n) {
  ++transfers;
  if (!test_data_mode) { assert(n==1); command=data[0]; if(command==0x2C) cursor=0; return 0; }
  if(command==0x2A || command==0x2B) {
    assert(n==4);
    int start=(int(data[0])<<8)|data[1], end=(int(data[2])<<8)|data[3];
    if(command==0x2A) { window_x_first=start-52; window_x_last=end-52; assert(window_x_first>=0 && window_x_last<135 && window_x_first<=window_x_last); }
    else { window_y_first=start-40; window_y_last=end-40; assert(window_y_first>=0 && window_y_last<240 && window_y_first<=window_y_last); }
  } else if(command==0x2C) {
    assert(n%2==0);
    for(size_t i=0;i<n;i+=2) {
      const int x=window_x_first+cursor%(window_x_last-window_x_first+1), y=window_y_first+cursor/(window_x_last-window_x_first+1);
      assert(y<=window_y_last); pixels[size_t(y)*135+x]=uint16_t(data[i])<<8|data[i+1]; ++cursor;
    }
  }
  return 0;
}
void Reset() {
  pixels.fill(0); transfers=0;
  fall_display::g_ready=true; fall_display::g_spi=reinterpret_cast<void*>(1);
  fall_display::g_has_result=false;
  device_ui::g_screen=-1; device_ui::g_battery=-999; device_ui::g_progress=-1;
  device_ui::g_presentation=device_ui::CollectionPresentation{};
  for(auto& t:device_ui::g_text) t=device_ui::TextCache{};
}
bool HasText(const char* value) {
  for(const auto& t:device_ui::g_text) if(t.valid && std::string(t.text)==value) return true;
  return false;
}
void CheckLayout() {
  for(int a=0;a<16;++a) {
    const auto& t=device_ui::g_text[a]; if(!t.valid) continue;
    const int w=device_ui::TextWidth(t.text,t.scale), h=8*t.scale;
    assert(t.scale>=2 && t.x>=0 && t.y>=0 && t.x+w<=135 && t.y+h<=240);
    for(int b=a+1;b<16;++b) {
      const auto& other=device_ui::g_text[b]; if(!other.valid) continue;
      assert(t.x+w<=other.x || other.x+device_ui::TextWidth(other.text,other.scale)<=t.x ||
             t.y+h<=other.y || other.y+8*other.scale<=t.y);
    }
  }
}
void Save(const std::filesystem::path& directory, const char* name) {
  CheckLayout();
  if(directory.empty()) return;
  std::ofstream file(directory/(std::string(name)+".ppm"),std::ios::binary);
  file << "P6\n135 240\n255\n";
  for(uint16_t color:pixels) {
    const char rgb[3]={char(((color>>11)&31)*255/31),char(((color>>5)&63)*255/63),char((color&31)*255/31)};
    file.write(rgb,3);
  }
}
void TestBattery() {
  using m5_battery::PercentageFromMillivolts;
  assert(PercentageFromMillivolts(-1)==-1 && PercentageFromMillivolts(0)==-1);
  assert(PercentageFromMillivolts(3000)==0 && PercentageFromMillivolts(3300)==0);
  assert(PercentageFromMillivolts(3380)==10 && PercentageFromMillivolts(3460)==20);
  assert(PercentageFromMillivolts(3700)==50 && PercentageFromMillivolts(4100)==100);
  assert(PercentageFromMillivolts(4200)==100 && PercentageFromMillivolts(2147483647)==100);
  int previous=0;
  for(int mv=3000;mv<=4500;++mv) { int pct=PercentageFromMillivolts(mv); assert(pct>=previous && pct<=100); previous=pct; }
}
void TestPresentation() {
  using m5ble::RecordState;
  device_ui::CollectionPresentation ui;
  m5ble::Status s; s.session=42; s.state=RecordState::Complete; s.samples=552;
  assert(ui.Observe(s,1000000).screen==RecordState::Complete);
  assert(ui.Observe(s,2799999).screen==RecordState::Complete);
  s.connected=true; s.ready=true;
  assert(ui.Observe(s,2800000).screen==RecordState::Ready);
  assert(s.state==RecordState::Complete && s.samples==552 && s.session==42);
  assert(ui.Observe(s,60000000).screen==RecordState::Ready);
  s.state=RecordState::Recording; s.session=43;
  assert(ui.Observe(s,60000001).screen==RecordState::Recording);
  s.state=RecordState::Complete;
  assert(ui.Observe(s,60000002).screen==RecordState::Complete);
  s.last_error=1;
  assert(ui.Observe(s,61000000).error);
  assert(!ui.Observe(s,62800000).error); // Sticky transport flag cannot cover UI forever.
  s.last_error=0; assert(!ui.Observe(s,63000000).error);
  s.last_error=1; assert(ui.Observe(s,64000000).error);
  s.samples=552; s.pending=155;
  assert(device_ui::SavedCount(s)==397 && device_ui::ProgressPercent(s)==71);
  s.pending=999; assert(device_ui::ProgressPercent(s)==0);
  s.samples=0; assert(device_ui::ProgressPercent(s)==100);
  char duration[16]; device_ui::FormatDuration(552,duration); assert(std::string(duration)=="00:18.4");
  device_ui::FormatDuration(8192,duration); assert(std::string(duration)=="04:33.0");
}
int main(int argc,char** argv) {
  std::filesystem::path output;
  if(argc==2) { output=argv[1]; std::filesystem::create_directories(output); }
  TestBattery(); TestPresentation();
  using m5ble::RecordState;
  m5ble::Status s;
  Reset(); device_ui::DrawStarting(false); Save(output,"01-starting");
  Reset(); fall_display::ShowResult(0.023f,false,450); device_ui::DrawDetectScreen(82); Save(output,"02-detect");
  const auto unchanged=transfers; device_ui::DrawDetectScreen(82); assert(transfers==unchanged);
  fall_display::ShowResult(1.0f,true,450,3000000); device_ui::DrawDetectScreen(10); Save(output,"03-fall-alert");
  Reset(); device_ui::DrawDetectScreen(-1); Save(output,"04-detect-warming");
  for(int mode=0;mode<3;++mode) {
    Reset(); s={}; s.ready=mode==2; s.connected=mode>0;
    device_ui::DrawCollectionScreen(s,82,1000000);
    assert(HasText(mode==2?"READY":mode==1?"CONNECTING":"OFFLINE"));
    Save(output,mode==2?"07-ready":mode==1?"06-connecting":"05-offline");
  }
  Reset(); s={}; s.state=RecordState::Recording; s.samples=552; s.session=42;
  device_ui::DrawCollectionScreen(s,81,1000000); assert(HasText("00:18.4") && !HasText("MAC")); Save(output,"08-recording");
  auto stable_pixels=pixels; auto stable_transfers=transfers;
  device_ui::DrawCollectionScreen(s,81,1100000); assert(transfers==stable_transfers);
  s.samples+=3; device_ui::DrawCollectionScreen(s,81,1250000);
  assert(transfers-stable_transfers<=6); // One changed tenths glyph, no whole-screen redraw.
  assert(pixels!=stable_pixels); CheckLayout();
  s.state=RecordState::Review; s.samples=552;
  device_ui::DrawCollectionScreen(s,80,1300000); assert(HasText("KEEP")&&HasText("DISCARD")); Save(output,"09-review");
  s.state=RecordState::Full; s.samples=8192; s.overflow=true;
  device_ui::DrawCollectionScreen(s,20,1400000); assert(HasText("04:33.0")&&HasText("8192")); Save(output,"10-buffer-full");
  s.state=RecordState::Saving; s.samples=552; s.pending=155; s.ready=true;
  device_ui::DrawCollectionScreen(s,77,1500000); assert(HasText("397/552")); Save(output,"11-saving");
  s.samples=8192; s.pending=0;
  device_ui::DrawCollectionScreen(s,77,1600000); assert(HasText("100%")&&HasText("8192/8192")); Save(output,"12-saving-max");
  s.ready=false; s.pending=8192;
  device_ui::DrawCollectionScreen(s,-1,1700000); assert(HasText("SAFE")&&HasText("8192")); Save(output,"13-waiting");
  s.state=RecordState::Complete; s.samples=552; s.pending=0;
  device_ui::DrawCollectionScreen(s,76,2000000); assert(HasText("SAVED")); Save(output,"14-saved");
  device_ui::DrawCollectionScreen(s,76,3800000); assert(HasText("START") && s.state==RecordState::Complete); Save(output,"15-ready-after-saved");
  s.last_error=1; device_ui::DrawCollectionScreen(s,76,4000000); assert(HasText("FAILED")); Save(output,"16-error");
  device_ui::DrawCollectionScreen(s,76,5800000); assert(HasText("START") && !HasText("FAILED")); Save(output,"17-error-cleared");
  Reset(); s={}; s.state=RecordState::Saving; s.samples=8192; s.pending=8192; s.ready=true;
  device_ui::DrawCollectionScreen(s,0,1000000); assert(HasText("0%")); Save(output,"18-saving-zero");
  puts("PASS: battery mapping, SAVED/error timing, immutable collector state, text bounds/overlap, actual ST7789 framebuffer and incremental redraw.");
}
