#include "ble_collector.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <new>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_bt.h"
#include "esp_mac.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "model_identity.h"
#include "nvs.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"
#if __has_include("ble_transport_config.local.h")
#include "ble_transport_config.local.h"
#endif
#ifndef M5BLE_UNPAIRED_TRANSPORT
#define M5BLE_UNPAIRED_TRANSPORT 0
#endif

extern "C" void ble_store_config_init(void);

namespace m5ble {
namespace {
constexpr char kTag[] = "m5ble";
constexpr bool kEncryptedTransport = !M5BLE_UNPAIRED_TRANSPORT;
// UUID: 6f4d0001-8f2b-4e3d-9a11-189580000001; characteristics end in 2..5.
#define M5UUID(n) BLE_UUID128_INIT(n,0,0,0x80,0x95,0x18,0x11,0x9a,0x3d,0x4e,0x2b,0x8f,1,0,0x4d,0x6f)
const ble_uuid128_t kService=M5UUID(1), kInfo=M5UUID(2), kControl=M5UUID(3), kSamples=M5UUID(4), kStatus=M5UUID(5);
uint16_t g_sample_handle=0, g_status_handle=0;
uint8_t g_addr_type=0;
std::atomic<uint16_t> g_conn{BLE_HS_CONN_HANDLE_NONE};
std::atomic<bool> g_subscribed{false}, g_ready{false}, g_detect{false}, g_status_dirty{true};
std::atomic<uint8_t> g_error{0};
portMUX_TYPE g_lock=portMUX_INITIALIZER_UNLOCKED;
RecordingBuffer* g_buffer=nullptr;
uint16_t g_marker=0;
uint64_t g_boot=0;
char g_device_id[18]{};
struct Command { uint8_t op; uint64_t session; uint32_t exclusive; uint16_t marker; };
QueueHandle_t g_commands=nullptr;
uint8_t g_packet[kRecordBytes*kMaxBatchRecords]{};
size_t g_packet_len=0, g_part=0, g_parts=0, g_chunk=0;
uint64_t g_packet_session=0;
uint32_t g_batch=0, g_exclusive=0;
int64_t g_offered_at=0, g_next_send=0, g_status_at=0;
ble_addr_t g_owner{}; bool g_has_owner=false;

void ClearPacket() { g_packet_len=g_part=g_parts=g_chunk=0; g_offered_at=0; }
uint64_t RandomSession() { uint64_t v; do { v=(uint64_t(esp_random())<<32)|esp_random(); } while(!v); return v; }

void StatusBytes(uint8_t out[20]) {
  auto s=GetStatus(); out[0]=1; out[1]=1; out[2]=static_cast<uint8_t>(s.state);
  out[3]=(s.connected?1:0)|(s.ready?2:0)|(s.overflow?4:0)|(s.last_error?8:0);
  Put64(out+4,s.session); Put32(out+12,s.samples); Put32(out+16,s.pending);
}

bool ConnectionAllowed(uint16_t conn) {
  if (!kEncryptedTransport) return conn != BLE_HS_CONN_HANDLE_NONE && conn == g_conn;
  ble_gap_conn_desc d{}; return ble_gap_conn_find(conn,&d)==0 && d.sec_state.encrypted;
}

int Access(uint16_t conn, uint16_t, ble_gatt_access_ctxt* ctx, void* arg) {
  const auto which=reinterpret_cast<uintptr_t>(arg);
  if (!ConnectionAllowed(conn)) return BLE_ATT_ERR_INSUFFICIENT_AUTHEN;
  if (ctx->op==BLE_GATT_ACCESS_OP_READ_CHR && which==2) {
    char info[512];
    int n=std::snprintf(info,sizeof(info),
      "{\"protocol\":1,\"device_id\":\"%s\",\"boot_id\":\"%016llx\",\"firmware\":\"%s\","
      "\"model_sha256\":\"%s\",\"rate_hz\":30,\"accel_range_g\":8,\"gyro_range_dps\":2000,"
      "\"accel_g_per_lsb\":0.000244140625,\"gyro_dps_per_lsb\":0.06103515625,"
      "\"buffer_records\":8192,\"time_us\":%lld,\"last_error\":%u,\"transport\":\"%s\"}",
      g_device_id,static_cast<unsigned long long>(g_boot),kFirmwareVersion,kModelSha256,
      static_cast<long long>(esp_timer_get_time()),unsigned(g_error.load()),kEncryptedTransport?"ble_encrypted":"ble_unpaired");
    return n>0 && n<int(sizeof(info)) && os_mbuf_append(ctx->om,info,n)==0 ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
  }
  if (ctx->op==BLE_GATT_ACCESS_OP_READ_CHR && which==5) {
    uint8_t bytes[20]; StatusBytes(bytes);
    return os_mbuf_append(ctx->om,bytes,sizeof(bytes))==0?0:BLE_ATT_ERR_INSUFFICIENT_RES;
  }
  if (ctx->op==BLE_GATT_ACCESS_OP_WRITE_CHR && which==3) {
    uint8_t bytes[16]; uint16_t n=0;
    if (OS_MBUF_PKTLEN(ctx->om)!=16 || ble_hs_mbuf_to_flat(ctx->om,bytes,16,&n)!=0 || bytes[0]!=1)
      return BLE_ATT_ERR_INVALID_ATTR_VALUE_LEN;
    if (bytes[1]<1 || bytes[1]>6) return BLE_ATT_ERR_VALUE_NOT_ALLOWED;
    Command cmd{bytes[1],Get64(bytes+2),Get32(bytes+10),Get16(bytes+14)};
    return xQueueSend(g_commands,&cmd,0)==pdTRUE?0:BLE_ATT_ERR_INSUFFICIENT_RES;
  }
  return BLE_ATT_ERR_READ_NOT_PERMITTED;
}

ble_gatt_chr_def g_characteristics[5]{};
ble_gatt_svc_def g_services[2]{};
int GapEvent(ble_gap_event*, void*);

void Advertise() {
  ble_hs_adv_fields fields{};
  fields.flags=BLE_HS_ADV_F_DISC_GEN|BLE_HS_ADV_F_BREDR_UNSUP;
  fields.uuids128=const_cast<ble_uuid128_t*>(&kService); fields.num_uuids128=1; fields.uuids128_is_complete=1;
  if (ble_gap_adv_set_fields(&fields)!=0) return;
  ble_hs_adv_fields response{};
  auto name=ble_svc_gap_device_name(); response.name=reinterpret_cast<const uint8_t*>(name);
  response.name_len=std::strlen(name); response.name_is_complete=1;
  if (ble_gap_adv_rsp_set_fields(&response)!=0) return;
  ble_gap_adv_params params{}; params.conn_mode=BLE_GAP_CONN_MODE_UND; params.disc_mode=BLE_GAP_DISC_MODE_GEN;
  int rc=ble_gap_adv_start(g_addr_type,nullptr,BLE_HS_FOREVER,&params,GapEvent,nullptr);
  if(rc) ESP_LOGW(kTag,"Advertising failed: %d",rc);
}

int GapEvent(ble_gap_event* event, void*) {
  switch(event->type) {
    case BLE_GAP_EVENT_CONNECT:
      ESP_LOGI(kTag,"Connection result=%d handle=%u",event->connect.status,unsigned(event->connect.conn_handle));
      if(event->connect.status==0) {
        g_conn=event->connect.conn_handle; g_ready=false;
        // The Mac starts encryption when reading the encrypted information
        // characteristic. Do not start a second security procedure here.
        if(!kEncryptedTransport) {
          ble_gap_upd_params params{}; params.itvl_min=24; params.itvl_max=36;
          params.latency=0; params.supervision_timeout=600;
          ble_gap_update_params(g_conn,&params);
        }
      } else Advertise();
      break;
    case BLE_GAP_EVENT_ENC_CHANGE: {
      if(!kEncryptedTransport) break;
      ble_gap_conn_desc desc{};
      ESP_LOGI(kTag,"Encryption result=%d handle=%u",event->enc_change.status,unsigned(event->enc_change.conn_handle));
      if(event->enc_change.status || ble_gap_conn_find(event->enc_change.conn_handle,&desc)!=0 || !desc.sec_state.encrypted) {
        ble_gap_terminate(event->enc_change.conn_handle,BLE_ERR_REM_USER_CONN_TERM); break;
      }
      // Bond to the first central used in collection mode; future centrals must match.
      if(g_has_owner && ble_addr_cmp(&g_owner,&desc.peer_id_addr)!=0) {
        ESP_LOGW(kTag,"Owner identity mismatch: stored=%u/%02x%02x%02x%02x%02x%02x peer=%u/%02x%02x%02x%02x%02x%02x",g_owner.type,g_owner.val[5],g_owner.val[4],g_owner.val[3],g_owner.val[2],g_owner.val[1],g_owner.val[0],desc.peer_id_addr.type,desc.peer_id_addr.val[5],desc.peer_id_addr.val[4],desc.peer_id_addr.val[3],desc.peer_id_addr.val[2],desc.peer_id_addr.val[1],desc.peer_id_addr.val[0]);
        ble_gap_terminate(desc.conn_handle,BLE_ERR_REM_USER_CONN_TERM); break;
      }
      if(!g_has_owner) {
        nvs_handle_t h;
        if(nvs_open("m5ble",NVS_READWRITE,&h)!=ESP_OK) { ble_gap_terminate(desc.conn_handle,BLE_ERR_REM_USER_CONN_TERM); break; }
        esp_err_t err=nvs_set_blob(h,"owner",&desc.peer_id_addr,sizeof(desc.peer_id_addr));
        if(err==ESP_OK) err=nvs_commit(h);
        nvs_close(h);
        if(err!=ESP_OK) { ble_gap_terminate(desc.conn_handle,BLE_ERR_REM_USER_CONN_TERM); break; }
        g_owner=desc.peer_id_addr; g_has_owner=true;
      }
      ble_gap_upd_params params{};
      params.itvl_min=24; params.itvl_max=36; // 30--45 ms, compatible with macOS.
      params.latency=0; params.supervision_timeout=600;
      ESP_LOGI(kTag,"Connection parameter request=%d",ble_gap_update_params(desc.conn_handle,&params));
      // Use legacy-sized encrypted link packets while diagnosing the Mac's
      // observed MIC failures. This changes transport only, not sensor records.
      ESP_LOGI(kTag,"Conservative data-length request=%d",ble_gap_set_data_len(desc.conn_handle,27,328));
      break;
    }
    case BLE_GAP_EVENT_CONN_UPDATE: {
      ble_gap_conn_desc desc{};
      if(ble_gap_conn_find(event->conn_update.conn_handle,&desc)==0)
        ESP_LOGI(kTag,"Connection update result=%d interval=%u latency=%u timeout=%u",event->conn_update.status,unsigned(desc.conn_itvl),unsigned(desc.conn_latency),unsigned(desc.supervision_timeout));
      break;
    }
    case BLE_GAP_EVENT_DISCONNECT:
      ESP_LOGW(kTag,"Disconnect reason=%d",event->disconnect.reason);
      g_conn=BLE_HS_CONN_HANDLE_NONE; g_subscribed=false; g_ready=false; Advertise(); break;
    case BLE_GAP_EVENT_SUBSCRIBE:
      ESP_LOGI(kTag,"Subscription attr=%u notify=%u",unsigned(event->subscribe.attr_handle),unsigned(event->subscribe.cur_notify));
      if(event->subscribe.attr_handle==g_sample_handle) g_subscribed=event->subscribe.cur_notify;
      break;
    case BLE_GAP_EVENT_REPEAT_PAIRING:
      ESP_LOGW(kTag,"Repeat pairing requested");
      // Do not silently replace an existing bond. Document explicit bond reset.
      return BLE_GAP_REPEAT_PAIRING_IGNORE;
    case BLE_GAP_EVENT_ADV_COMPLETE: Advertise(); break;
    default: break;
  }
  g_status_dirty=true; return 0;
}
void Sync() {
  if(ble_hs_util_ensure_addr(0)!=0) return;
  if(!kEncryptedTransport) g_addr_type=BLE_OWN_ADDR_RANDOM;
  else if(ble_hs_id_infer_auto(0,&g_addr_type)!=0) return;
  Advertise();
}
void HostTask(void*) { nimble_port_run(); nimble_port_freertos_deinit(); }

void Process(const Command& cmd) {
  bool ok=false;
  switch(cmd.op) {
    case 1: { // start
      if(g_ready && g_subscribed) {
        portENTER_CRITICAL(&g_lock); ok=g_buffer->Start(RandomSession()); g_marker=0; portEXIT_CRITICAL(&g_lock);
        if(ok) ClearPacket();
      }
      break;
    }
    case 2: { // stop only the named session
      portENTER_CRITICAL(&g_lock);
      ok=cmd.session==g_buffer->Session(); if(ok) g_buffer->Stop();
      portEXIT_CRITICAL(&g_lock); break;
    }
    case 3: { // cumulative, exclusive acknowledgement after durable host write
      portENTER_CRITICAL(&g_lock); ok=g_buffer->Ack(cmd.session,cmd.exclusive); portEXIT_CRITICAL(&g_lock);
      if(ok && g_packet_len && cmd.session==g_packet_session && cmd.exclusive>=g_exclusive) ClearPacket();
      break;
    }
    case 4:
      portENTER_CRITICAL(&g_lock);
      ok=cmd.session==g_buffer->Session() && g_buffer->State()==RecordState::Recording && cmd.marker && !g_marker;
      if(ok) g_marker=cmd.marker;
      portEXIT_CRITICAL(&g_lock); break;
    case 5: g_ready=g_subscribed && ConnectionAllowed(g_conn); ok=g_ready; break;
    case 6: ok=CanLeave(); if(ok) g_detect=true; break;
  }
  g_error=ok?0:1; g_status_dirty=true;
}
}  // namespace

bool Init() {
  constexpr size_t kCapacity=8192;
  auto storage=static_cast<uint8_t*>(heap_caps_malloc(kCapacity*kRecordBytes,MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT));
  if(!storage) return false;
  g_buffer=new(std::nothrow) RecordingBuffer(storage,kCapacity);
  g_commands=xQueueCreate(16,sizeof(Command));
  if(!g_buffer || !g_commands) return false;
  g_boot=RandomSession(); uint8_t mac[6]; if(esp_read_mac(mac,ESP_MAC_BT)!=ESP_OK) return false;
  std::snprintf(g_device_id,sizeof(g_device_id),"%02x%02x%02x%02x%02x%02x",mac[0],mac[1],mac[2],mac[3],mac[4],mac[5]);
  nvs_handle_t h;
  if(nvs_open("m5ble",NVS_READWRITE,&h)!=ESP_OK) return false;
  size_t len=sizeof(g_owner); g_has_owner=nvs_get_blob(h,"owner",&g_owner,&len)==ESP_OK && len==sizeof(g_owner); nvs_close(h);
  if(nimble_port_init()!=ESP_OK) return false;
  if(!kEncryptedTransport) {
    // A distinct static address avoids the Mac automatically restoring the
    // existing encrypted bond. The physical device ID in metadata is unchanged.
    uint8_t address[6]; for(int i=0;i<6;++i) address[i]=mac[5-i];
    address[0]^=0x80; address[5]|=0xc0;
    if(ble_hs_id_set_rnd(address)!=0) return false;
  }
  // Keep the radio awake during field collection; detector mode never calls Init.
  ESP_LOGI(kTag,"Radio sleep disabled result=%d",int(esp_bt_sleep_disable()));
  esp_log_level_set("NimBLE",ESP_LOG_WARN);
  ble_svc_gap_init(); ble_svc_gatt_init();
  const ble_uuid_t* uuids[]={&kInfo.u,&kControl.u,&kSamples.u,&kStatus.u};
  uint16_t flags[]={uint16_t(BLE_GATT_CHR_F_READ|BLE_GATT_CHR_F_READ_ENC),
    uint16_t(BLE_GATT_CHR_F_WRITE|BLE_GATT_CHR_F_WRITE_ENC), BLE_GATT_CHR_F_NOTIFY,
    uint16_t(BLE_GATT_CHR_F_READ|BLE_GATT_CHR_F_READ_ENC|BLE_GATT_CHR_F_NOTIFY)};
  if(!kEncryptedTransport) {
    flags[0]=BLE_GATT_CHR_F_READ; flags[1]=BLE_GATT_CHR_F_WRITE;
    flags[3]=BLE_GATT_CHR_F_READ|BLE_GATT_CHR_F_NOTIFY;
  }
  for(int i=0;i<4;++i) {
    auto& c=g_characteristics[i]; c.uuid=uuids[i]; c.access_cb=Access;
    c.arg=reinterpret_cast<void*>(uintptr_t(i+2)); c.flags=flags[i];
  }
  g_characteristics[2].val_handle=&g_sample_handle; g_characteristics[3].val_handle=&g_status_handle;
  g_services[0].type=BLE_GATT_SVC_TYPE_PRIMARY; g_services[0].uuid=&kService.u; g_services[0].characteristics=g_characteristics;
  if(ble_gatts_count_cfg(g_services)!=0 || ble_gatts_add_svcs(g_services)!=0) return false;
  ble_hs_cfg.sync_cb=Sync; ble_hs_cfg.sm_io_cap=BLE_HS_IO_NO_INPUT_OUTPUT;
  ble_hs_cfg.sm_bonding=kEncryptedTransport; ble_hs_cfg.sm_sc=1; ble_hs_cfg.sm_mitm=0;
  ble_hs_cfg.sm_our_key_dist=BLE_SM_PAIR_KEY_DIST_ENC|BLE_SM_PAIR_KEY_DIST_ID;
  ble_hs_cfg.sm_their_key_dist=BLE_SM_PAIR_KEY_DIST_ENC|BLE_SM_PAIR_KEY_DIST_ID;
  ble_store_config_init(); ble_att_set_preferred_mtu(185);
  char name[24]; std::snprintf(name,sizeof(name),"M5Motion-%02X%02X",mac[4],mac[5]);
  if(ble_svc_gap_device_name_set(name)!=0) return false;
  nimble_port_freertos_init(HostTask);
  ESP_LOGI(kTag,"BLE collection ready, model remains embedded and inference is paused");
  return true;
}

Status GetStatus() {
  Status s{}; portENTER_CRITICAL(&g_lock);
  s.state=g_buffer->State(); s.session=g_buffer->Session(); s.samples=g_buffer->Next();
  s.pending=g_buffer->Pending(); s.overflow=g_buffer->Overflowed(); portEXIT_CRITICAL(&g_lock);
  s.connected=g_conn!=BLE_HS_CONN_HANDLE_NONE; s.ready=g_ready; s.last_error=g_error; return s;
}
bool CanLeave() { portENTER_CRITICAL(&g_lock); bool yes=g_buffer->CanLeave(); portEXIT_CRITICAL(&g_lock); return yes; }
bool DetectRequested() { return g_detect; }
bool ResetPairing() {
  if(!CanLeave()) return false;
  nvs_handle_t h;
  if(nvs_open("m5ble",NVS_READWRITE,&h)!=ESP_OK) return false;
  esp_err_t err=nvs_erase_key(h,"owner");
  if(err==ESP_ERR_NVS_NOT_FOUND) err=ESP_OK;
  if(err==ESP_OK) err=nvs_commit(h);
  nvs_close(h);
  // Only BLE bonds and this app's owner key are reset. Never erase shared NVS.
  return err==ESP_OK && ble_store_clear()==0;
}
bool ToggleRecording() {
  auto s=GetStatus();
  if(s.state==RecordState::Recording) { Process(Command{2,s.session,0,0}); return true; }
  Process(Command{1,0,0,0}); return !g_error;
}
void AddMarker(uint16_t marker) { auto s=GetStatus(); Process(Command{4,s.session,0,marker}); }
void Capture(uint64_t time_us,const int16_t raw[6],uint16_t flags) {
  portENTER_CRITICAL(&g_lock);
  bool appended=g_buffer->Append(time_us,raw,flags|(g_marker?SampleFlags::Marker:0),g_marker);
  if(appended) g_marker=0;
  portEXIT_CRITICAL(&g_lock);
}

void Tick() {
  Command cmd{}; while(xQueueReceive(g_commands,&cmd,0)==pdTRUE) Process(cmd);
  uint16_t conn=g_conn;
  if(conn==BLE_HS_CONN_HANDLE_NONE || !g_subscribed || !g_ready || !ConnectionAllowed(conn)) { ClearPacket(); return; }
  int64_t now=esp_timer_get_time();
  if(g_status_dirty || now-g_status_at>=1000000) {
    uint8_t bytes[20]; StatusBytes(bytes); auto mbuf=ble_hs_mbuf_from_flat(bytes,20);
    if(mbuf && ble_gatts_notify_custom(conn,g_status_handle,mbuf)==0) { g_status_at=now; g_status_dirty=false; }
  }
  if(g_offered_at && now-g_offered_at>2000000) { g_part=0; g_offered_at=0; }
  if(!g_packet_len) {
    if(now<g_next_send) return;
    portENTER_CRITICAL(&g_lock);
    g_packet_len=g_buffer->Peek(g_packet,kMaxBatchRecords)*kRecordBytes; g_packet_session=g_buffer->Session();
    portEXIT_CRITICAL(&g_lock);
    if(!g_packet_len) return;
    uint16_t mtu=ble_att_mtu(conn);
    g_chunk=std::min<size_t>(64,mtu>19?mtu-19:1);
    g_parts=(g_packet_len+g_chunk-1)/g_chunk; g_part=0; ++g_batch;
    g_exclusive=Get32(g_packet+g_packet_len-kRecordBytes)+1; g_next_send=now+100000;
  }
  // Send fragments until stack backpressure; retry the unsent fragment next tick.
  while(g_part<g_parts) {
    uint8_t bytes[256]; bytes[0]=1; bytes[1]=1; bytes[2]=g_part; bytes[3]=g_parts;
    Put64(bytes+4,g_packet_session); Put32(bytes+12,g_batch);
    size_t offset=g_part*g_chunk, n=std::min(g_chunk,g_packet_len-offset);
    std::memcpy(bytes+16,g_packet+offset,n);
    auto mbuf=ble_hs_mbuf_from_flat(bytes,16+n); if(!mbuf) return;
    if(ble_gatts_notify_custom(conn,g_sample_handle,mbuf)!=0) return;
    ++g_part;
  }
  if(!g_offered_at) {
    portENTER_CRITICAL(&g_lock); g_buffer->Offered(g_exclusive); portEXIT_CRITICAL(&g_lock);
    g_offered_at=now;
  }
}
}  // namespace m5ble
