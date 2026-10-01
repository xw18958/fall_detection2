# M5 BLE protocol v1

Service: `6F4D0001-8F2B-4E3D-9A11-189580000001`. Characteristic suffixes `000002` = information (encrypted read), `000003` = control (encrypted write with response), `000004` = samples (notify), `000005` = status (encrypted read / notify). All integers below are little-endian. One encrypted, bonded central is supported.

Information is JSON with protocol version, device/boot identifiers, firmware version, full model SHA-256, rate, sensor ranges/conversion factors, buffer capacity and current monotonic `time_us`. ATT long reads are supported by NimBLE for the information characteristic.

## Control: 16 bytes

| Offset | Length | Field |
|---|---|---|
| 0 | 1 | Protocol version = 1 |
| 1 | 1 | Opcode |
| 2 | 8 | Session ID |
| 10 | 4 | ACK exclusive sequence watermark |
| 14 | 2 | Reserved zero |

Opcodes: `1=start`, `2=stop`, `3=ack`, `5=Mac ready`, `6=switch to detector`, `7=keep`, `8=discard`. Opcode 4 is no longer used. Start creates a random nonzero 64-bit session and begins local PSRAM buffering even if the Mac is temporarily unavailable. Stop moves the trial to REVIEW and still exposes no sample records. KEEP is the only transition that enables sample transfer; DISCARD clears the local trial. ACK is valid only for the current session and fully offered samples. Mode switching requires no recording, review, transfer or pending samples.

The GATT write response confirms command enqueueing, not completion. Read status to confirm state transitions. Rejected commands set status flag 8 and `last_error=1` in information. After a lost response, read status before repeating start. The Mac sends ready only after it has verified device information, read state and subscribed. On reconnect it opens/recovers the current session journal before data transfer.

## Status: 20 bytes

| Offset | Length | Field |
|---|---|---|
| 0 | 1 | Protocol version = 1 |
| 1 | 1 | Collection mode = 1 |
| 2 | 1 | State: 0 ready, 1 recording, 2 review, 3 saving, 4 complete, 5 full |
| 3 | 1 | Flags: 1 connected, 2 Mac ready, 4 overflow, 8 rejected command |
| 4 | 8 | Session ID (zero before the first session) |
| 12 | 4 | Total produced sample count |
| 16 | 4 | Pending unacknowledged sample count |

## Sample records: 32 bytes each

| Offset | Length | Field |
|---|---|---|
| 0 | 4 | Zero-based session sequence |
| 4 | 8 | Monotonic acquisition microseconds since boot |
| 12 | 12 | Six signed int16 readings: ax, ay, az, gx, gy, gz |
| 24 | 2 | Quality flags: 1 read error, 2 timing gap, 4 saturation |
| 26 | 6 | Reserved zero |

The device stores and transfers the signed int16 MPU6886 register counts unchanged. Physical-unit conversion, normalization and training preprocessing are intentionally left to downstream Mac/server code. Failed reads retain a timestamp/sequence and zero readings with quality flag 1; the quality flags are retained in the hidden journal and metadata `quality` summary but are not exported as model-input columns in `samples.csv`.

## Sample notification framing

Each notification has a 16-byte header followed by a fragment of up to six concatenated sample records:

| Offset | Length | Field |
|---|---|---|
| 0 | 1 | Protocol version = 1 |
| 1 | 1 | Payload type = 1 |
| 2 | 1 | Zero-based fragment index |
| 3 | 1 | Fragment count |
| 4 | 8 | Session ID |
| 12 | 4 | Batch ID |

The sender uses `min(64, ATT_MTU-19)` fragment bytes. Default MTU 23 supports four fragment bytes per 20-byte notification; preferred MTU is 185. A batch is at most 192 bytes / 48 fragments. The receiver tolerates identical duplicate fragments, rejects conflicting duplicates/oversized batches, and resets incomplete reassembly on disconnect.

One batch is outstanding at a time. Backpressure retries the unsent fragment. After two seconds without a durable-write ACK, the entire batch is retransmitted. The Mac validates contiguous sequences and increasing device timestamps, synchronizes the journal, then ACKs the exclusive watermark. Matching duplicate samples are acknowledged again without duplicate journal rows. `stop` freezes production in REVIEW without exposing samples. `keep` enters SAVING and enables transfer; only a drained kept buffer is `complete`. The Mac additionally verifies that its journal count equals the device's final produced count before setting `metadata.json` → `completion.complete` to true. Schema 3 keeps the journal at `.recovery/journal.jsonl`; it does not create a visible `completion.json`.

BLE link encryption protects transport; session/sequence framing handles application replay and restart. It does not provide persistent device storage or acquisition UTC synchronization.

For the acquisition-file schema and training limitations, see [COLLECTED_DATA.md](COLLECTED_DATA.md).
