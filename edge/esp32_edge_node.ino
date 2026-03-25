/*
 * ESP32 Edge Node — Soil Moisture LSTM Inference
 * ================================================
 * Hardware:  ESP32-WROOM-32 + Capacitive Soil Sensor v1.2
 *            DHT22 (temperature/humidity) + SX1276 LoRa module
 *            Solar panel + TP4056 LiPo charger
 *
 * Firmware:  TensorFlow Lite Micro (int8) for LSTM inference
 *            Arduino LoRa library for uplink/downlink
 *            Deep-sleep between readings (30 min interval)
 *
 * Memory:    Model flash ~11 KB  |  Activation RAM ~3 KB
 *            Total sketch target: < 300 KB flash, < 40 KB SRAM
 *
 * NOTE: This is a simulation-ready implementation.
 *       Replace model_data[] with exported edge_lstm_int8.tflite
 *       converted via xxd -i edge_lstm_int8.tflite > model_data.h
 */

#include <Arduino.h>
#include <LoRa.h>
#include <DHT.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"   // generated from TFLite flatbuffer

// ── Pin Configuration ─────────────────────────────────────────
#define SOIL_PIN        34      // ADC1 CH6
#define DHT_PIN         4
#define LORA_SS         18
#define LORA_RST        14
#define LORA_DIO0       26
#define LORA_SCK        5
#define LORA_MISO       19
#define LORA_MOSI       27
#define VALVE_PIN       25      // MOSFET gate for solenoid valve
#define LED_PIN         2
#define BATT_ADC_PIN    35      // Voltage divider for LiPo (1MΩ / 1MΩ)

// ── System Config ─────────────────────────────────────────────
#define PLOT_ID         0       // 0–7, set per device
#define SLEEP_US        (30ULL * 60 * 1000000)   // 30 minutes
#define SEQ_LEN         24
#define N_FEATURES      4
#define HORIZON         4
#define SOIL_DRY_ADC    2800    // calibrate per sensor
#define SOIL_WET_ADC    1200    // calibrate per sensor
#define MOISTURE_PWP    0.20f
#define MOISTURE_FC     0.40f

// ── TFLite Micro Setup ────────────────────────────────────────
constexpr int kTensorArenaSize = 10 * 1024;    // 10 KB arena
uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model*       model      = nullptr;
tflite::MicroInterpreter*  interpreter = nullptr;
TfLiteTensor*              input_tensor = nullptr;
TfLiteTensor*              output_tensor = nullptr;

// ── Circular Buffer for Sequence ─────────────────────────────
float sensor_buffer[SEQ_LEN][N_FEATURES];   // [moisture, temp, rain, et_proxy]
uint8_t buf_head = 0;
bool    buf_full = false;

// ── RTC Memory (persists across deep sleep) ───────────────────
RTC_DATA_ATTR uint16_t  fcnt         = 0;
RTC_DATA_ATTR float     buf_rtc[SEQ_LEN][N_FEATURES];
RTC_DATA_ATTR uint8_t   rtc_head     = 0;
RTC_DATA_ATTR bool      rtc_full     = false;
RTC_DATA_ATTR uint32_t  last_irr_ts  = 0;    // unix seconds


// ══════════════════════════════════════════════════════════════
// SENSOR READING
// ══════════════════════════════════════════════════════════════

DHT dht(DHT_PIN, DHT22);

float read_soil_moisture() {
    int raw = 0;
    for (int i = 0; i < 8; i++) {
        raw += analogRead(SOIL_PIN);
        delay(5);
    }
    raw /= 8;
    float vwc = 1.0f - (float)(raw - SOIL_WET_ADC) /
                        (float)(SOIL_DRY_ADC - SOIL_WET_ADC);
    return constrain(vwc, 0.0f, 1.0f);
}

float read_temperature() {
    float t = dht.readTemperature();
    return isnan(t) ? 28.0f : t;
}

uint16_t read_battery_mv() {
    int raw = analogRead(BATT_ADC_PIN);
    // 3.3V ref, 12-bit ADC, 1:1 divider → ×2 for actual voltage
    return (uint16_t)((raw / 4095.0f) * 3300 * 2);
}


// ══════════════════════════════════════════════════════════════
// TFLITE MICRO INFERENCE
// ══════════════════════════════════════════════════════════════

bool init_tflite() {
    model = tflite::GetModel(g_model_data);   // from model_data.h
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[TFLite] Schema version mismatch");
        return false;
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[TFLite] AllocateTensors failed");
        return false;
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);
    Serial.printf("[TFLite] Arena used: %d bytes\n",
                  interpreter->arena_used_bytes());
    return true;
}

/**
 * Run LSTM inference on the current circular buffer.
 * Returns predicted soil moisture for next HORIZON steps.
 * Input tensor: int8 [1, SEQ_LEN, N_FEATURES]
 * Output tensor: int8 [1, HORIZON]
 */
bool run_inference(float predictions[HORIZON]) {
    if (!buf_full && rtc_head < SEQ_LEN) {
        Serial.println("[Inference] Buffer not full yet, skipping");
        return false;
    }

    // Quantization parameters (from TFLite metadata)
    const float in_scale  = input_tensor->params.scale;
    const int   in_zero   = input_tensor->params.zero_point;
    const float out_scale = output_tensor->params.scale;
    const int   out_zero  = output_tensor->params.zero_point;

    int8_t* in_data = input_tensor->data.int8;

    // Fill input tensor from circular buffer (oldest → newest)
    for (int t = 0; t < SEQ_LEN; t++) {
        int idx = (rtc_head + t) % SEQ_LEN;
        for (int f = 0; f < N_FEATURES; f++) {
            float val      = buf_rtc[idx][f];
            int8_t q       = (int8_t)((val / in_scale) + in_zero);
            in_data[t * N_FEATURES + f] = q;
        }
    }

    TfLiteStatus status = interpreter->Invoke();
    if (status != kTfLiteOk) {
        Serial.println("[Inference] Invoke failed");
        return false;
    }

    int8_t* out_data = output_tensor->data.int8;
    for (int h = 0; h < HORIZON; h++) {
        predictions[h] = ((float)out_data[h] - out_zero) * out_scale;
        predictions[h] = constrain(predictions[h], 0.0f, 1.0f);
    }
    return true;
}


// ══════════════════════════════════════════════════════════════
// LORA COMMUNICATION
// ══════════════════════════════════════════════════════════════

bool init_lora() {
    SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_SS);
    LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);
    if (!LoRa.begin(865E6)) {     // IN865 band
        Serial.println("[LoRa] Init failed");
        return false;
    }
    LoRa.setSpreadingFactor(9);
    LoRa.setSignalBandwidth(125E3);
    LoRa.setCodingRate4(5);
    LoRa.setTxPower(14);
    Serial.println("[LoRa] Ready (SF9, BW125, 865 MHz)");
    return true;
}

void send_sensor_packet(float moisture, float temperature,
                         float pred_1h, uint16_t batt_mv) {
    /*
     * Payload layout (10 bytes):
     *  [0]   plot_id  u8
     *  [1-2] moisture u16  ×10000
     *  [3-4] temp     s16  ×100
     *  [5-6] pred_1h  u16  ×10000
     *  [7-8] battery  u16  mV
     *  [9]   fcnt_lsb u8   (for dedup at fog)
     */
    LoRa.beginPacket();
    LoRa.write(PLOT_ID);
    LoRa.write((uint8_t*)&(uint16_t){(uint16_t)(moisture * 10000)}, 2);
    LoRa.write((uint8_t*)&(int16_t){(int16_t)(temperature * 100)}, 2);
    LoRa.write((uint8_t*)&(uint16_t){(uint16_t)(pred_1h * 10000)}, 2);
    LoRa.write((uint8_t*)&batt_mv, 2);
    LoRa.write((uint8_t)(fcnt & 0xFF));
    LoRa.endPacket();
    Serial.printf("[LoRa] Sent: moisture=%.3f temp=%.1f pred=%.3f batt=%dmV fcnt=%d\n",
                  moisture, temperature, pred_1h, batt_mv, fcnt);
}

bool receive_command(uint8_t timeout_ms) {
    /*
     * Downlink command from fog node (4 bytes):
     *  [0] plot_id   u8
     *  [1] valve     u8  (0=close, 1=open)
     *  [2-3] dur_s   u16
     */
    unsigned long start = millis();
    while (millis() - start < timeout_ms) {
        int pkt = LoRa.parsePacket();
        if (pkt >= 4) {
            uint8_t  pid   = LoRa.read();
            uint8_t  valve = LoRa.read();
            uint16_t dur   = (uint16_t)LoRa.read() << 8;
            dur            |= LoRa.read();

            if (pid == PLOT_ID) {
                Serial.printf("[CMD] valve=%d dur=%ds\n", valve, dur);
                if (valve && dur > 0) {
                    open_valve(dur * 1000UL);
                } else {
                    close_valve();
                }
                return true;
            }
        }
    }
    return false;
}


// ══════════════════════════════════════════════════════════════
// IRRIGATION CONTROL
// ══════════════════════════════════════════════════════════════

void open_valve(unsigned long duration_ms) {
    Serial.printf("[Valve] OPEN for %lu ms\n", duration_ms);
    digitalWrite(VALVE_PIN, HIGH);
    last_irr_ts = millis() / 1000;   // rough unix-ish timestamp
    delay(duration_ms);
    close_valve();
}

void close_valve() {
    digitalWrite(VALVE_PIN, LOW);
    Serial.println("[Valve] CLOSED");
}

/**
 * Emergency local decision: irrigate without fog confirmation.
 * Triggered when predicted moisture < PWP + 0.02 AND
 * last irrigation was > 6 hours ago.
 */
bool local_emergency_irrigate(float predicted_min, uint32_t now_s) {
    bool critical = predicted_min < (MOISTURE_PWP + 0.02f);
    bool cooldown = (now_s - last_irr_ts) > 6 * 3600;
    if (critical && cooldown) {
        Serial.println("[Emergency] Local irrigation triggered!");
        open_valve(15 * 60 * 1000UL);   // 15 min
        return true;
    }
    return false;
}


// ══════════════════════════════════════════════════════════════
// SETUP & LOOP
// ══════════════════════════════════════════════════════════════

void setup() {
    Serial.begin(115200);
    pinMode(VALVE_PIN, OUTPUT);
    digitalWrite(VALVE_PIN, LOW);
    pinMode(LED_PIN, OUTPUT);

    dht.begin();

    // Restore buffer from RTC memory
    memcpy(sensor_buffer, buf_rtc, sizeof(buf_rtc));
    buf_head = rtc_head;
    buf_full = rtc_full;

    Serial.printf("\n[Boot] Plot %d | FCnt %d | BufHead %d\n",
                  PLOT_ID, fcnt, buf_head);

    // ── 1. Read sensors ───────────────────────────────────────
    float moisture = read_soil_moisture();
    float temp     = read_temperature();
    uint16_t batt  = read_battery_mv();
    float et_proxy = max(0.0f, 0.0023f * (temp + 17.8f) * sqrtf(15.0f) * 0.408f / 48.0f);

    // ── 2. Push to circular buffer ────────────────────────────
    buf_rtc[buf_head][0] = moisture;
    buf_rtc[buf_head][1] = temp / 50.0f;   // normalize
    buf_rtc[buf_head][2] = 0.0f;            // rain (no sensor, fog provides)
    buf_rtc[buf_head][3] = et_proxy;
    buf_head = (buf_head + 1) % SEQ_LEN;
    if (buf_head == 0) rtc_full = true;
    rtc_head = buf_head;

    // ── 3. TFLite inference ───────────────────────────────────
    float predictions[HORIZON] = {moisture, moisture, moisture, moisture};
    if (init_tflite()) {
        if (run_inference(predictions)) {
            Serial.printf("[Inference] Next 2h: %.3f → %.3f → %.3f → %.3f\n",
                          predictions[0], predictions[1],
                          predictions[2], predictions[3]);
        }
    }

    // ── 4. LoRa transmit ──────────────────────────────────────
    if (init_lora()) {
        send_sensor_packet(moisture, temp, predictions[1], batt);
        fcnt++;

        // Wait for fog command (2-second window)
        bool got_cmd = receive_command(2000);
        if (!got_cmd) {
            // ── 5. Local emergency fallback ───────────────────
            float min_pred = *std::min_element(predictions, predictions + HORIZON);
            local_emergency_irrigate(min_pred, millis() / 1000);
        }
    }

    // ── 6. Deep sleep ─────────────────────────────────────────
    Serial.println("[Sleep] Entering deep sleep (30 min)");
    Serial.flush();
    LoRa.sleep();
    esp_sleep_enable_timer_wakeup(SLEEP_US);
    esp_deep_sleep_start();
}

void loop() {
    // Never reached — deep sleep cycles through setup()
}
