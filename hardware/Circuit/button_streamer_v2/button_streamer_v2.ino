// Arduino Nano - button state streamer v2 (publication-grade timing)
// =====================================================================
// CSV line:  t_ms,B,A1,A2,...,A8      @ ~200 Hz, 115200 baud
//   t_ms = device millis() at sample time. USE THIS for inter-button
//          offsets (synchronicity) - it is immune to USB/OS read jitter,
//          unlike a host-side arrival timestamp.
//   B    = 1 if ANY B-side button is pressed (all wired to D10).
//   Ai   = 1 if pair i's A button is pressed.
//
// WHY THIS VERSION
//   v1 ran at 20 Hz (delay(50)) and 9600 baud, so the two buttons of a
//   pair could only be timed to +-50 ms and the host stamped arrival time,
//   not press time. That is too coarse for a bimanual-synchronicity metric
//   (human two-hand offsets are ~20-200 ms). This version samples at
//   ~200 Hz (5 ms), debounces contact bounce, and emits a DEVICE timestamp
//   so |t_rise(B) - t_rise(Ai)| is resolved to a few ms.
//
// WIRING (unchanged, INPUT_PULLUP, LOW = pressed):
//   D10 -> all B buttons (other sides to GND)
//   D2,D3,D4,D6,D7,D8,D11,D12 -> A button of pairs 1..8 (other sides to GND)
//   GND -> breadboard GND rail
//
// NOTE: at 200 Hz a 1 h session is ~720k rows. Log the raw stream to CSV,
// not .xlsx (Excel's ~1,048,576-row limit + slow writes). Per-sequence
// trial files (~30-90 s) are small either way.

const int B_PIN     = 10;
const int A_PINS[]  = {2, 3, 4, 6, 7, 8, 11, 12};
const int N_PAIRS   = 8;
const int N_CH      = 1 + N_PAIRS;          // index 0 = B, 1..8 = A pins

const unsigned long DEBOUNCE_MS = 3;        // ignore contact bounce shorter than this
const unsigned long PERIOD_US   = 5000;     // 5 ms  -> ~200 Hz sampling

int           stableState[N_CH];            // debounced, reported state
int           lastRaw[N_CH];                // last raw read (for edge timing)
unsigned long lastChange[N_CH];             // ms of last raw change

int readCh(int idx) {                       // idx 0 -> B, 1..8 -> A pins
  int pin = (idx == 0) ? B_PIN : A_PINS[idx - 1];
  return (digitalRead(pin) == LOW) ? 1 : 0; // INPUT_PULLUP: LOW = pressed
}

void setup() {
  pinMode(B_PIN, INPUT_PULLUP);
  for (int i = 0; i < N_PAIRS; i++) pinMode(A_PINS[i], INPUT_PULLUP);
  unsigned long t = millis();
  for (int i = 0; i < N_CH; i++) {
    stableState[i] = readCh(i);
    lastRaw[i]     = stableState[i];
    lastChange[i]  = t;
  }
  Serial.begin(115200);
}

void loop() {
  unsigned long t_us = micros();            // loop-start (for the rate limiter)
  unsigned long t    = millis();

  // Debounce each channel: accept a new state only once the raw read has
  // held for DEBOUNCE_MS. This keeps a single bounce from faking an edge,
  // so the first sustained edge time is a clean press instant.
  for (int i = 0; i < N_CH; i++) {
    int r = readCh(i);
    if (r != lastRaw[i]) { lastRaw[i] = r; lastChange[i] = t; }
    if (r != stableState[i] && (t - lastChange[i]) >= DEBOUNCE_MS)
      stableState[i] = r;
  }

  // Emit: device timestamp first, then B, then A1..A8 (same channel order
  // as v1 so PAIR_TO_FACE mapping is unchanged; only the leading t_ms is new).
  Serial.print(t);
  for (int i = 0; i < N_CH; i++) { Serial.print(','); Serial.print(stableState[i]); }
  Serial.println();

  // Hold ~200 Hz. Unsigned subtraction is overflow-safe across the ~71 min
  // micros() wrap, so this stays correct for a full session.
  while ((unsigned long)(micros() - t_us) < PERIOD_US) { /* busy-wait */ }
}
