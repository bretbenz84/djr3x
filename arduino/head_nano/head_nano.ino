/*
 * head_nano.ino — DJ-R3X Head LED Board (v2)
 *
 * Hardware
 * --------
 *   82 WS2812B NeoPixels on D6 (FastLED)
 *   Pixels 0–1   : left and right eyes
 *   Pixels 2–81  : mouth (80-pixel trapezoid PCB)
 *
 * Mouth layout — 10 rows × 8 cols, serpentine wiring
 * ---------------------------------------------------
 *   Even rows (0,2,4,6,8) wire left→right.
 *   Odd  rows (1,3,5,7,9) wire right→left.
 *   Physical center of the array is at grid position (row=4.5, col=3.5).
 *
 *   Centre cluster (zone 0): pixels 37, 38, 45, 46 (mouth offset +2)
 *     37 = row4 col3   38 = row4 col4
 *     45 = row5 col4*  46 = row5 col3*   (* serpentine reversal)
 *
 *   Zones by Euclidean distance from (4.5, 3.5):
 *     Zone 0  dist < 1.0   — centre cluster        ( 4 pixels)
 *     Zone 1  dist < 2.4   — inner ring             (12 pixels)
 *     Zone 2  dist < 3.2   — middle ring            (16 pixels)
 *     Zone 3  dist < 4.5   — outer ring             (28 pixels)
 *     Zone 4  dist ≥ 4.5   — outermost edges        (20 pixels)
 *
 * Serial protocol — 115200 baud, ASCII, newline-terminated
 * ---------------------------------------------------------
 *   SPEAK:{emotion}       Start speaking animation.
 *                         emotion = neutral | happy | excited | sad | angry
 *   SPEAK_LEVEL:{0-255}   Update audio intensity — drives pulse speed + brightness.
 *                         Send as often as needed; non-blocking.
 *   SPEAK_STOP            Mouth off immediately; eyes unchanged.
 *   IDLE                  Mouth off; eyes slow blue breathing pulse.
 *   ACTIVE                Mouth off; eyes bright white.
 *   EYE:{r},{g},{b}       Set both eyes to RGB colour.
 *   OFF                   All 82 pixels off immediately.
 */

#include <FastLED.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Pin / layout constants
// ---------------------------------------------------------------------------

#define DATA_PIN    6
#define NUM_EYES    2
#define NUM_MOUTH   80
#define NUM_LEDS    (NUM_EYES + NUM_MOUTH)   // 82 total; eyes first, mouth second
#define MOUTH_START NUM_EYES                 // mouth pixels begin at index 2
#define NUM_ZONES   5

#define BAUD_RATE  115200
#define SERIAL_BUF 64

// ---------------------------------------------------------------------------
// Zone lookup table  (stored in flash — saves ~80 bytes of SRAM)
// ---------------------------------------------------------------------------
//
// Layout:   10 rows × 8 pixels.  Rows alternate L→R / R→L (serpentine).
// Symmetry: the table is symmetric top↔bottom and left↔right, which gives
//           the concentric diamond / ellipse pattern radiating from centre.
//
//   Row 0  top edge  (even, L→R)
//   Row 1            (odd,  R→L : phys cols 7→0)
//   Row 2            (even, L→R)
//   Row 3            (odd,  R→L)
//   Row 4  ← zone-0 pixels 35,36 are at this row, cols 3 & 4
//   Row 5  ← zone-0 pixels 43,44 are at this row, cols 4 & 3 (serpentine)
//   Row 6
//   Row 7
//   Row 8
//   Row 9  bottom edge (odd, R→L)

const uint8_t PIXEL_ZONE[NUM_MOUTH] PROGMEM = {
    4, 4, 4, 4, 4, 4, 4, 4,   // row 0 — top edge
    4, 3, 3, 3, 3, 3, 3, 4,   // row 1
    3, 3, 2, 2, 2, 2, 3, 3,   // row 2
    3, 2, 1, 1, 1, 1, 2, 3,   // row 3
    3, 2, 1, 0, 0, 1, 2, 3,   // row 4  ← pixels 35,36 = zone 0
    3, 2, 1, 0, 0, 1, 2, 3,   // row 5  ← pixels 43,44 = zone 0
    3, 2, 1, 1, 1, 1, 2, 3,   // row 6
    3, 3, 2, 2, 2, 2, 3, 3,   // row 7
    4, 3, 3, 3, 3, 3, 3, 4,   // row 8
    4, 4, 4, 4, 4, 4, 4, 4,   // row 9 — bottom edge
};

// ---------------------------------------------------------------------------
// Emotion colour table
// ---------------------------------------------------------------------------

struct EmotionColor { uint8_t r, g, b; };

#define EMO_NEUTRAL  0
#define EMO_HAPPY    1
#define EMO_EXCITED  2
#define EMO_SAD      3
#define EMO_ANGRY    4
#define EMO_COUNT    5

const EmotionColor EMOTION_COLORS[EMO_COUNT] PROGMEM = {
    { 255, 140,   0 },   // neutral  — warm amber
    {   0, 200, 255 },   // happy    — cyan blue
    { 255, 200,   0 },   // excited  — yellow-orange
    {  40,   0, 200 },   // sad      — deep blue-purple
    { 255,   0,   0 },   // angry    — red
};

// ---------------------------------------------------------------------------
// LED array
// ---------------------------------------------------------------------------

CRGB leds[NUM_LEDS];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

enum AnimMode : uint8_t {
    ANIM_OFF,
    ANIM_SPEAK,
    ANIM_IDLE,
    ANIM_ACTIVE,
};

AnimMode animMode = ANIM_OFF;

// Speaking state
EmotionColor speakColor  = { 255, 140, 0 };  // default: neutral amber
uint8_t      speakLevel  = 0;                 // 0–255 audio intensity
float        speakPhase  = 0.0f;              // wave front 0.0 – NUM_ZONES

// Idle breathing state
float        idlePhase   = 0.0f;              // 0.0 – TWO_PI

uint32_t     lastMs      = 0;

// ---------------------------------------------------------------------------
// Serial
// ---------------------------------------------------------------------------

char    serialBuf[SERIAL_BUF];
uint8_t serialPos = 0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

inline uint8_t clampByte(int v) {
    if (v < 0)   return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

inline void setEyes(uint8_t r, uint8_t g, uint8_t b) {
    leds[0] = CRGB(r, g, b);
    leds[1] = CRGB(r, g, b);
}

inline void mouthOff() {
    for (uint8_t i = MOUTH_START; i < NUM_LEDS; i++) leds[i] = CRGB::Black;
}

static uint8_t parseEmotion(const char *s) {
    if (strcmp(s, "happy")   == 0) return EMO_HAPPY;
    if (strcmp(s, "excited") == 0) return EMO_EXCITED;
    if (strcmp(s, "sad")     == 0) return EMO_SAD;
    if (strcmp(s, "angry")   == 0) return EMO_ANGRY;
    return EMO_NEUTRAL;
}

// ---------------------------------------------------------------------------
// Command dispatch
// ---------------------------------------------------------------------------

void handleCommand(char *cmd) {

    // SPEAK_LEVEL:{0-255}  — check before SPEAK: to avoid prefix collision
    if (strncmp(cmd, "SPEAK_LEVEL:", 12) == 0) {
        speakLevel = clampByte(atoi(cmd + 12));
        return;
    }

    // SPEAK_STOP
    if (strcmp(cmd, "SPEAK_STOP") == 0) {
        animMode = ANIM_OFF;
        mouthOff();
        FastLED.show();
        return;
    }

    // SPEAK:{emotion}
    if (strncmp(cmd, "SPEAK:", 6) == 0) {
        uint8_t emo = parseEmotion(cmd + 6);
        // Read colour from PROGMEM
        EmotionColor ec;
        memcpy_P(&ec, &EMOTION_COLORS[emo], sizeof(EmotionColor));
        speakColor = ec;
        speakPhase = 0.0f;
        animMode   = ANIM_SPEAK;
        return;
    }

    // IDLE — mouth off, eyes slow blue breathing
    if (strcmp(cmd, "IDLE") == 0) {
        animMode  = ANIM_IDLE;
        idlePhase = 0.0f;
        mouthOff();
        FastLED.show();
        return;
    }

    // ACTIVE — mouth off, eyes bright white, ready for speaking
    if (strcmp(cmd, "ACTIVE") == 0) {
        animMode = ANIM_ACTIVE;
        mouthOff();
        setEyes(255, 255, 255);
        FastLED.show();
        return;
    }

    // EYE:{r},{g},{b}
    if (strncmp(cmd, "EYE:", 4) == 0) {
        int r, g, b;
        if (sscanf(cmd + 4, "%d,%d,%d", &r, &g, &b) == 3) {
            setEyes(clampByte(r), clampByte(g), clampByte(b));
            FastLED.show();
        }
        return;
    }

    // OFF — everything off
    if (strcmp(cmd, "OFF") == 0) {
        animMode = ANIM_OFF;
        FastLED.clear();
        FastLED.show();
        return;
    }

    // Unknown — ignore silently
}

// ---------------------------------------------------------------------------
// Speaking pulse animation
// ---------------------------------------------------------------------------
//
// A sine-shaped wave front advances from zone 0 outward to zone 4, looping
// continuously.  Speed and peak brightness both scale with speakLevel.
//
// For each pixel at zone Z, brightness is:
//   diff = speakPhase - Z           (how far the wave has passed this zone)
//   pulse = sin(π × (diff+LEAD) / WINDOW)   for diff in [-LEAD, WINDOW-LEAD]
//
// LEAD  = 0.30  slight pre-glow before the wave arrives
// WINDOW= 1.70  total pulse width in zone units (enter 0.30 before, exit 1.40 after peak)
//
// An ambient floor (0.12) keeps the mouth dimly lit at all times while speaking.

#define SPEAK_LEAD    0.30f
#define SPEAK_WINDOW  1.70f

void tickSpeak(float dt) {
    // Wave speed: 1.5 zones/s at level 0 → 8.0 zones/s at level 255
    float speed = 1.5f + (speakLevel / 255.0f) * 6.5f;
    speakPhase += speed * dt;
    if (speakPhase >= (float)NUM_ZONES) speakPhase -= (float)NUM_ZONES;

    // Peak brightness: 0.30 at level 0 → 1.00 at level 255
    float peak    = 0.30f + (speakLevel / 255.0f) * 0.70f;
    float ambient = 0.12f;

    for (uint8_t i = 0; i < NUM_MOUTH; i++) {
        float zone = (float)pgm_read_byte(&PIXEL_ZONE[i]);
        // Mouth pixels start at index MOUTH_START (2); zone table is 0-indexed
        uint8_t ledIdx = i + MOUTH_START;
        float diff = speakPhase - zone;

        // Wrap so waves look continuous when front passes zone 4 → zone 0
        if (diff < -SPEAK_LEAD) diff += (float)NUM_ZONES;

        float pulse = 0.0f;
        if (diff >= -SPEAK_LEAD && diff <= (SPEAK_WINDOW - SPEAK_LEAD)) {
            pulse = sin(PI * (diff + SPEAK_LEAD) / SPEAK_WINDOW);
            if (pulse < 0.0f) pulse = 0.0f;
        }

        float brightness = ambient + pulse * peak;
        if (brightness > 1.0f) brightness = 1.0f;

        uint8_t sc = (uint8_t)(brightness * 255.0f);
        leds[ledIdx] = CRGB(
            scale8(speakColor.r, sc),
            scale8(speakColor.g, sc),
            scale8(speakColor.b, sc)
        );
    }
    FastLED.show();
}

// ---------------------------------------------------------------------------
// Idle animation
// ---------------------------------------------------------------------------
//
// Eyes: solid — left untouched so the EYE:{r,g,b} command from the Pi holds.
// Mouth: completely off.  mouthOff() already cleared all pixels on IDLE entry;
//        tickIdle() does not touch mouth pixels, so they stay dark.

void tickIdle(float dt) {
    (void)dt;
    // Eyes are not touched — they remain solid at whatever EYE:{r,g,b} set.
    // Mouth is already off from mouthOff() called in the IDLE command handler.
    // Nothing to do; no FastLED.show() needed since nothing changed.
}

// ---------------------------------------------------------------------------
// Main animation tick — call every loop()
// ---------------------------------------------------------------------------

void tickAnimation() {
    if (animMode == ANIM_OFF || animMode == ANIM_ACTIVE) return;

    uint32_t now     = millis();
    float    dt      = (now - lastMs) * 0.001f;   // seconds since last tick
    lastMs           = now;

    if (dt > 0.1f) dt = 0.1f;   // clamp: ignore stalls > 100 ms (e.g. first tick)

    if (animMode == ANIM_SPEAK) { tickSpeak(dt); return; }
    if (animMode == ANIM_IDLE)  { tickIdle(dt);  return; }
}

// ---------------------------------------------------------------------------
// setup / loop
// ---------------------------------------------------------------------------

void setup() {
    FastLED.addLeds<WS2812B, DATA_PIN, GRB>(leds, NUM_LEDS);
    FastLED.setBrightness(255);
    FastLED.clear();
    FastLED.show();

    Serial.begin(BAUD_RATE);
    serialPos = 0;
    lastMs    = millis();
}

void loop() {
    // Serial command reader — buffer until newline, then dispatch
    while (Serial.available()) {
        char c = (char)Serial.read();
        if (c == '\n' || c == '\r') {
            if (serialPos > 0) {
                serialBuf[serialPos] = '\0';
                handleCommand(serialBuf);
                serialPos = 0;
            }
        } else if (serialPos < SERIAL_BUF - 1) {
            serialBuf[serialPos++] = c;
        }
        // If buffer overflows, discard characters until next newline
    }

    tickAnimation();
}
