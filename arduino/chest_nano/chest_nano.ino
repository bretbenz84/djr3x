// DJ Rex LED Panels
// Requires FASTLED Library - https://github.com/FastLED/FastLED

#include <FastLED.h>
#include <EEPROM.h>


// How many NeoPixels are attached
#define NUM_LEDS 98
#define BRIGHTNESS 90

// Setup the LED Matrix
#define LED_PIN    4
#define AUDINPIN   3
#define LED_TYPE    WS2811
#define COLOR_ORDER GRB
#define FRAMES_PER_SECOND  30

#define ARRAY_SIZE(A) (sizeof(A) / sizeof((A)[0]))

#define DECAYTIME 80;

// define the 4 LED block starting LED numbers
#define PanelAStart 0
#define PanelA1 PanelAStart + 12
#define PanelA2 PanelA1 + 12
#define PanelA3 PanelA2 + 4

#define PanelBStart PanelAStart + 32
#define PanelB1 PanelBStart + 8
#define PanelB2 PanelB1 + 4
#define PanelB3 PanelB2 + 4

#define PanelCStart PanelBStart + 32
#define PanelC1 PanelCStart + 20
#define PanelC2 PanelC1 + 4
#define PanelC3 PanelC2 + 4

uint16_t IntervalTime[NUM_LEDS];
unsigned long LEDMillis[NUM_LEDS];
bool LEDOn[NUM_LEDS];
const uint8_t StartLEDNum[9] = {PanelA1, PanelA2, PanelA3, PanelB1, PanelB2, PanelB3, PanelC1, PanelC2, PanelC3};
uint8_t Bar1Length = 4;
uint8_t Bar2Length = 4;
uint8_t Bar3Length = 4;

CRGB DJLEDs[NUM_LEDS];
uint8_t Brightness = 150;
#define BLOCKBRIGHTNESS 225

uint8_t LEDBrightness[NUM_LEDS] = { BRIGHTNESS,BRIGHTNESS };
uint8_t LEDMinBrightness[NUM_LEDS] = { BRIGHTNESS,BRIGHTNESS };

// Command loop processing times
unsigned long previousMillis = millis();
unsigned long interval = 5;

unsigned long LEDUpdateMillis = millis();
unsigned long LEDUpdateInterval = 20;

unsigned long FadeMillis = millis();
unsigned long FadeInterval = 0;

uint16_t DecayTime = DECAYTIME;

#define cRED 0xFF0000
#define cBLUE 0x0000FF
#define cWHITE 0xFFFFFF
#define cGOLD 0xFFDD88

#define cRED2 0x500000
#define cBLUE2 0x000055
#define cWHITE2 0x608888

const CRGB SmallLEDColors[9] = { cRED2, cRED2, cWHITE2, cWHITE2, cBLUE2, cBLUE2, cRED2, cWHITE2, cWHITE2 };
const CRGB BlockLEDColors[4] = { cRED, cWHITE, cGOLD, cBLUE };

byte LEDIndex = 0;
bool inout = 0;
byte State = 0;
// setup() function -- runs once at startup --------------------------------

void setup() {

	Serial.begin(115200);

	//DataSetup();

	pinMode(13, OUTPUT);
	digitalWrite(13, HIGH);

	// tell FastLED about the LED strip configuration
	FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(DJLEDs, NUM_LEDS).setCorrection(TypicalLEDStrip);


	// set master brightness control
	FastLED.setBrightness(BRIGHTNESS);

	randomSeed(analogRead(0));
	// Seed the Array
	for (byte x = 0; x < NUM_LEDS; x++) {
		IntervalTime[x] = random16(3000);
		LEDMillis[x] = millis();
		LEDOn[x] = 0;
	}

}

// List of patterns to cycle through.  Each is defined as a separate function below.
typedef void (*SimplePatternList[])();
SimplePatternList gPatterns = {
  LEDsOff,	// 0
  RandomBlocks2,
  AllRed,	// 2
  AllGreen,	//3
  ShortCircuit,	// 4
  ConfettiRedWhite, // 10
  rainbow,
  rainbowWithGlitter,
  confetti,
  juggle,
  bpm
};

uint8_t gCurrentPatternNumber = 1; // Index number of which pattern is current



uint8_t gHue = 0; // rotating "base color" used by many of the patterns
uint8_t gSat = 0; // saturation value
bool updown = 0;
// loop() function -- runs repeatedly as long as board is on ---------------

void loop() {
	if (millis() - previousMillis > interval) {
		//	DataLoop();
		previousMillis = millis();
		// Call the current pattern function once, updating the 'leds' array
		gPatterns[gCurrentPatternNumber]();
    RandomEyes();
	}

	if (millis() - LEDUpdateMillis > LEDUpdateInterval) {

		LEDUpdateMillis = millis();
		FastLED.show();
	}


	// do some periodic updates
	EVERY_N_MILLISECONDS(20) {
		gHue++;  // slowly cycle the "base color" through the rainbow
		if (updown) {
			gSat++;
			if (gSat == 255) updown = 0;
		}
		else {
			gSat--;
			if (gSat == 0) updown = 1;
		}
	}
	// EVERY_N_SECONDS( 5 ) { nextPattern(); } // change patterns periodically

}

void SetMode(uint16_t mode)
{

	switch (mode)
	{
	case 0: //  LEDs off
		gCurrentPatternNumber = 0;
		break;
	case 1: //  LEDs default
		gCurrentPatternNumber = 1;
		break;

	case 99:
		nextPattern();
		break;
	default:
		break;
	}

}

void nextPattern()
{
	// add one to the current pattern number, and wrap around at the end
	gCurrentPatternNumber = (gCurrentPatternNumber + 1) % ARRAY_SIZE(gPatterns);
	// skip 0
	if (gCurrentPatternNumber == 0) gCurrentPatternNumber = 1;
}

// Turns on block of 4 LEDs based on start number
void LEDBlockOn(uint8_t LEDStart, CRGB Color, int Brightness)
{
	byte i;
	for (i = 0; i < 4; i++) {
		DJLEDs[LEDStart + i] = Color;
	}
}

// Random Eyes
// Simple mostly solid eyes with a bit of flicker to them.
void RandomEyes()
{
  byte i;
  byte y;
  int pos;

  for (pos = 96; pos < NUM_LEDS; pos++) {
    if (!LEDOn[pos]) {	// Fade LEDs up or down
        DJLEDs[pos].maximizeBrightness(LEDBrightness[pos]);
        if (LEDBrightness[pos] < BRIGHTNESS) LEDBrightness[pos]++;
//      DJLEDs[pos].fadeToBlackBy(8);
    }
    else {
        DJLEDs[pos].maximizeBrightness(LEDBrightness[pos]);
        if (LEDBrightness[pos] > LEDMinBrightness[pos]) LEDBrightness[pos]--;
    }
    if (millis() - LEDMillis[pos] > IntervalTime[pos]) {
      if (!LEDOn[pos]) { // LED Off - turn in on
          DJLEDs[pos] = cGOLD;
        IntervalTime[pos] = random(200, 1600);
        LEDMillis[pos] = millis();
        LEDOn[pos] = 1;
        LEDMinBrightness[pos] = random(BRIGHTNESS * 0.2, BRIGHTNESS);
      }
      else {	// Turn the LED off
        IntervalTime[pos] = random(200, 2000);
        LEDMillis[pos] = millis();
        LEDOn[pos] = 0;
      }
    }
  }
}
// Random Blocks
// 8 LED bars are random individual
// 4 LED groups are random together as groups of 4.
void RandomBlocks()
{
	byte i;
	byte y;
	int pos;

	// Fade all LEDs
//	fadeToBlackBy(DJLEDs, NUM_LEDS, 10);

	// Turn On LEDs
	// Random single bars
	//int pos = random16(8);
	for (i = 0; i < 8; i++) {
		for (y = 0; y < 3; y++) {
			pos = PanelAStart + i + y * 20;

			if (!LEDOn[pos]) {	// LED off, fade it 
				DJLEDs[pos].fadeToBlackBy(8);
			}
			else {
				//	DJLEDs[pos].fadeLightBy(-5);
			}
			if (millis() - LEDMillis[pos] > IntervalTime[pos]) {
				if (!LEDOn[pos]) { // LED Off - turn in on
					DJLEDs[pos] = SmallLEDColors[random(0, 9)];// CHSV(random(100, 200), random(0, 200), 255); //CHSV(gHue + random8(64), 200, 0);
					IntervalTime[pos] = random(500, 2500);
					LEDMillis[pos] = millis();
					LEDOn[pos] = 1;
				}
				else {	// Turn the LED off
					IntervalTime[pos] = random(500, 3000);
					LEDMillis[pos] = millis();
					LEDOn[pos] = 0;
				}
			}
		}

	}

	// Large Block LEDs
	for (i = 0; i < 3; i++) {
		for (y = 0; y < 3; y++) {
			pos = PanelA1 + i * 4 + y * 20;

			if (!LEDOn[pos]) {	// LED off, fade it 
				for (byte x = 0; x < 4; x++) {
					DJLEDs[pos + x].fadeToBlackBy(8);
				}
			}
			else {
				//	DJLEDs[pos].fadeLightBy(-5);
			}
			if (millis() - LEDMillis[pos] > IntervalTime[pos]) {
				if (!LEDOn[pos]) { // LED Off - turn in on
					byte c = random(0, 3);
					//Serial.println(c);
					LEDBlockOn(pos, BlockLEDColors[c], BLOCKBRIGHTNESS);//CHSV(random(100, 200), random(0, 200), 255), 50);
					IntervalTime[pos] = random(200, 1500);
					LEDMillis[pos] = millis();
					LEDOn[pos] = 1;
				}
				else {	// Turn the LED off
					IntervalTime[pos] = random(200, 2000);
					LEDMillis[pos] = millis();
					LEDOn[pos] = 0;
				}
			}
		}

	}

}

// Random Blocks
// 8 LED bars are bars and random individual
// 4 LED groups are random together as groups of 4.
void RandomBlocks2()
{
	byte i;
	byte y;
	int pos;

	// Fade all LEDs
//	fadeToBlackBy(DJLEDs, NUM_LEDS, 10);

	// Turn On LEDs
	// Random single bars
	//int pos = random16(8);

	// Bar 1 

	// Blink end LED;
	pos = PanelAStart + Bar1Length;
	DJLEDs[pos].fadeToBlackBy(8);
	if (millis() - LEDMillis[pos] > IntervalTime[pos]) {
		if (!LEDOn[pos]) { // LED Off - turn in on
			
			
			IntervalTime[pos] = random(200, 500);
			LEDMillis[pos] = millis();
			LEDOn[pos] = 1;
		}
		else {	// Turn the LED off
			DJLEDs[pos] = SmallLEDColors[random(0, 9)];
			IntervalTime[pos] = random(200, 500);
			LEDMillis[pos] = millis();
			LEDOn[pos] = 0;
		}
	}

	// Change the Length
	pos = PanelAStart;
	if (millis() - LEDMillis[pos] > IntervalTime[pos]) {
			IntervalTime[pos] = random(1000, 3500);
			LEDMillis[pos] = millis();
			Bar1Length = random(1, 7);
			//Serial.print("bar1length: ");
			//Serial.println(Bar1Length);
			for (i = 0; i < 8; i++) {
				pos = PanelAStart + i;
				if (i < Bar1Length)	DJLEDs[pos] = cRED2;
				else DJLEDs[pos] = 0;
			}
	}


	// Bar 2
	for (i = 0; i < 8; i++) {
		//for (y = 1; y < 3; y++) {
		y = 1;
			pos = PanelBStart + i;// + y * 20;

			if (!LEDOn[pos]) {	// LED off, fade it 
				DJLEDs[pos].fadeToBlackBy(8);
			}
			else {
				//	DJLEDs[pos].fadeLightBy(-5);
			}
			if (millis() - LEDMillis[pos] > IntervalTime[pos]) {
				if (!LEDOn[pos]) { // LED Off - turn in on
					DJLEDs[pos] = SmallLEDColors[random(0, 9)];// CHSV(random(100, 200), random(0, 200), 255); //CHSV(gHue + random8(64), 200, 0);
					IntervalTime[pos] = random(500, 2500);
					LEDMillis[pos] = millis();
					LEDOn[pos] = 1;
				}
				else {	// Turn the LED off
					IntervalTime[pos] = random(500, 3000);
					LEDMillis[pos] = millis();
					LEDOn[pos] = 0;
				}
			}
		//}

	}

	// Bar 3 

// Blink end LED;
	pos = PanelCStart + Bar3Length;
	DJLEDs[pos].fadeToBlackBy(8);
	if (millis() - LEDMillis[pos] > IntervalTime[pos]) {
		if (!LEDOn[pos]) { // LED Off - turn in on


			IntervalTime[pos] = random(200, 500);
			LEDMillis[pos] = millis();
			LEDOn[pos] = 1;
		}
		else {	// Turn the LED off
			DJLEDs[pos] = SmallLEDColors[random(0, 9)];
			IntervalTime[pos] = random(200, 500);
			LEDMillis[pos] = millis();
			LEDOn[pos] = 0;
		}
	}

	// Change the Length
	pos = PanelCStart;
	if (millis() - LEDMillis[pos] > IntervalTime[pos]) {
		IntervalTime[pos] = random(1000, 3500);
		LEDMillis[pos] = millis();
		Bar3Length = random(1, 7);
		for (i = 0; i < 8; i++) {
			pos = PanelCStart + i;
			if (i < Bar3Length)	DJLEDs[pos] = cWHITE2;
			else DJLEDs[pos] = 0;
		}
	}



	 //Large Block LEDs
	for (i = 0; i < 9; i++) {
	
			pos = StartLEDNum[i]; 

			if (!LEDOn[pos]) {	// LED off, fade it 
				for (byte x = 0; x < 4; x++) {
					DJLEDs[pos + x].fadeToBlackBy(8);
				}
			}
			else {
				//	DJLEDs[pos].fadeLightBy(-5);
			}
			if (millis() - LEDMillis[pos] > IntervalTime[pos]) {
				if (!LEDOn[pos]) { // LED Off - turn in on
					byte c = random(0, 4);
					//Serial.println(c);
					LEDBlockOn(pos, BlockLEDColors[c], BLOCKBRIGHTNESS);//CHSV(random(100, 200), random(0, 200), 255), 50);
					IntervalTime[pos] = random(200, 1500);
					LEDMillis[pos] = millis();
					LEDOn[pos] = 1;
				}
				else {	// Turn the LED off
					IntervalTime[pos] = random(200, 2000);
					LEDMillis[pos] = millis();
					LEDOn[pos] = 0;
				}
			}
		

	}

}


void rainbow()
{
	// FastLED's built-in rainbow generator
	fill_rainbow(DJLEDs, NUM_LEDS, gHue, 7);

}

void rainbowWithGlitter()
{
	// built-in FastLED rainbow, plus some random sparkly glitter
	rainbow();
	addGlitter(80);
}

void addGlitter(fract8 chanceOfGlitter)
{
	if (random8() < chanceOfGlitter) {
		DJLEDs[random16(NUM_LEDS)] += CRGB::White;

	}
}

void addGlitter4(fract8 chanceOfGlitter)
{
	for (byte i = 0; i < 4; i++) {
		if (random8() < chanceOfGlitter) {
			DJLEDs[random16(NUM_LEDS)] += CRGB::White;

		}
	}
}

void confetti()
{
	// random colored speckles that blink in and fade smoothly
	fadeToBlackBy(DJLEDs, NUM_LEDS, 10);

	int pos = random16(NUM_LEDS);
	DJLEDs[pos] += CHSV(gHue + random8(64), 200, 255);

}
void ConfettiRedWhite()
{
	// random colored speckles that blink in and fade smoothly
	fadeToBlackBy(DJLEDs, NUM_LEDS, 10);

	int pos = random16(NUM_LEDS);
	DJLEDs[pos] += CHSV(0, gSat, 192);

}


void bpm()
{
	// colored stripes pulsing at a defined Beats-Per-Minute (BPM)
	uint8_t BeatsPerMinute = 62;
	CRGBPalette16 palette = PartyColors_p;
	uint8_t beat = beatsin8(BeatsPerMinute, 64, 255);
	for (int i = 0; i < NUM_LEDS; i++) { //9948
		DJLEDs[i] = ColorFromPalette(palette, gHue + (i * 2), beat - gHue + (i * 10));

	}
}

void juggle() {
	// eight colored dots, weaving in and out of sync with each other
	fadeToBlackBy(DJLEDs, NUM_LEDS, 20);

	byte dothue = 0;
	for (int i = 0; i < 8; i++) {
		DJLEDs[beatsin16(i + 7, 0, NUM_LEDS - 1)] |= CHSV(dothue, 200, 255);

		dothue += 32;
	}
}
void AllRed() {
	fill_solid(DJLEDs, NUM_LEDS, CRGB(255, 0, 0));

}

void AllGreen() {
	fill_solid(DJLEDs, NUM_LEDS, CRGB(0, 255, 0));

}

void AllBlue() {
	fill_solid(DJLEDs, NUM_LEDS, CRGB(0, 0, 255));

}






void ShortCircuit() {
	if (millis() - FadeMillis > FadeInterval) {
		addGlitter4(150);
		DecayTime--;
		FadeInterval += 4;
		FadeMillis = millis();
	}

	if (DecayTime == 0) {
		DecayTime = DECAYTIME;
		gCurrentPatternNumber = 0;
		FadeInterval = 0;
	}
	fadeToBlackBy(DJLEDs, NUM_LEDS, 10);

}

void LEDsOff() {
	//  Turn LEDs Off
	fadeToBlackBy(DJLEDs, NUM_LEDS, 5);


}