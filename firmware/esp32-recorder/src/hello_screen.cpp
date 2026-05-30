#include <Arduino.h>
#include <SPI.h>
#include <GxEPD2_BW.h>
#include <Fonts/FreeMonoBold12pt7b.h>
#include <Fonts/FreeSansBold9pt7b.h>

// Waveshare ESP32-S3-ePaper-1.54 V2 (ESP32-S3-PICO-1 N8R8).
// Pins from waveshareteam/ESP32-S3-ePaper-1.54 reference firmware.
#define EPD_BUSY 8
#define EPD_RST  9
#define EPD_DC   10
#define EPD_CS   11
#define EPD_SCK  12
#define EPD_MOSI 13
#define EPD_PWR  6   // Active-LOW: drive 0 to power the panel.

GxEPD2_BW<GxEPD2_154_D67, GxEPD2_154_D67::HEIGHT> display(
    GxEPD2_154_D67(EPD_CS, EPD_DC, EPD_RST, EPD_BUSY)
);

void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n[hello-screen] boot");

    pinMode(EPD_PWR, OUTPUT);
    digitalWrite(EPD_PWR, LOW);
    delay(50);

    SPI.begin(EPD_SCK, -1, EPD_MOSI, EPD_CS);

    display.init(115200, true, 2, false);
    display.setRotation(1);
    display.setTextColor(GxEPD_BLACK);

    display.setFullWindow();
    display.firstPage();
    do {
        display.fillScreen(GxEPD_WHITE);

        display.setFont(&FreeMonoBold12pt7b);
        display.setCursor(10, 50);
        display.print("Hello,");
        display.setCursor(10, 90);
        display.print("World!");

        display.setFont(&FreeSansBold9pt7b);
        display.setCursor(10, 150);
        display.print("LocalLexis");
        display.setCursor(10, 180);
        display.print("ESP32-S3 OK");
    } while (display.nextPage());

    display.hibernate();
    Serial.println("[hello-screen] refresh done");
}

void loop() {
    delay(5000);
    Serial.println("[hello-screen] alive");
}
