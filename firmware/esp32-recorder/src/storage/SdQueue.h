#pragma once

#include <Arduino.h>
#include <Preferences.h>
#include <vector>

namespace locallexis::storage {

struct QueueStats {
    uint32_t pending = 0;
    uint64_t totalBytes = 0;
    uint64_t usedBytes = 0;
};

class SdQueue {
public:
    bool begin(int clkPin, int cmdPin, int d0Pin,
               const char* queueDir = "/queue");
    bool ready() const { return ready_; }
    bool enqueue(const std::vector<uint8_t>& wavBytes, String* outPath = nullptr);
    bool peekOldest(String& outPath, std::vector<uint8_t>& outBytes);
    bool removeFile(const String& path);
    QueueStats stats();

private:
    bool ready_ = false;
    String queueDir_;
    Preferences prefs_;
    uint64_t nextN_ = 1;

    bool sweepPartials();
    String nextFilenameStem();
};

}  // namespace locallexis::storage
