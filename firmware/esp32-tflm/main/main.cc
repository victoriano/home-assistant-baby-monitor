#include "baby_monitor_classifier.h"

#include "esp_log.h"

extern "C" void app_main() {
  if (!InitializeBabyMonitorClassifier()) {
    ESP_LOGE("baby_monitor_ml", "Classifier initialization failed");
    return;
  }
  ESP_LOGI(
      "baby_monitor_ml",
      "Classifier initialized. Connect the board-specific camera adapter and "
      "pass cropped grayscale frames to ClassifyBabyMonitorCrop().");
}
