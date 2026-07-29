#include "baby_monitor_classifier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <new>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "generated/model_data.h"
#include "generated/model_metadata.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr char kTag[] = "baby_monitor_ml";
const tflite::Model* g_model = nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
std::uint8_t* g_tensor_arena = nullptr;
bool g_initialization_attempted = false;
alignas(tflite::MicroInterpreter)
    std::uint8_t g_interpreter_storage[sizeof(tflite::MicroInterpreter)];

float Dequantize(const TfLiteTensor* tensor) {
  const int value = tensor->data.int8[0];
  return (value - tensor->params.zero_point) * tensor->params.scale;
}

bool ReadScores(BabyMonitorScores* scores) {
  const auto* presence =
      g_interpreter->output(baby_monitor_model::kPresenceOutputSlot);
  const auto* awake = g_interpreter->output(baby_monitor_model::kAwakeOutputSlot);
  const auto* pacifier =
      g_interpreter->output(baby_monitor_model::kPacifierOutputSlot);
  if (presence == nullptr || awake == nullptr || pacifier == nullptr ||
      presence->type != kTfLiteInt8 || awake->type != kTfLiteInt8 ||
      pacifier->type != kTfLiteInt8) {
    ESP_LOGE(kTag, "Unexpected model output tensors");
    return false;
  }
  scores->presence = Dequantize(presence);
  scores->awake = Dequantize(awake);
  scores->pacifier = Dequantize(pacifier);
  scores->baby_present =
      scores->presence >= baby_monitor_model::kPresenceThreshold;
  scores->baby_awake =
      scores->baby_present &&
      scores->awake >= baby_monitor_model::kAwakeThreshold;
  scores->has_pacifier =
      scores->baby_present &&
      scores->pacifier >= baby_monitor_model::kPacifierThreshold;
  return true;
}

}  // namespace

bool InitializeBabyMonitorClassifier() {
  if (g_initialization_attempted) {
    return g_interpreter != nullptr;
  }
  g_initialization_attempted = true;
  g_model = tflite::GetModel(g_baby_monitor_model);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(kTag, "Model schema %d does not match runtime schema %d",
             g_model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  g_tensor_arena = static_cast<std::uint8_t*>(heap_caps_aligned_alloc(
      16, baby_monitor_model::kTensorArenaBytes,
      MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (g_tensor_arena == nullptr) {
    ESP_LOGE(kTag, "Could not allocate %u-byte tensor arena in PSRAM",
             static_cast<unsigned>(baby_monitor_model::kTensorArenaBytes));
    return false;
  }

  static tflite::MicroMutableOpResolver<5> resolver;
  if (resolver.AddConv2D() != kTfLiteOk ||
      resolver.AddDepthwiseConv2D() != kTfLiteOk ||
      resolver.AddMean() != kTfLiteOk ||
      resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddLogistic() != kTfLiteOk) {
    ESP_LOGE(kTag, "Could not register model operators");
    return false;
  }

  g_interpreter = new (g_interpreter_storage) tflite::MicroInterpreter(
      g_model, resolver, g_tensor_arena,
      baby_monitor_model::kTensorArenaBytes);
  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(kTag, "AllocateTensors failed");
    g_interpreter = nullptr;
    return false;
  }
  g_input = g_interpreter->input(0);
  const std::size_t expected =
      baby_monitor_model::kInputHeight * baby_monitor_model::kInputWidth *
      baby_monitor_model::kInputChannels;
  if (g_input == nullptr || g_input->type != kTfLiteInt8 ||
      static_cast<std::size_t>(g_input->bytes) != expected) {
    ESP_LOGE(kTag, "Unexpected model input tensor");
    g_interpreter = nullptr;
    return false;
  }
  ESP_LOGI(kTag, "Model ready: %ux%ux%u, arena=%u bytes",
           baby_monitor_model::kInputWidth,
           baby_monitor_model::kInputHeight,
           baby_monitor_model::kInputChannels,
           static_cast<unsigned>(baby_monitor_model::kTensorArenaBytes));
  return true;
}

bool ClassifyBabyMonitorCrop(const std::uint8_t* grayscale,
                             std::size_t byte_count,
                             BabyMonitorScores* scores) {
  if (grayscale == nullptr || scores == nullptr ||
      !InitializeBabyMonitorClassifier()) {
    return false;
  }
  const std::size_t expected =
      baby_monitor_model::kInputHeight * baby_monitor_model::kInputWidth *
      baby_monitor_model::kInputChannels;
  if (byte_count != expected) {
    ESP_LOGE(kTag, "Expected %u input bytes, received %u",
             static_cast<unsigned>(expected),
             static_cast<unsigned>(byte_count));
    return false;
  }

  const float multiplier = 1.0F / (255.0F * g_input->params.scale);
  for (std::size_t index = 0; index < expected; ++index) {
    const int quantized = static_cast<int>(
        std::lround(grayscale[index] * multiplier) +
        g_input->params.zero_point);
    g_input->data.int8[index] =
        static_cast<std::int8_t>(std::clamp(quantized, -128, 127));
  }
  if (g_interpreter->Invoke() != kTfLiteOk) {
    ESP_LOGE(kTag, "Model invocation failed");
    return false;
  }
  return ReadScores(scores);
}
