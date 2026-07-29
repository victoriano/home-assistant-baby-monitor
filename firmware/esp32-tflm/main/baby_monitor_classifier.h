#pragma once

#include <cstddef>
#include <cstdint>

struct BabyMonitorScores {
  float presence;
  float awake;
  float pacifier;
  bool baby_present;
  bool baby_awake;
  bool has_pacifier;
};

// Allocates TensorFlow Lite Micro state and validates the generated model.
bool InitializeBabyMonitorClassifier();

// Runs one already-cropped 160x96 grayscale image.
//
// The exact width, height, and byte count are available in model_metadata.h.
// For a camera with multiple configured regions, invoke this once per region
// and retain the result with the highest presence score.
bool ClassifyBabyMonitorCrop(
    const std::uint8_t* grayscale,
    std::size_t byte_count,
    BabyMonitorScores* scores);
