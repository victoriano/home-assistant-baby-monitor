set(generated_sources
    "${CMAKE_CURRENT_LIST_DIR}/generated/model_data.cc"
)

if(NOT EXISTS "${generated_sources}")
    message(FATAL_ERROR
        "The private model is missing. Run the ml training command with "
        "--generated-dir ../firmware/esp32-tflm/main/generated first."
    )
endif()

idf_component_register(
    SRCS
        "baby_monitor_classifier.cc"
        "main.cc"
        "${generated_sources}"
    INCLUDE_DIRS
        "."
        "generated"
)
