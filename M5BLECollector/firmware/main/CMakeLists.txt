# IDF 5.5 target_add_binary_data does not parse ALIGN. Explicit assembly keeps
# the FlatBuffer and integer weights aligned without a second RAM model copy.
if(NOT CMAKE_BUILD_EARLY_EXPANSION)
    configure_file("${CMAKE_CURRENT_LIST_DIR}/model_data.S.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/model_data.S" @ONLY)
    set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/model_data.S" PROPERTIES
        OBJECT_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/model.tflite")
endif()
idf_component_register(
    SRCS "main.cc" "gelu_kernel.cc" "layer_norm_v4.cc" "fast_conv1d.cc" "fall_op_resolver.cc"
         "wifi_ota.cc" "ble_collector.cc" "battery_monitor.cc"
         "${CMAKE_CURRENT_BINARY_DIR}/model_data.S"
    INCLUDE_DIRS "."
    PRIV_REQUIRES driver esp_timer app_update esp_event esp_http_server esp_netif
                  lwip esp_wifi nvs_flash bt esp_adc mbedtls
)
