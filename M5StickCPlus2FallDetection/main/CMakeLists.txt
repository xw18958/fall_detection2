idf_component_register(
    SRCS
        "main.cc"
        "gelu_kernel.cc"
        "fall_op_resolver.cc"
        "wifi_ota.cc"
    INCLUDE_DIRS "."
    PRIV_REQUIRES
        driver
        esp_timer
        app_update
        esp_event
        esp_http_server
        esp_netif
        esp_wifi
        nvs_flash
)

target_add_binary_data(
    ${COMPONENT_LIB}
    "${CMAKE_CURRENT_LIST_DIR}/model.tflite"
    BINARY
    RENAME_TO "model_tflite"
    ALIGN 16
)
