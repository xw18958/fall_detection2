# Keep the development OTA private key as a local, gitignored build input.
# Using ESP-IDF's component EMBED_TXTFILES with a component-relative path avoids
# PlatformIO/SCons path rewriting issues seen with an absolute target_add_binary_data path.
set(FALL_OTA_PRIVATE_KEY "${CMAKE_CURRENT_LIST_DIR}/ota_private.pem")
set(FALL_OTA_EMBED_TXTFILES "")
set(FALL_ENCRYPTED_INTERNET_OTA_VALUE 0)
if(EXISTS "${FALL_OTA_PRIVATE_KEY}")
    list(APPEND FALL_OTA_EMBED_TXTFILES "ota_private.pem")
    set(FALL_ENCRYPTED_INTERNET_OTA_VALUE 1)
endif()

idf_component_register(
    SRCS
        "main.cc"
        "gelu_kernel.cc"
        "fall_op_resolver.cc"
        "wifi_ota.cc"
        "internet_ota.cc"
    INCLUDE_DIRS "."
    PRIV_REQUIRES
        driver
        esp_timer
        app_update
        esp_http_client
        esp_https_ota
        mbedtls
        esp_event
        esp_http_server
        esp_netif
        lwip
        esp_wifi
        nvs_flash
        json
    EMBED_TXTFILES ${FALL_OTA_EMBED_TXTFILES}
)

target_compile_definitions(
    ${COMPONENT_LIB}
    PRIVATE FALL_ENCRYPTED_INTERNET_OTA=${FALL_ENCRYPTED_INTERNET_OTA_VALUE}
)

if(NOT FALL_ENCRYPTED_INTERNET_OTA_VALUE)
    message(WARNING "main/ota_private.pem not found: encrypted GitHub Internet OTA will be disabled in this build")
endif()

target_add_binary_data(
    ${COMPONENT_LIB}
    "${CMAKE_CURRENT_LIST_DIR}/model.tflite"
    BINARY
    RENAME_TO "model_tflite"
    ALIGN 16
)
