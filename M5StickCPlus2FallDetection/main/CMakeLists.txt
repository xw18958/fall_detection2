# The development OTA private key is converted into a small generated assembly
# source before the build. Treating it as an ordinary source file avoids the
# PlatformIO/ESP-IDF binary-embedding path issue seen with target_add_binary_data
# and EMBED_TXTFILES.
set(FALL_OTA_PRIVATE_KEY "${CMAKE_CURRENT_LIST_DIR}/ota_private.pem")
set(FALL_OTA_PRIVATE_BLOB "${CMAKE_CURRENT_LIST_DIR}/ota_private_blob.S")
set(FALL_OTA_EXTRA_SRCS "")
set(FALL_ENCRYPTED_INTERNET_OTA_VALUE 0)
if(EXISTS "${FALL_OTA_PRIVATE_KEY}" AND EXISTS "${FALL_OTA_PRIVATE_BLOB}")
    list(APPEND FALL_OTA_EXTRA_SRCS "ota_private_blob.S")
    set(FALL_ENCRYPTED_INTERNET_OTA_VALUE 1)
endif()

idf_component_register(
    SRCS
        "main.cc"
        "gelu_kernel.cc"
        "fall_op_resolver.cc"
        "wifi_ota.cc"
        "internet_ota.cc"
        ${FALL_OTA_EXTRA_SRCS}
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
)

target_compile_definitions(
    ${COMPONENT_LIB}
    PRIVATE FALL_ENCRYPTED_INTERNET_OTA=${FALL_ENCRYPTED_INTERNET_OTA_VALUE}
)

if(NOT FALL_ENCRYPTED_INTERNET_OTA_VALUE)
    message(WARNING "Development OTA key blob not found: encrypted GitHub Internet OTA will be disabled in this build. Run tools/generate_ota_keys.sh to create it.")
endif()

target_add_binary_data(
    ${COMPONENT_LIB}
    "${CMAKE_CURRENT_LIST_DIR}/model.tflite"
    BINARY
    RENAME_TO "model_tflite"
    ALIGN 16
)
