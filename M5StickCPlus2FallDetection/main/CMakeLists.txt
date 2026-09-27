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
)

target_add_binary_data(
    ${COMPONENT_LIB}
    "${CMAKE_CURRENT_LIST_DIR}/model.tflite"
    BINARY
    RENAME_TO "model_tflite"
    ALIGN 16
)

# GitHub Internet OTA is enabled only when a local RSA private key exists.
# The private key is intentionally gitignored; clean CI/dev checkouts still build,
# but Internet OTA is skipped until tools/generate_ota_keys.sh has been run.
set(FALL_OTA_PRIVATE_KEY "${CMAKE_CURRENT_LIST_DIR}/../security/ota_private.pem")
if(EXISTS "${FALL_OTA_PRIVATE_KEY}")
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FALL_ENCRYPTED_INTERNET_OTA=1)
    target_add_binary_data(
        ${COMPONENT_LIB}
        "${FALL_OTA_PRIVATE_KEY}"
        TEXT
        RENAME_TO "ota_private_pem"
    )
else()
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FALL_ENCRYPTED_INTERNET_OTA=0)
    message(WARNING "security/ota_private.pem not found: encrypted GitHub Internet OTA will be disabled in this build")
endif()
