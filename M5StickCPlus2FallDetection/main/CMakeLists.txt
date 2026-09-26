idf_component_register(
    SRCS
        "main.cc"
        "gelu_kernel.cc"
        "fall_op_resolver.cc"
    INCLUDE_DIRS "."
    PRIV_REQUIRES driver esp_timer
)
