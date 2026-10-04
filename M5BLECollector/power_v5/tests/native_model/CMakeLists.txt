cmake_minimum_required(VERSION 3.20)
project(v4_micro_replay C CXX)
option(FALL_ESP_NN_PORTABLE "Use the ESP32 firmware's portable ESP-NN FC/softmax path" ON)
option(FALL_NATIVE_SPLIT_CONV "Check the exact dual-core output-row partition math" OFF)
set(CMAKE_CXX_STANDARD 17)
get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../firmware" ABSOLUTE)
set(T "${ROOT}/managed_components/espressif__esp-tflite-micro")
file(GLOB CORE "${T}/tensorflow/lite/micro/*.cc")
list(FILTER CORE EXCLUDE REGEX "(test|mock)\\.cc$")
file(GLOB ALLOC "${T}/tensorflow/lite/micro/arena_allocator/*.cc" "${T}/tensorflow/lite/micro/memory_planner/*.cc")
file(GLOB COMMON "${T}/tensorflow/lite/micro/kernels/*_common.cc")
set(KERNELS)
foreach(NAME if neg add concatenation conv depthwise_conv pad dequantize div elementwise fully_connected kernel_util mul pack quantize reduce reshape softmax squared_difference strided_slice sub transpose)
 if(FALL_ESP_NN_PORTABLE AND (NAME STREQUAL "fully_connected" OR NAME STREQUAL "softmax"))
  list(APPEND KERNELS "${T}/tensorflow/lite/micro/kernels/esp_nn/${NAME}.cc")
  continue()
 endif()
 list(APPEND KERNELS "${T}/tensorflow/lite/micro/kernels/${NAME}.cc")
endforeach()
add_executable(replay replay.cc ${CORE} ${ALLOC} ${COMMON} ${KERNELS}
 "${ROOT}/main/gelu_kernel.cc" "${ROOT}/main/layer_norm_v4.cc" "${ROOT}/main/fast_conv1d.cc" "${ROOT}/main/fall_op_resolver.cc" "${ROOT}/main/parallel_conv1d.cc"
 "${T}/tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cc"
 "${T}/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cc"
 "${T}/tensorflow/lite/core/api/flatbuffer_conversions.cc"
 "${T}/tensorflow/lite/core/api/tensor_utils.cc"
 "${T}/tensorflow/lite/core/c/common.cc"
 "${T}/tensorflow/lite/kernels/kernel_util.cc"
 "${T}/tensorflow/lite/kernels/internal/common.cc"
 "${T}/tensorflow/lite/kernels/internal/quantization_util.cc"
 "${T}/tensorflow/lite/kernels/internal/portable_tensor_utils.cc"
 "${T}/tensorflow/lite/kernels/internal/tensor_utils.cc"
 "${T}/tensorflow/lite/kernels/internal/tensor_ctypes.cc"
 "${T}/tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc"
 "${T}/tensorflow/compiler/mlir/lite/core/api/error_reporter.cc"
 "${T}/tensorflow/compiler/mlir/lite/schema/schema_utils.cc")
target_include_directories(replay PRIVATE "${ROOT}/main" "${T}" "${T}/third_party/gemmlowp" "${T}/third_party/flatbuffers/include" "${T}/third_party/ruy")
target_compile_definitions(replay PRIVATE TF_LITE_STATIC_MEMORY TF_LITE_DISABLE_X86_NEON)
if(FALL_NATIVE_SPLIT_CONV)
 target_compile_definitions(replay PRIVATE FALL_NATIVE_SPLIT_CONV=1)
endif()
target_compile_options(replay PRIVATE -O2 -ffunction-sections -fdata-sections
 "$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions;-fno-rtti>")
if(FALL_ESP_NN_PORTABLE)
 set(NN "${ROOT}/managed_components/espressif__esp-nn")
 target_sources(replay PRIVATE
  "${NN}/src/fully_connected/esp_nn_fully_connected_ansi.c"
  "${NN}/src/softmax/esp_nn_softmax_ansi.c"
  "${NN}/src/softmax/esp_nn_softmax_opt.c")
 target_include_directories(replay PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/stubs" "${NN}/include" "${NN}/src/common")
 target_compile_definitions(replay PRIVATE ESP_NN=1 CONFIG_NN_OPTIMIZED=1 CONFIG_IDF_TARGET_ESP32=1)
endif()
