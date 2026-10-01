#pragma once
#include <cstddef>
#include <cstdint>
using spi_device_handle_t = void*;
constexpr int SPI2_HOST=2, SPI_DMA_CH_AUTO=1;
struct spi_transaction_t { size_t length; const void* tx_buffer; };
struct spi_bus_config_t { int mosi_io_num, miso_io_num, sclk_io_num, quadwp_io_num, quadhd_io_num, max_transfer_sz; };
struct spi_device_interface_config_t { int clock_speed_hz, mode, spics_io_num, queue_size; };
extern int test_transmit(const uint8_t*, size_t);
inline int spi_device_polling_transmit(void*, spi_transaction_t* t) { return test_transmit(static_cast<const uint8_t*>(t->tx_buffer),t->length/8); }
inline int spi_bus_initialize(int, const spi_bus_config_t*, int) { return 0; }
inline int spi_bus_add_device(int, const spi_device_interface_config_t*, spi_device_handle_t* handle) { *handle=reinterpret_cast<void*>(1); return 0; }
