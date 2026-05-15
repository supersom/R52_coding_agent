/* PL011 UART driver for FVP_BaseR_Cortex-R52 (UART0 at 0x1C090000) */
#include <stdint.h>
#include "uart.h"

#define UART_BASE  0x1C090000UL

#define UART_DR   (*(volatile uint32_t *)(UART_BASE + 0x00))
#define UART_FR   (*(volatile uint32_t *)(UART_BASE + 0x18))
#define UART_IBRD (*(volatile uint32_t *)(UART_BASE + 0x24))
#define UART_FBRD (*(volatile uint32_t *)(UART_BASE + 0x28))
#define UART_LCRH (*(volatile uint32_t *)(UART_BASE + 0x2C))
#define UART_CR   (*(volatile uint32_t *)(UART_BASE + 0x30))

#define FR_TXFF      (1u << 5)
#define LCRH_FEN     (1u << 4)
#define LCRH_WLEN_8  (3u << 5)
#define CR_UARTEN    (1u << 0)
#define CR_TXE       (1u << 8)
#define CR_RXE       (1u << 9)

#define DSB() __asm__ volatile("dsb sy" ::: "memory")

void uart_init(void) {
    UART_CR = 0;
    DSB();
    /* 115200 baud @ 24 MHz: IBRD=13, FBRD=1 */
    UART_IBRD = 13;
    UART_FBRD = 1;
    UART_LCRH = LCRH_WLEN_8 | LCRH_FEN;
    DSB();
    UART_CR = CR_UARTEN | CR_TXE | CR_RXE;
    DSB();
}

void uart_putc(char c) {
    while (UART_FR & FR_TXFF) {}
    UART_DR = (uint32_t)c;
    DSB();
}
