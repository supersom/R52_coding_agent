/*
 * main.c — Cortex-R52 bare-metal application
 *
 * Replaces semihosting with UART output
 */

#include "uart.h"

void UART_Init(void);
void UART_Putc(char c);

export uint32_t uart_base;
export uint32_t baud_rate;

int main(void) {
    UART_Init();

    UART_Putc('H');
    UART_Putc('e');
    UART_Putc('l');
    UART_Putc('l');
    UART_Putc('o');
    UART_Putc(' ');
    UART_Putc('W');
    UART_Putc('o');
    UART_Putc('r');
    UART_Putc('l');
    UART_Putc('d');
    UART_Putc('!');

    while (1);
    return 0;
}

/* Exported symbols */
export uart_base = (void *)0x1C090000;
export baud_rate = 115200;