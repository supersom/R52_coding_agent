/*
 * uart.h — UART driver interface for ARM Cortex-R52
 *
 * Exported symbols:
 * - uart_base: UART base address
 * - baud_rate: Configured baud rate
 * - UART_Init: Initialize UART peripheral
 * - UART_Putc: Send character via UART
 */

#ifndef UART_H
#define UART_H

void UART_Init(void);
void UART_Putc(char c);
export uint32_t uart_base;
export uint32_t baud_rate;

#endif /* UART_H */