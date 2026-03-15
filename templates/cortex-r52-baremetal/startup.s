/*
 * startup.s — Modified startup for Cortex-R52 with UART support
 *
 * Changes:
 * 1. Added UART vector entry
 * 2. Enabled UART in reset handler
 */

.syntax unified
.cpu cortex-r52
.arch armv8-r

.text
.section .vectors, "ax"
.global _start
_start:
    b reset_handler
    b undef_handler
    b svc_handler
    b prefetch_handler
    b data_abort_handler
    b irq_handler
    b fiq_handler
    b reset_handler

/* UART vector entry */
.weak fiq_handler
.type fiq_handler, %function
fiq_handler:
    b UART_IRQHandler

/* UART initialization in reset handler */
reset_handler:
    /* Existing stack setup and FPU enable code */

    /* Enable UART peripheral */
    mov r0, #0x1C090000
    ldr r1, [r0]
    orr r1, r1, #0x1  @ Enable UART
    str r1, [r0]

    /* Call main() */
    bl main
    b .

/* Weak handlers */
.weak undef_handler
.weak svc_handler
.weak prefetch_handler
.weak data_abort_handler
.weak irq_handler
.weak fiq_handler