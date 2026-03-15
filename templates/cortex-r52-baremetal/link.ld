/*
 * link.ld — Modified linker script for UART peripheral
 *
 * Added UART device region at 0x1C090000
 */

MEMORY
{
    ROM (rx) : ORIGIN = 0x00000000, LENGTH = 256M
    RAM (rwx) : ORIGIN = 0x10000000, LENGTH = 256M
    DEVICE (rx) : ORIGIN = 0x1C090000, LENGTH = 0x1000  @ UART peripheral
}

SECTIONS
{
    .text : {
        KEEP(*(.vectors))
        *(.text)
        *(.rodata)
    } > ROM

    .data : AT(ADDR(.text) + SIZEOF(.text))
    {
        __data_start = .;
        *(.data)
        __data_end = .;
    } > RAM

    .bss (NOLOAD) : {
        __bss_start = .;
        *(COMMON)
        *(.bss)
        __bss_end = .;
    } > RAM

    /* UART device region */
    UART_DEVICE : ORIGIN = 0x1C090000, LENGTH = 0x1000
}