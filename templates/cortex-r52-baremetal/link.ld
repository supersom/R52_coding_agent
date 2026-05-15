MEMORY
{
    ATCM (rx)  : ORIGIN = 0x00000000, LENGTH = 32K   /* Tightly-Coupled Memory A */
    BRAM (rwx) : ORIGIN = 0x10000000, LENGTH = 512K  /* Block RAM — stack/data */
}

SECTIONS
{
    .text : {
        KEEP(*(.vectors))
        *(.text*)
        *(.rodata*)
    } > ATCM

    .data : AT(ADDR(.text) + SIZEOF(.text))
    {
        __data_start = .;
        *(.data*)
        __data_end = .;
    } > BRAM

    .bss (NOLOAD) : {
        __bss_start = .;
        *(.bss*)
        *(COMMON)
        __bss_end = .;
    } > BRAM

    _stack_top = ORIGIN(BRAM) + LENGTH(BRAM);  /* 0x10080000 — top of BRAM */
}
