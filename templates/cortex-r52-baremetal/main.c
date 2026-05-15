#include <stdint.h>
#include "uart.h"

int main(void) {
    uart_init();

    const char *msg = "Hello World!\r\n";
    for (const char *p = msg; *p; p++) {
        uart_putc(*p);
    }

    while (1) {}
    return 0;
}
