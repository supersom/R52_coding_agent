#
# Makefile — Cortex-R52 bare-metal project with UART
#
CROSS ?= arm-none-eabi-
CC := $(CROSS)gcc
LD := $(CROSS)ld
OBJCOPY := $(CROSS)objcopy

TARGET := firmware
SRC := startup.s main.c uart.c
OBJS := $(SRC:.c=.o) $(SRC:.s=.o)

all: $(TARGET).elf

$(TARGET).elf: $(OBJS)
    $(LD) -T link.ld -nostdlib -o $@ $^

%.o: %.c
    $(CC) -mcpu=cortex-r52 -marm -O2 -c -o $@ $<

%.o: %.s
    $(CC) -mcpu=cortex-r52 -marm -O2 -c -o $@ $<

clean:
    rm -f $(OBJS) $(TARGET).elf