CROSS   ?= arm-none-eabi-
CC      := $(CROSS)gcc
LD      := $(CROSS)ld
OBJCOPY := $(CROSS)objcopy

TARGET   := firmware
C_SRCS   := main.c uart.c
ASM_SRCS := startup.s
OBJS     := $(C_SRCS:.c=.o) $(ASM_SRCS:.s=.o)

CFLAGS := -mcpu=cortex-r52 -marm -O2 -nostdlib -c

all: $(TARGET).elf

$(TARGET).elf: $(OBJS)
	$(LD) -T link.ld -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -o $@ $<

%.o: %.s
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET).elf
