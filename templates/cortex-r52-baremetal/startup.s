.syntax unified
.arch armv8-r
.arm

.section .vectors, "ax"
.align 5
.global _start
_start:
    b reset_handler         /* 0x00 Reset                 */
    b undef_handler         /* 0x04 Undefined Instruction */
    b svc_handler           /* 0x08 SVC                   */
    b prefetch_handler      /* 0x0C Prefetch Abort        */
    b data_abort_handler    /* 0x10 Data Abort            */
    b .                     /* 0x14 Reserved/HVC          */
    b irq_handler           /* 0x18 IRQ                   */
    b fiq_handler           /* 0x1C FIQ                   */

.section .text, "ax"
.global reset_handler
reset_handler:
    ldr r0, =_stack_top
    mov sp, r0
    bl main
.L_hang:
    wfe
    b .L_hang

.weak undef_handler
undef_handler:
    b .

.weak svc_handler
svc_handler:
    b .

.weak prefetch_handler
prefetch_handler:
    b .

.weak data_abort_handler
data_abort_handler:
    b .

.weak irq_handler
irq_handler:
    b .

.weak fiq_handler
fiq_handler:
    b .
