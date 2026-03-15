# Feature: Semi-hosting Counter

## Description

Implement a function `count_and_print(int n)` that counts from 1 to n,
printing each value using ARM semi-hosting (SYS_WRITE0 via HLT #0xF000).

Call it from `main()` with n=5.

## Requirements

- Use only C99 and ARM intrinsics — no libc.
- Implement a minimal integer-to-string converter (no sprintf).
- Output format per line: `Count: N\n`
- Exit cleanly via SYS_EXIT semi-hosting after counting.

## Constraints

- Do NOT modify startup.s or link.ld.

## Expected Output

Count: 1
Count: 2
Count: 3
Count: 4
Count: 5
