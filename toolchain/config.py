"""
Toolchain configuration — resolved from (in priority order):
  1. CLI flags
  2. Environment variables
  3. ~/.r52agent/config.toml
  4. Hard-coded defaults
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULTS = {
    "armclang_path":   "/opt/arm/developmentstudio-2025.0-1/sw/ARMCompiler6.24/bin/armclang",
    "armlink_path":    "/opt/arm/developmentstudio-2025.0-1/sw/ARMCompiler6.24/bin/armlink",
    "gnu_gcc_path":    "arm-none-eabi-gcc",
    "gnu_ld_path":     "arm-none-eabi-ld",
    "gnu_objcopy_path":"arm-none-eabi-objcopy",
    "fvp_path":        "/opt/arm/developmentstudio-2025.0-1/bin/FVP_BaseR_Cortex-R52",
    "qemu_path":       "qemu-system-arm",
    "simulator_timeout": 600,
    "default_toolchain": "gnu",
}

CONFIG_PATH = Path.home() / ".r52agent" / "config.toml"


@dataclass
class ToolchainConfig:
    armclang_path:    str = DEFAULTS["armclang_path"]
    armlink_path:     str = DEFAULTS["armlink_path"]
    gnu_gcc_path:     str = DEFAULTS["gnu_gcc_path"]
    gnu_ld_path:      str = DEFAULTS["gnu_ld_path"]
    gnu_objcopy_path: str = DEFAULTS["gnu_objcopy_path"]
    fvp_path:         str = DEFAULTS["fvp_path"]
    qemu_path:        str = DEFAULTS["qemu_path"]
    simulator_timeout: int = DEFAULTS["simulator_timeout"]
    default_toolchain: str = DEFAULTS["default_toolchain"]
    extra_cflags:     list[str] = field(default_factory=list)
    extra_ldflags:    list[str] = field(default_factory=list)

    @classmethod
    def load(cls) -> "ToolchainConfig":
        cfg = cls()
        # Load from TOML
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "rb") as f:
                data = tomllib.load(f)
            tc = data.get("toolchain", {})
            for key in DEFAULTS:
                if key in tc:
                    setattr(cfg, key, tc[key])

        # Environment variable overrides (R52_ARMCLANG_PATH, etc.)
        env_map = {
            "R52_ARMCLANG_PATH":    "armclang_path",
            "R52_ARMLINK_PATH":     "armlink_path",
            "R52_GNU_GCC_PATH":     "gnu_gcc_path",
            "R52_FVP_PATH":         "fvp_path",
            "R52_QEMU_PATH":        "qemu_path",
            "R52_SIMULATOR_TIMEOUT":"simulator_timeout",
        }
        for env_var, attr in env_map.items():
            val = os.environ.get(env_var)
            if val:
                setattr(cfg, attr, int(val) if attr == "simulator_timeout" else val)

        return cfg

    def save(self) -> None:
        """Persist current config to ~/.r52agent/config.toml."""
        import tomli_w
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "toolchain": {
                "armclang_path":    self.armclang_path,
                "armlink_path":     self.armlink_path,
                "gnu_gcc_path":     self.gnu_gcc_path,
                "fvp_path":         self.fvp_path,
                "qemu_path":        self.qemu_path,
                "simulator_timeout": self.simulator_timeout,
                "default_toolchain": self.default_toolchain,
            }
        }
        with open(CONFIG_PATH, "wb") as f:
            tomli_w.dump(data, f)
