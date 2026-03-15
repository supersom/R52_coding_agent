# Lazy imports to avoid circular dependency.
# Individual modules are imported on demand.

def get_config():
    from .config import ToolchainConfig
    return ToolchainConfig

def run_build(*args, **kwargs):
    from .build_system import run_build as _run_build
    return _run_build(*args, **kwargs)

def run_simulator(*args, **kwargs):
    from .simulator import run_simulator as _run_simulator
    return _run_simulator(*args, **kwargs)
