"""Suppress harmless langchain pydantic v1 compat warning on Python 3.14."""
import warnings
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality",
    category=UserWarning,
)
