"""
platform_utils.py — Runtime platform detection for DJ-R3X.

Returns 'macos_silicon' on Apple Silicon Macs (darwin + arm64), and 'pi'
for everything else (Raspberry Pi, Linux x86, CI, etc.).
"""

from __future__ import annotations

import platform
import sys


def get_platform() -> str:
    """Return 'macos_silicon' on Apple Silicon macOS, 'pi' everywhere else."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "macos_silicon"
    return "pi"
