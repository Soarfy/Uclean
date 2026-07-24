"""Make wheel-bundled VTK DLLs discoverable in frozen Windows builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []
if sys.platform == "win32":
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    for relative in ("vtk.libs", "open3d", "open3d/cpu"):
        dll_dir = bundle_root / relative
        if dll_dir.is_dir():
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(dll_dir)))

