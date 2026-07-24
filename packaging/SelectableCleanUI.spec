# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller one-folder build for the Windows portable edition."""

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


project = Path(SPECPATH).parent
datas = [
    (str(project / "segmentationfolder"), "segmentationfolder"),
    (str(project / "pointsdata" / "LowerJawScans.ply"), "pointsdata"),
    (str(project / "pointsdata" / "原始牙模 LowerJawScan.ply"), "pointsdata"),
    (str(project / "pointsdata" / "UncleanLowerJawScan.ply"), "pointsdata"),
    (str(project / "pointsdata" / "IO9-3 LowerJawScan.ply"), "pointsdata"),
]
datas += collect_data_files("open3d")
vtk_lib_dir = Path(sys.prefix) / "Lib" / "site-packages" / "vtk.libs"
vtk_load_order = vtk_lib_dir / ".load-order-vtk-9.3.1"
if vtk_load_order.is_file():
    datas.append((str(vtk_load_order), "vtk.libs"))

binaries = []
binaries += collect_dynamic_libs("open3d")
binaries += [(str(path), "vtk.libs") for path in vtk_lib_dir.glob("*.dll")]

hiddenimports = [
    "PyQt5.sip",
    "pyvistaqt",
    "vtk",
    "vtkmodules",
    "vtkmodules.qt.QVTKRenderWindowInteractor",
]

excluded = [
    "torch",
    "tensorflow",
    "tensorboard",
    "pandas",
    "IPython",
    "jupyter",
    "jupyter_server",
    "notebook",
    "ipywidgets",
    "skimage",
    "trame",
    "panel",
    "tkinter",
]

a = Analysis(
    [str(project / "SelectableCleanUI.py")],
    pathex=[str(project)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(project / "packaging" / "rthook_vtk_dll_path.py")],
    excludes=excluded,
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="UClean清洁度分析",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="UClean清洁度分析",
)
