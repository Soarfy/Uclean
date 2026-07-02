# -*- coding: utf-8 -*-
"""
交互式 Mesh 分割 + 清洁度评估

工作流：
  1. 加载原始 Mesh，Shift+左/右键笔刷选区并标注
  2. 「分割并评估清洁度」：各区域导出为独立子 mesh PLY（可供 CleanUI 后续使用）
  3. 逐区打开原有颜色选点清洁度窗口；Esc = 跳过不参与
  4. 汇总显示清洁度结果（与 CleanUI.py 一致的灰度加权面积比公式）

清洁度公式（与 CleanUI.py 一致）：
  cleanliness = (remaining_gray_mean × remaining_area) / (picked_gray_mean × segment_area)
  - remaining_gray_mean: 剩余（清洁）面的平均灰度
  - picked_gray_mean: 整个分区的平均灰度
  - remaining_area: 剩余清洁面的面积
  - segment_area: 分区总面积
"""

import json
import os
import sys
import traceback

import numpy as np
import pyvista as pv
import open3d as o3d
from pyvistaqt import QtInteractor
from PyQt5.QtCore import Qt, QEventLoop
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QMessageBox, QFileDialog,
    QDoubleSpinBox, QFormLayout, QInputDialog, QListWidget, QListWidgetItem,
    QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
)

try:
    from GeneratePathOffset66initAllnewdorobotsnew import AdvancedCleanerWindow
    _BACKEND_OK = True
    _BACKEND_ERR = ""
except Exception as _e:
    AdvancedCleanerWindow = None
    _BACKEND_OK = False
    _BACKEND_ERR = traceback.format_exc()

_PALETTE = np.array([
    [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
    [255, 127, 0], [255, 255, 51], [166, 86, 40], [247, 129, 191],
    [153, 153, 153], [26, 188, 156], [241, 196, 15], [142, 68, 173],
    [52, 152, 219], [231, 76, 60], [46, 204, 113], [211, 84, 0],
], dtype=np.uint8)

_SCRIBBLE_COLOR = np.array([255, 220, 0], dtype=np.uint8)  # 当前笔刷选区：黄色


def _triangulate_mesh(mesh: pv.PolyData) -> pv.PolyData:
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    return mesh.extract_surface().clean()


def _rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """
    将每个面的 RGB 颜色 (n, 3) 转为灰度值 (n,)。
    颜色越黑值越大，白色=0，黑色=1。
    """
    gray = 1.0 - (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114)
    return np.clip(gray, 0.0, 1.0)


def _compute_face_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """逐三角面片计算面积，返回 (n_faces,)"""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    ab, ac = v1 - v0, v2 - v0
    cross = np.cross(ab, ac)
    return np.linalg.norm(cross, axis=1) * 0.5


def _face_colors_from_mesh(mesh: pv.PolyData) -> np.ndarray:
    """每个三角面的灰度值 (n_faces,)：黑色=1.0，白色=0.0"""
    n = mesh.n_cells
    for data in (mesh.cell_data, mesh.point_data):
        for name in data.keys():
            arr = data[name]
            if arr.ndim == 2 and arr.shape[1] in (3, 4):
                if arr.dtype == np.uint8 or name.lower() in ("rgb", "rgba", "colors", "color"):
                    on_cell = data is mesh.cell_data
                    if on_cell:
                        c = np.asarray(arr[:, :3], dtype=np.float64)
                        if c.max() > 1.0:
                            c /= 255.0
                    else:
                        pts = np.asarray(arr[:, :3], dtype=np.float64)
                        if pts.max() > 1.0:
                            pts /= 255.0
                        ids = mesh.faces.reshape(-1, 4)[:, 1:4]
                        c = pts[ids].mean(axis=1)
                    return _rgb_to_grayscale(c)

    return np.full(n, 0.25, dtype=np.float64)


def _face_rgb_from_mesh(mesh: pv.PolyData) -> np.ndarray:
    """每个三角面的 RGB 颜色 (n_faces, 3), float 0~1"""
    n = mesh.n_cells
    rgb_name = None
    for data in (mesh.cell_data, mesh.point_data):
        for name in data.keys():
            arr = data[name]
            if arr.ndim == 2 and arr.shape[1] in (3, 4):
                if arr.dtype == np.uint8 or name.lower() in ("rgb", "rgba", "colors", "color"):
                    rgb_name = (data is mesh.cell_data, name)
                    break
        if rgb_name:
            break

    if rgb_name is None:
        return np.full((n, 3), 0.75, dtype=np.float64)

    on_cell, name = rgb_name
    if on_cell:
        c = np.asarray(mesh.cell_data[name][:, :3], dtype=np.float64)
        if c.max() > 1.0:
            c /= 255.0
        return c

    pts = np.asarray(mesh.point_data[name][:, :3], dtype=np.float64)
    if pts.max() > 1.0:
        pts /= 255.0
    ids = mesh.faces.reshape(-1, 4)[:, 1:4]
    return pts[ids].mean(axis=1)


def _mesh_edge_length_stats(mesh: pv.PolyData):
    """
    根据网格三角边长度估计模型空间尺度。
    min / p5 近似「最近两点距离」量级，用于自适应笔刷大小。
    """
    pts = np.asarray(mesh.points, dtype=np.float64)
    faces = mesh.faces.reshape(-1, 4)[:, 1:4]
    v0, v1, v2 = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    edges = np.concatenate([
        np.linalg.norm(v1 - v0, axis=1),
        np.linalg.norm(v2 - v1, axis=1),
        np.linalg.norm(v0 - v2, axis=1),
    ])
    edges = edges[np.isfinite(edges) & (edges > 1e-12)]
    bounds = mesh.bounds
    diag = float(np.linalg.norm([
        bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4],
    ]))
    if len(edges) == 0:
        return {"min": diag * 0.001, "p5": diag * 0.001, "median": diag * 0.01,
                "max": diag * 0.01, "diag": max(diag, 1e-6)}
    return {
        "min": float(np.min(edges)),
        "p5": float(np.percentile(edges, 5)),
        "median": float(np.median(edges)),
        "max": float(np.max(edges)),
        "diag": max(diag, 1e-6),
    }


def _submesh_from_faces(base_mesh, face_indices):
    cells = np.array(sorted(face_indices), dtype=np.int64)
    return base_mesh.extract_cells(cells).extract_surface()


def _point_colors_from_submesh(sub, face_colors, orig_face_indices):
    """
    为子 mesh 每个顶点生成 RGB（0~1），优先顶点色，
    否则由面色平均。face_colors 可为 RGB (n,3) 或灰度 (n,)。
    """
    n_pts = sub.n_points
    for name in sub.point_data.keys():
        arr = sub.point_data[name]
        if arr.ndim == 2 and arr.shape[1] >= 3:
            c = np.asarray(arr[:, :3], dtype=np.float64)
            if c.max() > 1.0:
                c /= 255.0
            return np.clip(c, 0.0, 1.0)

    orig_ids = sub.cell_data.get("vtkOriginalCellIds")
    if orig_ids is not None and face_colors is not None:
        face_color = np.asarray(face_colors, dtype=np.float64)[orig_ids.astype(np.int64)]
    elif face_colors is not None:
        cells = np.array(sorted(orig_face_indices), dtype=np.int64)
        face_color = np.asarray(face_colors, dtype=np.float64)[cells]
    else:
        return np.full((n_pts, 3), 0.75, dtype=np.float64)

    faces = sub.faces.reshape(-1, 4)[:, 1:4]
    acc = np.zeros(n_pts, dtype=np.float64)
    cnt = np.zeros(n_pts, dtype=np.float64)
    for fi, tri in enumerate(faces):
        for vi in tri:
            acc[vi] += face_color[fi]
            cnt[vi] += 1.0
    cnt = np.maximum(cnt, 1.0)

    is_grayscale = face_color.ndim == 1
    if is_grayscale:
        gray = acc / cnt
        gray = np.clip(gray, 0.0, 1.0)
        return np.stack([gray, gray, gray], axis=1)

    acc3 = np.zeros((n_pts, 3), dtype=np.float64)
    for fi, tri in enumerate(faces):
        for vi in tri:
            acc3[vi] += face_color[fi]
    return np.clip(acc3 / cnt[:, None], 0.0, 1.0)


def _submesh_surface_area(sub):
    try:
        faces = sub.faces.reshape(-1, 4)[:, 1:4]
        verts = np.asarray(sub.points, dtype=np.float64)
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        return float(o3d_mesh.get_surface_area())
    except Exception:
        return 0.0


def _compute_segment_cleanliness_on_mesh(
    face_colors,
    seg_face_indices,
    remaining_point_indices,
    submesh
):
    """
    Compute cleanliness for a segment mesh using the same formula as CleanUI.py:
      cleanliness = (remaining_gray_mean × remaining_area) / (picked_gray_mean × segment_area)

    Where:
      - remaining_gray_mean: average grayscale of remaining (clean) faces
      - picked_gray_mean: average grayscale of all selected faces in this segment
      - remaining_area: surface area of remaining/clean faces
      - segment_area: total surface area of the segment

    remaining_point_indices: vertex indices in the submesh's vertex array that are "remaining/clean"
    submesh: pyvista PolyData of the segment
    face_colors: (n_faces,) grayscale of the ORIGINAL base mesh, indexed by original face indices
    seg_face_indices: original face indices of this segment
    """
    verts = np.asarray(submesh.points, dtype=np.float64)
    sub_faces = submesh.faces.reshape(-1, 4)[:, 1:4]  # (n_sub_faces, 3) - local vertex indices

    sub_areas = _compute_face_areas(verts, sub_faces)
    sub_grays = np.asarray(face_colors, dtype=np.float64)[
        np.array(sorted(seg_face_indices), dtype=np.int64)
    ]

    rem_set = set(int(i) for i in remaining_point_indices)
    remaining_mask = np.array([
        all(int(v) in rem_set for v in tri)
        for tri in sub_faces
    ])

    # Calculate mean grayscale for remaining faces (clean area)
    if remaining_mask.any():
        remaining_gray_mean = float(np.mean(sub_grays[remaining_mask]))
    else:
        remaining_gray_mean = 0.0

    # Calculate mean grayscale for all faces (picked area = whole segment)
    picked_gray_mean = float(np.mean(sub_grays)) if len(sub_grays) > 0 else 0.0

    # Calculate areas
    remaining_area = float(np.sum(sub_areas[remaining_mask])) if remaining_mask.any() else 0.0
    segment_area = float(np.sum(sub_areas))

    # Apply CleanUI.py formula: (remaining_gray_mean × remaining_area) / (picked_gray_mean × segment_area)
    numerator = remaining_gray_mean * remaining_area
    denominator = picked_gray_mean * segment_area if segment_area > 0 else 0.0

    cleanliness = numerator / denominator if denominator > 0 else 0.0

    return {
        "cleanliness": cleanliness,
        "numerator": numerator,
        "denominator": denominator,
        "remaining_gray_mean": remaining_gray_mean,
        "picked_gray_mean": picked_gray_mean,
        "remaining_area": remaining_area,
        "segment_area": segment_area,
    }


def _export_segment_mesh(base_mesh, face_indices, face_colors, path):
    """将选中区域导出为带顶点颜色的三角 mesh PLY，并返回表面积（mm²）和子 mesh 对象"""
    sub = _submesh_from_faces(base_mesh, face_indices)
    verts = np.asarray(sub.points, dtype=np.float64)
    faces = sub.faces.reshape(-1, 4)[:, 1:4].astype(np.int32)
    colors = _point_colors_from_submesh(sub, face_colors, face_indices)
    area = _submesh_surface_area(sub)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    o3d.io.write_triangle_mesh(path, o3d_mesh)
    return path, area, sub


def _enrich_segment_stat(result, segment_name, segment_area=0.0, numerator=0.0, denominator=0.0):
    """
    根据选点分割结果计算清洁度统计。
    与 CleanUI.py 一致的公式:
      cleanliness = (remaining_gray_mean × remaining_area) / (picked_gray_mean × segment_area)
    """
    area = float(segment_area or 0)
    cleanliness = float(result.get("cleanliness", 0))
    if cleanliness <= 0 and denominator > 0:
        cleanliness = numerator / denominator

    remaining_gray_mean = float(result.get("remaining_gray_mean", 0))
    picked_gray_mean = float(result.get("picked_gray_mean", 0))
    remaining_area = float(result.get("remaining_area", 0))
    segment_area = float(result.get("segment_area", area))

    # dirty_area = segment_area - remaining_area
    dirty_area = segment_area - remaining_area

    return {
        "name": segment_name,
        "display_name": segment_name,
        "original_area": segment_area,
        "cleanliness": cleanliness * 100.0,  # Convert to percentage
        "cleaned_area": max(0.0, dirty_area),
        "remaining_area": remaining_area,
        "remaining_gray_mean": remaining_gray_mean,
        "picked_gray_mean": picked_gray_mean,
        "numerator": numerator,
        "denominator": denominator,
    }


if _BACKEND_OK:
    class SkippableCleanerWindow(AdvancedCleanerWindow):
        """包装原有清洁度窗口：Esc 跳过，保存时额外记录剩余顶点索引"""

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Escape:
                self.result_data = None
                self.close()
                return
            super().keyPressEvent(event)

        def _save_to_disk(self):
            """Override to capture remaining vertex indices before calling parent"""
            self._pending_remaining_indices = self.get_remaining_point_indices()
            super()._save_to_disk()

        def get_remaining_point_indices(self):
            """返回当前选点状态下"剩余（清洁）"的顶点索引"""
            keep_mask = ~self._get_selection_mask()
            return np.where(keep_mask)[0]


def _run_cleanliness_dialog(mesh_path, output_dir, color_threshold=12.0):
    """
    对分区 mesh 调用原有 AdvancedCleanerWindow；
    保存 → 返回 {remaining_point_indices, total_point_count}（用于面片级清洁度计算）
    Esc / 直接关闭 → 返回 None（跳过）
    """
    if not _BACKEND_OK:
        raise RuntimeError("未能导入清洁度后端模块\n" + _BACKEND_ERR)
    app = QApplication.instance() or QApplication(sys.argv)
    window = SkippableCleanerWindow(mesh_path, output_dir, color_threshold)
    if not hasattr(window, "points"):
        window.close()
        return None
    window.setWindowTitle(f"清洁度选点 — {os.path.basename(mesh_path)}  (Esc=跳过)")
    window.show()
    loop = QEventLoop()
    window._wait_loop = loop
    loop.exec_()

    if window.result_data is None:
        return None

    rem_indices = getattr(window, "_pending_remaining_indices", None)
    if rem_indices is None:
        rem_indices = window.get_remaining_point_indices()

    return {
        "remaining_point_indices": rem_indices,
        "total_point_count": len(window.points),
    }


class MeshSegmentationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_path = None
        self.base_mesh = None          # 原始/工作 mesh
        self.face_colors = None        # (n_faces,) 灰度值：黑色=1.0，白色=0.0
        self.face_rgb = None           # (n_faces, 3) RGB 原色，仅用于显示
        self.assigned = None           # bool[n_faces]，已被某类别占用

        self.scribble_faces = set()    # 当前笔刷选中的面
        self.labeled_segments = []     # [{name, faces:set}, ...]
        self.cleanliness_stats = []

        self._scribbling = False
        self._picker = None
        self._saved_interactor_style = None
        self._user_style = None
        self._face_centers = None
        self._face_tree = None
        self.mesh_scale = None
        self.tool_mode = "brush"  # brush | eraser

        self.setWindowTitle("Mesh 分割 + 清洁度评估")
        self.resize(1360, 920)
        self.setFocusPolicy(Qt.StrongFocus)
        self._init_ui()
        self._setup_scribble_events()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top_row = QHBoxLayout()
        panel = QWidget()
        panel.setFixedWidth(320)
        pl = QVBoxLayout(panel)

        title = QLabel("交互涂抹分割")
        title.setStyleSheet("font-weight:bold; font-size:16px; color:#0078d7;")
        pl.addWidget(title)

        btn_style = "padding:9px; font-size:12px; margin-bottom:3px;"

        self.btn_load = QPushButton("1) 加载原始 Mesh")
        self.btn_load.setStyleSheet(btn_style + "background:#0078d7;color:white;")
        self.btn_load.clicked.connect(self.load_mesh)
        pl.addWidget(self.btn_load)

        self.lbl_file = QLabel("未加载")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("color:#555;font-size:11px;")
        pl.addWidget(self.lbl_file)

        # --- 涂抹分割 ---
        grp_scribble = QGroupBox("鼠标涂抹 → 标注")
        gs = QVBoxLayout(grp_scribble)

        form = QFormLayout()
        self.spin_tool_diameter = QDoubleSpinBox()
        self.spin_tool_diameter.setRange(1e-6, 1e6)
        self.spin_tool_diameter.setSingleStep(0.1)
        self.spin_tool_diameter.setDecimals(4)
        self.spin_tool_diameter.setValue(1.0)
        self.spin_tool_diameter.setSuffix(" (模型单位)")
        form.addRow("笔刷/橡皮直径:", self.spin_tool_diameter)

        self.lbl_mesh_scale = QLabel("网格尺度: 加载模型后显示")
        self.lbl_mesh_scale.setWordWrap(True)
        self.lbl_mesh_scale.setStyleSheet("color:#666;font-size:10px;")
        form.addRow("", self.lbl_mesh_scale)
        gs.addLayout(form)

        tool_row = QHBoxLayout()
        self.btn_brush = QPushButton("笔刷 (B)")
        self.btn_brush.setCheckable(True)
        self.btn_brush.setChecked(True)
        self.btn_brush.clicked.connect(lambda: self.set_tool_mode("brush"))
        self.btn_eraser = QPushButton("橡皮擦 (D)")
        self.btn_eraser.setCheckable(True)
        self.btn_eraser.clicked.connect(lambda: self.set_tool_mode("eraser"))
        tool_row.addWidget(self.btn_brush)
        tool_row.addWidget(self.btn_eraser)
        gs.addLayout(tool_row)

        self.lbl_tool = QLabel("当前工具: 笔刷")
        self.lbl_tool.setStyleSheet("font-weight:bold;color:#c0392b;")
        gs.addWidget(self.lbl_tool)

        self.chk_scribble = QCheckBox("涂抹模式 (Shift + 左/右键)")
        self.chk_scribble.setChecked(True)
        self.chk_scribble.setStyleSheet("font-weight:bold; color:#c0392b;")
        self.chk_scribble.toggled.connect(self._apply_trackball_mode)
        gs.addWidget(self.chk_scribble)

        self.btn_clear_scribble = QPushButton("清除当前选区")
        self.btn_clear_scribble.clicked.connect(self.clear_scribble)
        gs.addWidget(self.btn_clear_scribble)

        self.btn_label = QPushButton("2) 标注类别并保存该分区")
        self.btn_label.setStyleSheet(btn_style + "background:#f0ad4e;color:white;font-weight:bold;")
        self.btn_label.clicked.connect(self.confirm_label)
        gs.addWidget(self.btn_label)

        self.list_segments = QListWidget()
        self.list_segments.setMaximumHeight(140)
        gs.addWidget(QLabel("已标注分区:"))
        gs.addWidget(self.list_segments)

        self.btn_clean = QPushButton("3) 分割并评估清洁度")
        self.btn_clean.setStyleSheet(btn_style + "background:#28a745;color:white;font-weight:bold;")
        self.btn_clean.clicked.connect(self.run_cleanliness_workflow)
        gs.addWidget(self.btn_clean)

        self.btn_export = QPushButton("4) 仅导出分区 Mesh")
        self.btn_export.clicked.connect(self.export_labeled)
        gs.addWidget(self.btn_export)

        if not _BACKEND_OK:
            gs.addWidget(QLabel("⚠ 清洁度后端未加载"))

        tip = QLabel(
            "操作：\n"
            "· B = 笔刷，D = 橡皮擦\n"
            "· Shift + 左键 或 Shift + 右键 = 涂抹选区\n"
            "· 不按 Shift：左键旋转，右键/滚轮缩放\n"
            "· 涂好后点「标注类别并保存」\n"
            "· 点「分割并评估清洁度」：先导出分区 mesh，再逐区选点\n"
            "· 清洁度窗口按 Esc = 跳过该区域")
        tip.setStyleSheet("color:#444;font-size:10px;line-height:1.4;")
        gs.addWidget(tip)
        pl.addWidget(grp_scribble)

        self.lbl_info = QLabel("状态: 请加载 Mesh")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("font-weight:bold;color:#0078d7;margin-top:6px;")
        pl.addWidget(self.lbl_info)

        pl.addStretch()
        top_row.addWidget(panel)

        self.plotter = QtInteractor(self)
        top_row.addWidget(self.plotter)
        self.plotter.set_background([0.2, 0.2, 0.2])
        root.addLayout(top_row, stretch=3)

        root.addWidget(QLabel("清洁度评估结果（与 CleanUI.py 一致的灰度加权面积比公式）："))
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["分区", "分区面积 (mm²)", "清洁度 (%)", "污染面积 (mm²)", "相对总面贡献"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMaximumHeight(220)
        root.addWidget(self.table, stretch=1)

    def _get_vtk_iren(self):
        """兼容 PyVista 包装器与原生 VTK interactor"""
        iren = self.plotter.iren
        if hasattr(iren, "interactor"):
            return iren.interactor
        rw = self.plotter.render_window
        if rw is not None and hasattr(rw, "GetInteractor"):
            return rw.GetInteractor()
        return iren

    def _get_event_position(self):
        iren = self.plotter.iren
        if hasattr(iren, "get_event_position"):
            return iren.get_event_position()
        return self._get_vtk_iren().GetEventPosition()

    def _apply_brush_scale_from_mesh(self, mesh):
        """按网格最短边 / 中位边长自适应笔刷范围与默认值"""
        stats = _mesh_edge_length_stats(mesh)
        self.mesh_scale = stats

        min_edge = stats["min"]
        p5 = stats["p5"]
        median = stats["median"]
        diag = stats["diag"]

        # 默认笔刷 ≈ 6 倍「稳健最短边」(p5)，大约覆盖数个三角面
        default_d = max(p5 * 6.0, min_edge * 4.0)
        default_d = min(default_d, median * 8.0, diag * 0.08)

        spin_min = max(min_edge * 0.5, 1e-9)
        spin_max = max(spin_min * 2.0, min(diag * 0.2, median * 80.0, p5 * 400.0))
        spin_step = max(p5 * 0.5, min_edge * 0.25, 1e-6)

        self.spin_tool_diameter.blockSignals(True)
        self.spin_tool_diameter.setDecimals(4 if spin_min < 0.01 else 3 if spin_min < 1 else 2)
        self.spin_tool_diameter.setRange(spin_min, spin_max)
        self.spin_tool_diameter.setSingleStep(round(spin_step, 6))
        self.spin_tool_diameter.setValue(round(default_d, 6))
        self.spin_tool_diameter.blockSignals(False)

        self.lbl_mesh_scale.setText(
            f"最短边 {min_edge:.4g} | P5 {p5:.4g} | 中位边 {median:.4g} | 对角线 {diag:.4g}"
        )

    def set_tool_mode(self, mode):
        self.tool_mode = mode
        self.btn_brush.setChecked(mode == "brush")
        self.btn_eraser.setChecked(mode == "eraser")
        if mode == "brush":
            self.lbl_tool.setText("当前工具: 笔刷 (B)")
            self.lbl_tool.setStyleSheet("font-weight:bold;color:#c0392b;")
        else:
            self.lbl_tool.setText("当前工具: 橡皮擦 (D)")
            self.lbl_tool.setStyleSheet("font-weight:bold;color:#2980b9;")
        self.setFocus()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_B:
            self.set_tool_mode("brush")
            event.accept()
            return
        if key == Qt.Key_D:
            self.set_tool_mode("eraser")
            event.accept()
            return
        super().keyPressEvent(event)

    def _rebuild_face_spatial_index(self):
        """预计算三角面中心，并建立空间索引供笔刷直径查询"""
        if self.base_mesh is None:
            self._face_centers = None
            self._face_tree = None
            return
        self._face_centers = np.asarray(self.base_mesh.cell_centers().points, dtype=np.float64)
        try:
            from scipy.spatial import cKDTree
            self._face_tree = cKDTree(self._face_centers)
        except Exception:
            self._face_tree = None

    def _faces_in_tool_radius(self, pick_pos):
        radius = self.spin_tool_diameter.value() / 2.0
        if radius <= 0:
            return []
        if self._face_tree is not None:
            return self._face_tree.query_ball_point(pick_pos, r=radius)
        dists = np.linalg.norm(self._face_centers - pick_pos, axis=1)
        return np.where(dists <= radius)[0]

    def _is_shift_pressed(self):
        for obj in (self._get_vtk_iren(), self.plotter.iren):
            if hasattr(obj, "GetShiftKey"):
                return bool(obj.GetShiftKey())
        return False

    def _ensure_user_style(self):
        if self._user_style is None:
            import vtk
            self._user_style = vtk.vtkInteractorStyleUser()

    def _start_scribble_session(self):
        if self._scribbling:
            return
        self._ensure_user_style()
        vtk_iren = self._get_vtk_iren()
        try:
            if hasattr(vtk_iren, "GetInteractorStyle"):
                self._saved_interactor_style = vtk_iren.GetInteractorStyle()
                if hasattr(vtk_iren, "SetInteractorStyle"):
                    vtk_iren.SetInteractorStyle(self._user_style)
        except Exception:
            self._saved_interactor_style = None
        self._scribbling = True

    def _stop_scribble_session(self):
        if not self._scribbling:
            return
        self._scribbling = False
        vtk_iren = self._get_vtk_iren()
        try:
            if (self._saved_interactor_style is not None
                    and hasattr(vtk_iren, "SetInteractorStyle")):
                vtk_iren.SetInteractorStyle(self._saved_interactor_style)
        except Exception:
            pass
        self._saved_interactor_style = None

    def _can_paint_now(self):
        return (self.chk_scribble.isChecked()
                and self.base_mesh is not None
                and self._is_shift_pressed())

    def _apply_trackball_mode(self):
        """不按 Shift 时保持常规视角操作；Shift+左/右键由 observer 接管涂抹"""
        self.plotter.enable_trackball_style()
        self._stop_scribble_session()

    def _setup_scribble_events(self):
        """Shift + 左/右键涂抹"""
        vtk_iren = self._get_vtk_iren()
        add_obs = getattr(self.plotter.iren, "add_observer", None)
        if add_obs is None:
            add_obs = vtk_iren.AddObserver

        def on_button_press(_obj, _evt):
            if not self._can_paint_now():
                return
            self._start_scribble_session()
            self._paint_at_cursor()

        def on_move(_obj, _evt):
            if self._scribbling and self._can_paint_now():
                self._paint_at_cursor()
            elif self._scribbling:
                self._stop_scribble_session()

        def on_button_release(_obj, _evt):
            self._stop_scribble_session()

        add_obs("LeftButtonPressEvent", on_button_press)
        add_obs("RightButtonPressEvent", on_button_press)
        add_obs("MouseMoveEvent", on_move)
        add_obs("LeftButtonReleaseEvent", on_button_release)
        add_obs("RightButtonReleaseEvent", on_button_release)
        self._apply_trackball_mode()

    def _ensure_picker(self):
        if self._picker is None:
            import vtk
            self._picker = vtk.vtkCellPicker()
            self._picker.SetTolerance(0.005)

    def _paint_at_cursor(self):
        if self.base_mesh is None:
            return
        self._ensure_picker()
        x, y = self._get_event_position()
        self._picker.Pick(x, y, 0, self.plotter.renderer)
        if self._picker.GetCellId() < 0:
            return

        pick_pos = np.array(self._picker.GetPickPosition(), dtype=np.float64)
        face_ids = self._faces_in_tool_radius(pick_pos)
        if len(face_ids) == 0:
            return

        changed = False
        if self.tool_mode == "brush":
            for cid in face_ids:
                cid = int(cid)
                if self.assigned[cid]:
                    continue
                if cid not in self.scribble_faces:
                    self.scribble_faces.add(cid)
                    changed = True
        else:  # eraser
            for cid in face_ids:
                cid = int(cid)
                if cid in self.scribble_faces:
                    self.scribble_faces.discard(cid)
                    changed = True

        if changed:
            self._refresh_view()

    # ---------------- 加载 ----------------
    def load_mesh(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择原始 Mesh", "", "Mesh Files (*.ply *.obj *.off *.stl)")
        if not path:
            return
        try:
            mesh = _triangulate_mesh(pv.read(path))
            self.input_path = path
            self.base_mesh = mesh
            self.face_rgb = _face_rgb_from_mesh(mesh)
            self.face_colors = _face_colors_from_mesh(mesh)
            self.assigned = np.zeros(mesh.n_cells, dtype=bool)
            self._rebuild_face_spatial_index()
            self._apply_brush_scale_from_mesh(mesh)

            self.scribble_faces.clear()
            self.labeled_segments.clear()
            self.list_segments.clear()

            self.lbl_file.setText(os.path.basename(path))
            self.lbl_info.setText(f"已加载: {mesh.n_cells} 面, {mesh.n_points} 点")
            self._refresh_view(reset_camera=True)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def _display_colors(self):
        """合成显示色：原色(RGB) + 已标注色 + 当前笔刷选区"""
        base = (np.clip(self.face_rgb, 0, 1) * 255).astype(np.uint8)

        for i, seg in enumerate(self.labeled_segments):
            col = _PALETTE[i % len(_PALETTE)]
            for fi in seg["faces"]:
                base[fi] = col

        for fi in self.scribble_faces:
            if not self.assigned[fi]:
                base[fi] = _SCRIBBLE_COLOR

        return base

    def _refresh_view(self, reset_camera=False):
        if self.base_mesh is None:
            return
        colors = self._display_colors()
        mesh = self.base_mesh.copy()
        mesh.cell_data["display_rgb"] = colors
        cam = None if reset_camera else self.plotter.camera_position
        self.plotter.clear()
        self.plotter.add_mesh(mesh, scalars="display_rgb", rgb=True,
                              show_edges=False, name="mesh", pickable=True)
        if reset_camera:
            self.plotter.reset_camera()
        elif cam is not None:
            self.plotter.camera_position = cam
        self.plotter.render()

    # ---------------- 涂抹分割 ----------------
    def clear_scribble(self):
        self.scribble_faces.clear()
        self._refresh_view()
        self.lbl_info.setText("已清除当前选区")

    def confirm_label(self):
        if not self.scribble_faces:
            QMessageBox.information(self, "提示", "请先用笔刷在模型上涂抹选区。")
            return

        faces = {f for f in self.scribble_faces if not self.assigned[f]}
        if not faces:
            QMessageBox.information(self, "提示", "选区落在已标注区域上，请重新涂抹。")
            return

        default = f"region_{len(self.labeled_segments) + 1}"
        name, ok = QInputDialog.getText(
            self, "标注类别",
            "请输入该分区的类别名称\n(例如: outsideface / tips / insideface):",
            text=default)
        if not ok or not name.strip():
            return
        name = name.strip()

        self.labeled_segments.append({"name": name, "faces": faces})
        for fi in faces:
            self.assigned[fi] = True

        item = QListWidgetItem(f"{name}  ({len(faces)} 面)")
        self.list_segments.addItem(item)

        self.scribble_faces.clear()
        self._refresh_view()
        self.lbl_info.setText(
            f"已标注 [{name}]，共 {len(self.labeled_segments)} 个分区，"
            f"剩余未标注 {int((~self.assigned).sum())} 面")

    def export_labeled(self):
        if not self.labeled_segments or self.base_mesh is None:
            QMessageBox.information(self, "提示", "还没有已标注的分区。")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not out_dir:
            return
        self._export_all_segments(out_dir, show_msg=True)

    def _export_all_segments(self, out_dir, show_msg=False):
        base = os.path.splitext(os.path.basename(self.input_path or "mesh"))[0]
        seg_dir = os.path.join(out_dir, "segments")
        os.makedirs(seg_dir, exist_ok=True)
        manifest = []
        exports = {}
        for seg in self.labeled_segments:
            fname = f"{base}_{seg['name']}.ply"
            fpath = os.path.join(seg_dir, fname)
            _, area, submesh = _export_segment_mesh(
                self.base_mesh, seg["faces"], self.face_colors, fpath)
            exports[seg["name"]] = {
                "path": fpath,
                "area": area,
                "submesh": submesh,
                "seg_face_indices": list(seg["faces"]),
            }
            manifest.append({
                "name": seg["name"],
                "face_count": len(seg["faces"]),
                "faces": sorted(int(f) for f in seg["faces"]),
                "mesh": os.path.join("segments", fname),
                "surface_area": area,
            })
        with open(os.path.join(out_dir, f"{base}_segmentation.json"), "w", encoding="utf-8") as f:
            json.dump({
                "format": "mesh",
                "source_mesh": os.path.abspath(self.input_path or ""),
                "segments": manifest,
            }, f, indent=2, ensure_ascii=False)
        if show_msg:
            QMessageBox.information(
                self, "导出完成",
                f"已导出 {len(manifest)} 个分区 mesh 到:\n{seg_dir}")
        return seg_dir, exports

    def run_cleanliness_workflow(self):
        if not self.labeled_segments or self.base_mesh is None:
            QMessageBox.information(self, "提示", "请先完成手动分区并标注类别。")
            return
        if not _BACKEND_OK:
            QMessageBox.critical(self, "后端未加载",
                                 "未能导入清洁度模块 GeneratePathOffset66initAllnewdorobotsnew。\n\n"
                                 + _BACKEND_ERR)
            return

        out_dir = QFileDialog.getExistingDirectory(self, "选择输出目录（分区 mesh + 清洁度结果）")
        if not out_dir:
            return

        QMessageBox.information(
            self, "开始评估",
            "即将逐区打开「颜色选点清洁度」窗口。\n\n"
            "· 完成选点后点「保存」→ 参与统计\n"
            "· 按 Esc 或直接关闭窗口 → 跳过该区域\n"
            "清洁度 = (剩余灰度均值 × 剩余面积) / (分区灰度均值 × 分区面积)")

        self.btn_clean.setEnabled(False)
        self.table.setRowCount(0)
        self.cleanliness_stats = []
        QApplication.processEvents()

        try:
            seg_dir, segment_exports = self._export_all_segments(out_dir, show_msg=False)
            cleaned_dir = os.path.join(out_dir, "cleaned_results")
            os.makedirs(cleaned_dir, exist_ok=True)

            for i, seg in enumerate(self.labeled_segments):
                name = seg["name"]
                export_info = segment_exports[name]
                mesh_path = export_info["path"]
                self.lbl_info.setText(f"清洁度评估 ({i + 1}/{len(self.labeled_segments)}): {name}")
                QApplication.processEvents()

                picking_result = _run_cleanliness_dialog(mesh_path, cleaned_dir)
                if picking_result is None:
                    print(f"跳过区域: {name}")
                    continue

                rem_indices = picking_result["remaining_point_indices"]
                submesh = export_info["submesh"]
                seg_fi = export_info["seg_face_indices"]

                clean_result = _compute_segment_cleanliness_on_mesh(
                    self.face_colors, seg_fi, rem_indices, submesh)

                stat = _enrich_segment_stat(
                    clean_result, name,
                    segment_area=export_info.get("area", 0.0),
                    numerator=clean_result["numerator"],
                    denominator=clean_result["denominator"])
                self.cleanliness_stats.append(stat)

            if not self.cleanliness_stats:
                QMessageBox.warning(self, "完成", "没有区域参与清洁度统计（均已跳过或未保存）。")
                return

            self._update_results_table(self.cleanliness_stats)

            report_path = os.path.join(cleaned_dir, "cleanliness_report.json")
            total_area = sum(self._get_weight(s) for s in self.cleanliness_stats)
            if total_area > 0:
                total_numerator = sum(
                    float(s.get('remaining_gray_mean', 0) or 0) * float(s.get('remaining_area', 0) or 0)
                    for s in self.cleanliness_stats
                )
                total_denominator = sum(
                    float(s.get('picked_gray_mean', 0) or 0) * float(s.get('original_area', 0) or 0)
                    for s in self.cleanliness_stats
                )
                avg = (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
            else:
                avg = 0
            report = {
                "average_cleanliness": avg,
                "total_count": len(self.cleanliness_stats),
                "formula": "cleanliness = (remaining_gray_mean × remaining_area) / (picked_gray_mean × segment_area)",
                "details": sorted(self.cleanliness_stats, key=lambda x: x["cleanliness"], reverse=True),
            }
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            total_dirty = sum(s.get("cleaned_area", 0) for s in self.cleanliness_stats)
            QMessageBox.information(
                self, "评估完成",
                f"参与统计: {len(self.cleanliness_stats)} 个区域\n"
                f"分区 mesh 目录: {seg_dir}\n"
                f"清洁度报告: {report_path}\n"
                f"总牙面面积: {total_area:.4f}\n"
                f"总污染面积: {total_dirty:.4f}\n"
                f"全口加权清洁度: {avg:.2f}%")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "评估失败", str(e))
        finally:
            self.btn_clean.setEnabled(True)
            self.lbl_info.setText(f"清洁度评估结束，共 {len(self.cleanliness_stats)} 区参与")

    @staticmethod
    def _get_weight(s):
        """优先使用真实表面积加权；若旧数据缺少面积字段，退回用点数加权，避免表格空白"""
        area = float(s.get('original_area', 0) or 0)
        if area > 0:
            return area
        return float(s.get('original_points', 0) or 0)

    def _update_results_table(self, data):
        self.table.setRowCount(0)
        total_area = sum(self._get_weight(s) for s in data)
        if total_area <= 0:
            return

        sorted_data = sorted(data, key=lambda x: x.get("cleanliness", 0))

        total_denominator = sum(
            float(s.get('picked_gray_mean', 0) or 0) * float(s.get('original_area', 0) or 0)
            for s in data
        )
        overall_weighted_rate = 0
        total_dirty_area = 0

        for row, s in enumerate(sorted_data):
            self.table.insertRow(row)
            name = str(s.get("display_name", s.get("name", "Unknown")))
            area = self._get_weight(s)
            local_rate = float(s.get("cleanliness", 0))
            dirty_area = float(s.get("cleaned_area", 0))
            total_dirty_area += dirty_area

            picked_gray = float(s.get('picked_gray_mean', 0) or 0)
            remaining_gray = float(s.get('remaining_gray_mean', 0) or 0)
            remaining_area = float(s.get('remaining_area', 0) or 0)

            numerator = remaining_gray * remaining_area
            if total_denominator > 0:
                contribution = numerator / total_denominator
            else:
                contribution = 0.0

            overall_weighted_rate += contribution

            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(f"{area:.2f}"))

            rate_item = QTableWidgetItem(f"{local_rate:.2f}%")
            self._apply_score_style(rate_item, local_rate)
            self.table.setItem(row, 2, rate_item)

            self.table.setItem(row, 3, QTableWidgetItem(f"{dirty_area:.2f}"))

            contrib_item = QTableWidgetItem(f"{contribution:.2f}%")
            contrib_item.setForeground(QColor("#666666"))
            self.table.setItem(row, 4, contrib_item)

        last = self.table.rowCount()
        self.table.insertRow(last)
        bold = QFont("Microsoft YaHei", 10, QFont.Bold)
        for col, text in enumerate(["★ 全口总计", f"{total_area:.2f}", "--",
                                    f"{total_dirty_area:.2f}", f"{overall_weighted_rate:.2f}%"]):
            item = QTableWidgetItem(text)
            item.setFont(bold)
            if col == 0:
                item.setBackground(QColor("#f8f9fa"))
            if col == 4:
                self._apply_score_style(item, overall_weighted_rate)
            self.table.setItem(last, col, item)
        self.table.viewport().update()

    @staticmethod
    def _apply_score_style(item, score):
        if score < 60:
            item.setForeground(QColor("#d9534f"))
        elif score < 85:
            item.setForeground(QColor("#f0ad4e"))
        else:
            item.setForeground(QColor("#5cb85c"))

    def closeEvent(self, event):
        try:
            self.plotter.close()
        except Exception:
            pass
        event.accept()


if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    win = MeshSegmentationWindow()
    win.show()
    sys.exit(app.exec_())
