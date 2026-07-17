# -*- coding: utf-8 -*-
"""交互式 Mesh 闭环分割标注工具。

工作流：加载 Mesh → 按 K 进入选点模式 → 左键逐点绘制闭环 →
提取闭环内部区域 → 命名保存 → 导出分区 PLY 与标注 JSON。
"""

import json
import heapq
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

# 本工具只负责 Mesh 分割标注，不加载清洁度评估后端。
AdvancedCleanerWindow = None
_BACKEND_OK = False
_BACKEND_ERR = "分割标注模式未启用清洁度后端"

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

        self._scribbling = False
        self._picker = None
        self._saved_interactor_style = None
        self._user_style = None
        self._face_centers = None
        self._face_tree = None
        self._line_face_ids = []
        self._line_points = []
        self._line_anchor_points = []
        self._face_neighbors = None
        self._display_cell_to_base = None
        self._overlay_renderer = None
        self._overlay_actors = []
        self._point_mode = False

        self.setWindowTitle("Mesh 闭环分割标注")
        self.resize(1280, 820)
        self.setFocusPolicy(Qt.StrongFocus)
        self._init_ui()
        self._setup_scribble_events()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        panel = QWidget()
        panel.setFixedWidth(300)
        pl = QVBoxLayout(panel)

        title = QLabel("Mesh 闭环分割标注")
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

        grp_scribble = QGroupBox("逐点闭环分割")
        gs = QVBoxLayout(grp_scribble)

        self.chk_show_edges = QCheckBox("显示三角面片网格")
        self.chk_show_edges.setChecked(True)
        self.chk_show_edges.toggled.connect(lambda _checked: self._refresh_view())
        gs.addWidget(self.chk_show_edges)

        edit_row = QHBoxLayout()
        self.btn_undo_point = QPushButton("撤销上一个点")
        self.btn_undo_point.clicked.connect(self.undo_last_line_point)
        edit_row.addWidget(self.btn_undo_point)
        self.btn_clear_scribble = QPushButton("清除当前线")
        self.btn_clear_scribble.clicked.connect(self.clear_scribble)
        edit_row.addWidget(self.btn_clear_scribble)
        gs.addLayout(edit_row)

        self.btn_close_line = QPushButton("闭合分割线并提取内部区域")
        self.btn_close_line.setStyleSheet(btn_style + "background:#8e44ad;color:white;font-weight:bold;")
        self.btn_close_line.clicked.connect(self.close_line_and_select)
        gs.addWidget(self.btn_close_line)

        self.btn_label = QPushButton("命名并保存当前分区")
        self.btn_label.setStyleSheet(btn_style + "background:#f0ad4e;color:white;font-weight:bold;")
        self.btn_label.clicked.connect(self.confirm_label)
        gs.addWidget(self.btn_label)

        self.list_segments = QListWidget()
        self.list_segments.setMaximumHeight(140)
        gs.addWidget(QLabel("已标注分区:"))
        gs.addWidget(self.list_segments)

        self.btn_export = QPushButton("导出全部标注分区")
        self.btn_export.setStyleSheet(btn_style + "background:#28a745;color:white;font-weight:bold;")
        self.btn_export.clicked.connect(self.export_labeled)
        gs.addWidget(self.btn_export)

        tip = QLabel(
            "操作：\n"
            "· 按 K 进入选点模式，直接左键逐点点击\n"
            "· 按 Esc 退出选点模式，恢复左键旋转\n"
            "· 点击「闭合分割线」后自动提取较小一侧\n"
            "· 非选点模式：左键旋转，右键/滚轮缩放\n"
            "· 检查黄色选区后命名保存，再继续下一分区")
        tip.setStyleSheet("color:#444;font-size:10px;line-height:1.4;")
        gs.addWidget(tip)
        pl.addWidget(grp_scribble)

        self.lbl_info = QLabel("状态: 请加载 Mesh")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("font-weight:bold;color:#0078d7;margin-top:6px;")
        pl.addWidget(self.lbl_info)

        pl.addStretch()
        root.addWidget(panel)

        self.plotter = QtInteractor(self)
        self.plotter.set_background([0.2, 0.2, 0.2])
        root.addWidget(self.plotter, stretch=1)
        self._init_overlay_renderer()

    def _init_overlay_renderer(self):
        """创建不受 Mesh 深度遮挡的前景渲染层。"""
        import vtk
        render_window = self.plotter.render_window
        render_window.SetNumberOfLayers(2)
        overlay = vtk.vtkRenderer()
        overlay.SetLayer(1)
        overlay.SetInteractive(False)
        overlay.SetPreserveColorBuffer(True)
        overlay.SetPreserveDepthBuffer(False)
        overlay.SetActiveCamera(self.plotter.renderer.GetActiveCamera())
        render_window.AddRenderer(overlay)
        self._overlay_renderer = overlay

    def _refresh_overlay(self):
        """在前景层绘制始终可见的直线和控制点。"""
        if self._overlay_renderer is None:
            return
        import vtk
        for actor in self._overlay_actors:
            self._overlay_renderer.RemoveActor(actor)
        self._overlay_actors.clear()
        self._overlay_renderer.SetActiveCamera(
            self.plotter.renderer.GetActiveCamera())

        if len(self._line_points) >= 2:
            line_data = pv.lines_from_points(
                np.asarray(self._line_points), close=False)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(line_data)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 1.0, 1.0)
            actor.GetProperty().SetLineWidth(2)
            self._overlay_renderer.AddActor(actor)
            self._overlay_actors.append(actor)

        if self._line_anchor_points:
            point_data = pv.PolyData(np.asarray(self._line_anchor_points))
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(point_data)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.0, 1.0, 1.0)
            actor.GetProperty().SetPointSize(4)
            actor.GetProperty().SetRenderPointsAsSpheres(True)
            self._overlay_renderer.AddActor(actor)
            self._overlay_actors.append(actor)

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

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_K:
            self._set_point_mode(True)
            event.accept()
            return
        if event.key() == Qt.Key_Escape and self._point_mode:
            self._set_point_mode(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def _set_point_mode(self, enabled):
        self._point_mode = bool(enabled)
        self._stop_scribble_session()
        self._apply_trackball_mode()
        if self._point_mode:
            self.lbl_info.setText("选点模式已开启：左键添加分割点，Esc 退出。")
        else:
            self.lbl_info.setText("选点模式已退出：左键可旋转模型，按 K 再次进入。")

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
        self._build_face_adjacency()

    def _build_face_adjacency(self):
        """建立共享边的三角面邻接表，供表面路径和闭环分割使用。"""
        if self.base_mesh is None:
            self._face_neighbors = None
            return
        faces = self.base_mesh.faces.reshape(-1, 4)[:, 1:4]
        edge_owner = {}
        neighbors = [set() for _ in range(len(faces))]
        for fi, tri in enumerate(faces):
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edge = (int(min(a, b)), int(max(a, b)))
                other = edge_owner.get(edge)
                if other is None:
                    edge_owner[edge] = fi
                else:
                    neighbors[fi].add(other)
                    neighbors[other].add(fi)
        self._face_neighbors = [tuple(items) for items in neighbors]

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
        return self.base_mesh is not None and self._point_mode

    def _apply_trackball_mode(self):
        """非选点状态保持常规模型视角操作。"""
        self.plotter.enable_trackball_style()
        self._stop_scribble_session()

    def _setup_scribble_events(self):
        """按 K 进入模式后，左键逐点添加闭环控制点。"""
        vtk_iren = self._get_vtk_iren()
        add_obs = getattr(self.plotter.iren, "add_observer", None)
        if add_obs is None:
            add_obs = vtk_iren.AddObserver

        def on_button_press(_obj, _evt):
            if not self._can_paint_now():
                return
            if _evt != "LeftButtonPressEvent":
                return
            self._start_scribble_session()
            self._paint_at_cursor()

        def on_move(_obj, _evt):
            if self._scribbling:
                self._stop_scribble_session()

        def on_button_release(_obj, _evt):
            self._stop_scribble_session()

        def on_key_press(obj, _evt):
            key = str(obj.GetKeySym()).lower() if hasattr(obj, "GetKeySym") else ""
            if key == "k":
                self._set_point_mode(True)
            elif key == "escape" and self._point_mode:
                self._set_point_mode(False)

        add_obs("LeftButtonPressEvent", on_button_press)
        add_obs("MouseMoveEvent", on_move)
        add_obs("LeftButtonReleaseEvent", on_button_release)
        add_obs("KeyPressEvent", on_key_press)
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
        display_cell = int(self._picker.GetCellId())
        if (self._display_cell_to_base is None
                or display_cell >= len(self._display_cell_to_base)):
            return
        picked_cell = int(self._display_cell_to_base[display_cell])
        if not self._line_face_ids:
            self._line_face_ids.append(picked_cell)
            self._line_anchor_points.append(pick_pos)
        elif picked_cell != self._line_face_ids[-1]:
            self._line_face_ids.append(picked_cell)
            self._line_anchor_points.append(pick_pos)
        else:
            return
        if not self._rebuild_line_geometry():
            self._line_face_ids.pop()
            self._line_anchor_points.pop()
            self.lbl_info.setText("该点与上一点不连通，请点击同一块网格表面。")
            return
        self.lbl_info.setText(
            f"已放置 {len(self._line_face_ids)} 个分割点；继续点击或闭合分割线。")
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

            self.scribble_faces.clear()
            self.labeled_segments.clear()
            self.list_segments.clear()
            self._line_face_ids.clear()
            self._line_points.clear()
            self._line_anchor_points.clear()

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
        source = self.base_mesh.copy()
        source.cell_data["_base_cell_id"] = np.arange(
            self.base_mesh.n_cells, dtype=np.int64)
        visible_ids = np.where(~self.assigned)[0]
        cam = None if reset_camera else self.plotter.camera_position
        self.plotter.clear()
        if len(visible_ids):
            mesh = source.extract_cells(visible_ids).extract_surface()
            base_ids = np.asarray(
                mesh.cell_data["_base_cell_id"], dtype=np.int64)
            self._display_cell_to_base = base_ids
            mesh.cell_data["display_rgb"] = colors[base_ids]
            self.plotter.add_mesh(
                mesh, scalars="display_rgb", rgb=True,
                show_edges=self.chk_show_edges.isChecked(),
                show_vertices=self.chk_show_edges.isChecked(),
                edge_color="#333333", line_width=1,
                vertex_color="#222222", point_size=2,
                name="mesh", pickable=True)
        else:
            self._display_cell_to_base = np.empty(0, dtype=np.int64)
        if reset_camera:
            self.plotter.reset_camera()
        elif cam is not None:
            self.plotter.camera_position = cam
        self._refresh_overlay()
        self.plotter.render()

    # ---------------- 涂抹分割 ----------------
    def _rebuild_line_geometry(self):
        """控制点之间使用直线显示；闭环分割时再计算表面路径。"""
        self._line_points.clear()
        if not self._line_face_ids:
            return True
        self._line_points.extend(
            np.asarray(point) for point in self._line_anchor_points)
        return True

    def undo_last_line_point(self):
        if not self._line_face_ids:
            return
        self._line_face_ids.pop()
        self._line_anchor_points.pop()
        self._rebuild_line_geometry()
        self._refresh_view()
        self.lbl_info.setText(f"已撤销，当前保留 {len(self._line_face_ids)} 个分割点。")

    def clear_scribble(self):
        self.scribble_faces.clear()
        self._line_face_ids.clear()
        self._line_points.clear()
        self._line_anchor_points.clear()
        self._refresh_view()
        self.lbl_info.setText("已清除当前选区")

    def _shortest_face_path(self, start, goal):
        """用 A* 沿网格表面补齐相邻鼠标采样点之间的轨迹。"""
        if start == goal:
            return [start]
        centers = self._face_centers
        first_score = float(np.linalg.norm(centers[start] - centers[goal]))
        queue = [(first_score, 0.0, start)]
        previous = {start: None}
        distance = {start: 0.0}
        while queue:
            _score, cost, current = heapq.heappop(queue)
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = previous[current]
                return path[::-1]
            if cost > distance.get(current, float("inf")):
                continue
            for nxt in self._face_neighbors[current]:
                step = float(np.linalg.norm(centers[current] - centers[nxt]))
                new_cost = cost + step
                if new_cost < distance.get(nxt, float("inf")):
                    distance[nxt] = new_cost
                    previous[nxt] = current
                    heuristic = float(np.linalg.norm(centers[nxt] - centers[goal]))
                    heapq.heappush(queue, (new_cost + heuristic, new_cost, nxt))
        return []

    def close_line_and_select(self):
        """闭合表面轨迹，并把闭环相邻的较小连通区域作为内部选区。"""
        if self.base_mesh is None or len(self._line_face_ids) < 3:
            QMessageBox.information(self, "提示", "请按 K 进入选点模式，再用左键添加至少三个分割点。")
            return

        samples = self._line_face_ids + [self._line_face_ids[0]]
        boundary = set()
        for start, goal in zip(samples[:-1], samples[1:]):
            path = self._shortest_face_path(start, goal)
            if not path:
                QMessageBox.warning(self, "闭环失败", "分割线跨越了不连通的网格，请重新绘制。")
                return
            boundary.update(path)

        available = {
            i for i in range(self.base_mesh.n_cells)
            if not self.assigned[i] and i not in boundary
        }
        components = []
        while available:
            seed = available.pop()
            component = {seed}
            stack = [seed]
            touches_boundary = False
            while stack:
                current = stack.pop()
                for nxt in self._face_neighbors[current]:
                    if nxt in boundary:
                        touches_boundary = True
                    elif nxt in available:
                        available.remove(nxt)
                        component.add(nxt)
                        stack.append(nxt)
            if touches_boundary:
                components.append(component)

        if len(components) < 2:
            QMessageBox.warning(
                self, "无法分割",
                "这条线没有把网格分成内外两侧。请让线完整环绕目标，并避免跨越孔洞。")
            return

        inside = min(components, key=len)
        inside.update(fi for fi in boundary if not self.assigned[fi])
        self.scribble_faces = inside
        self._line_face_ids.clear()
        self._line_points.clear()
        self._line_anchor_points.clear()
        self._refresh_view()
        self.lbl_info.setText(
            f"闭环已完成，选中 {len(inside)} 个三角面；请检查后命名保存。")

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
