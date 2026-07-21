# -*- coding: utf-8 -*-
"""GUI for selectable standard/detailed dental cleanliness calculation."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtCore import QProcess
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


BACKEND_IMPORT_ERROR = ""
try:
    import SelectableAreaCleanliness as backend
except Exception:
    backend = None
    BACKEND_IMPORT_ERROR = traceback.format_exc()

VIEWER_IMPORT_ERROR = ""
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except Exception:
    pv = None
    QtInteractor = None
    VIEWER_IMPORT_ERROR = traceback.format_exc()


MODE_DATA = {
    "标准大区域分割": "standard",
    "详细细分区域": "detailed",
}

GROUP_DISPLAY_ORDER = (
    ("insideface", "内侧面"),
    ("outsideface", "外侧面"),
    ("upface", "上槽牙（咬合面）"),
    ("tips", "牙缝"),
    ("baseface", "牙根区域（龈沟）"),
)


class SelectableCleanlinessWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.project_dir = Path(__file__).resolve().parent
        self.path_edits: dict[str, QLineEdit] = {}
        self.regions = []
        self._region_actors = {}
        self._region_meshes = {}
        self._build_ui()
        self._reload_regions(show_errors=False)

    def _build_ui(self) -> None:
        self.setWindowTitle("可选分割区域牙模清洁度分析")
        self.resize(1560, 980)
        self.setMinimumSize(1380, 900)
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(9)

        title = QLabel("牙模清洁度分析（标准大区域 / 详细细分区域可选）")
        title.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        layout.addWidget(title)

        defaults = {
            "standard": self.project_dir / "pointsdata" / "LowerJawScans.ply",
            "unclean": self.project_dir / "pointsdata" / "UncleanLowerJawScan.ply",
            "cleaned": self.project_dir / "pointsdata" / "IO9-3 LowerJawScan.ply",
            "segments": self.project_dir / "segmentationfolder",
            "output": self.project_dir / "selectable_cleanliness_results",
        }
        rows = (
            ("标准模型", "standard", False),
            ("刷牙前模型", "unclean", False),
            ("刷牙后模型", "cleaned", False),
            ("分割方案文件夹", "segments", True),
            ("结果输出文件夹", "output", True),
        )
        for caption_text, key, is_directory in rows:
            row = QHBoxLayout()
            caption = QLabel(caption_text + "：")
            caption.setFixedWidth(130)
            edit = QLineEdit(str(defaults[key]))
            button = QPushButton("选择…")
            button.setFixedWidth(84)
            button.clicked.connect(
                lambda _checked=False, k=key, d=is_directory: self._choose_path(k, d)
            )
            self.path_edits[key] = edit
            row.addWidget(caption)
            row.addWidget(edit, 1)
            row.addWidget(button)
            layout.addLayout(row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("分割模式："))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(MODE_DATA.keys())
        self.mode_combo.currentIndexChanged.connect(self._reload_regions)
        mode_row.addWidget(self.mode_combo)
        self.reload_button = QPushButton("重新读取分割")
        self.reload_button.clicked.connect(self._reload_regions)
        mode_row.addWidget(self.reload_button)
        mode_row.addStretch(1)
        auto_seg_button = QPushButton("GPU 自动单牙分割…")
        auto_seg_button.clicked.connect(self._open_auto_segmentation)
        mode_row.addWidget(auto_seg_button)
        layout.addLayout(mode_row)

        splitter = QSplitter(Qt.Horizontal)
        selection_panel = QWidget()
        selection_layout = QVBoxLayout(selection_panel)
        selection_layout.setContentsMargins(0, 0, 8, 0)
        self.region_summary = QLabel("尚未读取分割")
        self.region_summary.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        selection_layout.addWidget(self.region_summary)
        self.region_list = QListWidget()
        self.region_list.itemChanged.connect(self._update_selected_summary)
        selection_layout.addWidget(self.region_list, 1)

        selection_buttons = QHBoxLayout()
        select_all = QPushButton("全选")
        select_none = QPushButton("全不选")
        select_all.clicked.connect(lambda: self._set_all_checked(True))
        select_none.clicked.connect(lambda: self._set_all_checked(False))
        selection_buttons.addWidget(select_all)
        selection_buttons.addWidget(select_none)
        selection_layout.addLayout(selection_buttons)

        preview_buttons = QHBoxLayout()
        preview_all = QPushButton("3D 预览全部组合")
        preview_selected = QPushButton("3D 预览所选组合")
        preview_all.clicked.connect(lambda: self._preview(selected_only=False))
        preview_selected.clicked.connect(lambda: self._preview(selected_only=True))
        preview_buttons.addWidget(preview_all)
        preview_buttons.addWidget(preview_selected)
        selection_layout.addLayout(preview_buttons)

        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(8, 0, 0, 0)
        viewer_title = QLabel("所选分割区域实时 3D 预览（左键旋转 / 滚轮缩放 / 中键平移）")
        viewer_title.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        result_layout.addWidget(viewer_title)
        if QtInteractor is not None:
            self.plotter = QtInteractor(result_panel)
            self.plotter.set_background("white")
            self.plotter.add_axes()
            self.plotter.enable_anti_aliasing()
            result_layout.addWidget(self.plotter.interactor, 3)
        else:
            self.plotter = None
            viewer_error = QLabel(
                "无法加载嵌入式 3D 视图，请安装 pyvista、pyvistaqt 和 vtk。\n"
                + VIEWER_IMPORT_ERROR
            )
            viewer_error.setWordWrap(True)
            viewer_error.setStyleSheet("color:#d4380d;background:#fff1f0;padding:10px;")
            result_layout.addWidget(viewer_error, 1)
        result_title = QLabel("清洁度计算结果")
        result_title.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        result_layout.addWidget(result_title)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "区域", "所属大区", "刷前面积(mm²)", "刷后面积(mm²)",
            "刷前污渍积分", "刷后污渍积分", "清洁度",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.setMinimumHeight(285)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.verticalHeader().setDefaultSectionSize(36)
        result_layout.addWidget(self.table, 2)

        splitter.addWidget(selection_panel)
        splitter.addWidget(result_panel)
        splitter.setSizes([430, 800])
        layout.addWidget(splitter, 1)

        options = QHBoxLayout()
        self.show_matching_box = QCheckBox("显示配准结果")
        self.show_gray_box = QCheckBox("显示所选区域灰度对比")
        self.save_intermediate_box = QCheckBox("保存中间结果")
        self.show_matching_box.setChecked(True)
        self.show_gray_box.setChecked(False)
        self.save_intermediate_box.setChecked(True)
        options.addWidget(self.show_matching_box)
        options.addWidget(self.show_gray_box)
        options.addWidget(self.save_intermediate_box)
        options.addStretch(1)
        self.run_button = QPushButton("开始计算所选区域清洁度")
        self.run_button.setMinimumHeight(44)
        self.run_button.setStyleSheet(
            "QPushButton {background:#1677ff;color:white;font-weight:bold;font-size:14px;"
            "border-radius:5px;padding:8px 24px;}"
            "QPushButton:disabled {background:#9abff2;}"
        )
        self.run_button.clicked.connect(self._run)
        options.addWidget(self.run_button)
        layout.addLayout(options)

        self.status_label = QLabel("勾选或取消区域时，右侧 3D 模型会立即显示或隐藏对应模块。")
        self.status_label.setStyleSheet("color:#555;font-weight:bold;")
        layout.addWidget(self.status_label)

    def _choose_path(self, key: str, is_directory: bool) -> None:
        current = self.path_edits[key].text().strip()
        if is_directory:
            selected = QFileDialog.getExistingDirectory(self, "选择文件夹", current)
        else:
            selected, _ = QFileDialog.getOpenFileName(
                self, "选择彩色牙模", current, "PLY 模型 (*.ply);;所有文件 (*)"
            )
        if selected:
            self.path_edits[key].setText(selected)
            if key == "segments":
                self._reload_regions()

    def _mode(self) -> str:
        return MODE_DATA[self.mode_combo.currentText()]

    def _open_auto_segmentation(self) -> None:
        script = self.project_dir / "AutoToothSegUI.py"
        started = QProcess.startDetached(sys.executable, [str(script)], str(self.project_dir))
        if not started:
            QMessageBox.critical(self, "启动失败", f"无法启动自动单牙分割模块：{script}")

    def _reload_regions(self, _value=None, show_errors: bool = True) -> None:
        self._clear_embedded_preview()
        self.region_list.blockSignals(True)
        self.region_list.clear()
        self.regions = []
        try:
            if backend is None:
                raise RuntimeError(BACKEND_IMPORT_ERROR)
            self.regions = backend.discover_regions(
                self.path_edits["segments"].text().strip(), self._mode()
            )
            for index, region in enumerate(self.regions):
                label = region.name if self._mode() == "standard" else f"{region.group} / {region.name}"
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, region.name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                # Start with one visible module, matching the selection-first
                # workflow and avoiding loading all 65 detailed meshes at once.
                item.setCheckState(Qt.Checked if index == 0 else Qt.Unchecked)
                self.region_list.addItem(item)
        except Exception as exc:
            self.region_summary.setText("分割读取失败")
            if show_errors:
                QMessageBox.warning(self, "分割读取失败", str(exc))
        finally:
            self.region_list.blockSignals(False)
        self._update_selected_summary()
        self._sync_embedded_preview(reset_camera=True)

    def _selected_names(self) -> list[str]:
        return [
            self.region_list.item(i).data(Qt.UserRole)
            for i in range(self.region_list.count())
            if self.region_list.item(i).checkState() == Qt.Checked
        ]

    def _selected_regions(self):
        selected = {name.casefold() for name in self._selected_names()}
        return [region for region in self.regions if region.name.casefold() in selected]

    def _update_selected_summary(self, _item=None) -> None:
        total = self.region_list.count()
        selected = len(self._selected_names())
        mode_name = self.mode_combo.currentText()
        self.region_summary.setText(f"{mode_name}：共 {total} 个模块，已选择 {selected} 个")
        self._sync_embedded_preview()

    def _set_all_checked(self, checked: bool) -> None:
        self.region_list.blockSignals(True)
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.region_list.count()):
            self.region_list.item(i).setCheckState(state)
        self.region_list.blockSignals(False)
        self._update_selected_summary()

    def _clear_embedded_preview(self) -> None:
        self._region_actors.clear()
        self._region_meshes.clear()
        if getattr(self, "plotter", None) is not None:
            self.plotter.clear()
            self.plotter.add_axes()
            self.plotter.render()

    def _preview_mesh(self, region):
        key = region.name.casefold()
        if key not in self._region_meshes:
            mesh = pv.read(str(region.path))
            if isinstance(mesh, pv.MultiBlock):
                mesh = mesh.combine()
            self._region_meshes[key] = mesh
        return self._region_meshes[key]

    def _sync_embedded_preview(self, reset_camera: bool = False) -> None:
        if getattr(self, "plotter", None) is None:
            return
        selected = {name.casefold() for name in self._selected_names()}
        region_map = {region.name.casefold(): region for region in self.regions}

        for key in list(self._region_actors):
            if key not in selected:
                self.plotter.remove_actor(self._region_actors.pop(key), render=False)

        palette = (
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
            "#00a6a6", "#f1c40f", "#e5508a", "#73808c",
        )
        added = False
        try:
            for key in selected:
                if key in self._region_actors or key not in region_map:
                    continue
                region = region_map[key]
                region_index = self.regions.index(region)
                actor = self.plotter.add_mesh(
                    self._preview_mesh(region),
                    color=palette[region_index % len(palette)],
                    name=f"region_{key}",
                    smooth_shading=False,
                    render=False,
                )
                self._region_actors[key] = actor
                added = True
            if reset_camera or (added and len(self._region_actors) == 1):
                self.plotter.reset_camera(render=False)
            self.plotter.render()
        except Exception as exc:
            self.status_label.setText(f"3D 区域加载失败：{exc}")

    def _preview(self, selected_only: bool) -> None:
        regions = self._selected_regions() if selected_only else self.regions
        if not regions:
            QMessageBox.warning(self, "无法预览", "请至少选择一个区域。")
            return
        scope = "所选" if selected_only else "全部"
        self.status_label.setText(f"正在显示{scope}分割模块的组合 3D 模型，关闭窗口后返回。")
        QApplication.processEvents()
        try:
            backend.show_region_preview(
                regions, f"{self.mode_combo.currentText()} - {scope}模块组合预览"
            )
        except Exception as exc:
            QMessageBox.critical(self, "3D 预览失败", f"{exc}\n\n{traceback.format_exc()}")
        finally:
            self.status_label.setText("3D 预览已关闭，可以继续选择区域或开始计算。")

    def _validate_paths(self) -> dict[str, str]:
        paths = {key: edit.text().strip() for key, edit in self.path_edits.items()}
        for key in ("standard", "unclean", "cleaned"):
            if not os.path.isfile(paths[key]):
                raise FileNotFoundError(f"模型文件不存在：{paths[key]}")
        if not os.path.isdir(paths["segments"]):
            raise FileNotFoundError(f"分割方案文件夹不存在：{paths['segments']}")
        if not self._selected_names():
            raise ValueError("请至少选择一个参与计算的区域")
        return paths

    @staticmethod
    def _score_item(value: float) -> QTableWidgetItem:
        item = QTableWidgetItem(f"{value:.2f}%")
        item.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        item.setForeground(QColor("#d4380d" if value < 60 else "#d48806" if value < 85 else "#389e0d"))
        return item

    def _render_report(self, report: dict) -> None:
        """Render one aggregate row per large region in a fixed order."""
        self.table.setHorizontalHeaderLabels([
            "大区域", "已选详细分区", "刷前面积(mm²)", "刷后面积(mm²)",
            "刷前污渍积分", "刷后污渍积分", "清洁度",
        ])
        by_group = {
            item.get("group", "").casefold(): item
            for item in report.get("group_details", [])
        }
        self.table.setRowCount(len(GROUP_DISPLAY_ORDER) + 1)
        for row, (group, chinese_name) in enumerate(GROUP_DISPLAY_ORDER):
            item = by_group.get(group)
            name_item = QTableWidgetItem(chinese_name)
            name_item.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
            self.table.setItem(row, 0, name_item)
            if item is None:
                for column in range(1, 7):
                    cell = QTableWidgetItem("未选择" if column == 1 else "—")
                    cell.setForeground(QColor("#8c8c8c"))
                    self.table.setItem(row, column, cell)
                continue
            modules = item.get("selected_modules", [])
            self.table.setItem(row, 1, QTableWidgetItem(", ".join(modules)))
            before = item.get("unclean_3d", {})
            after = item.get("cleaned_3d", {})
            values = (
                item.get("before_surface_area", 0.0),
                item.get("after_surface_area", 0.0),
                before.get("darkness_integral_3d", 0.0),
                after.get("darkness_integral_3d", 0.0),
            )
            for column, value in enumerate(values, start=2):
                self.table.setItem(row, column, QTableWidgetItem(f"{float(value):.6f}"))
            self.table.setItem(row, 6, self._score_item(float(item.get("cleanliness", 0.0))))

        row = len(GROUP_DISPLAY_ORDER)
        total = QTableWidgetItem("所选区域整体")
        total.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.table.setItem(row, 0, total)
        self.table.setItem(row, 1, QTableWidgetItem("全部已选区域"))
        totals = (
            report.get("total_unclean_surface_area", 0.0),
            report.get("total_cleaned_surface_area", 0.0),
            report.get("total_unclean_darkness_integral", 0.0),
            report.get("total_cleaned_darkness_integral", 0.0),
        )
        for column, value in enumerate(totals, start=2):
            self.table.setItem(row, column, QTableWidgetItem(f"{float(value):.6f}"))
        self.table.setItem(
            row, 6, self._score_item(float(report.get("overall_cleanliness", 0.0)))
        )
        return

        details = report.get("details", [])
        self.table.setRowCount(len(details) + 1)
        for row, item in enumerate(details):
            before = item.get("unclean_3d", {})
            after = item.get("cleaned_3d", {})
            values = [
                item.get("name", ""), item.get("group", ""),
                f"{float(item.get('before_surface_area', 0.0)):.6f}",
                f"{float(item.get('after_surface_area', 0.0)):.6f}",
                f"{float(before.get('darkness_integral_3d', 0.0)):.6f}",
                f"{float(after.get('darkness_integral_3d', 0.0)):.6f}",
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))
            self.table.setItem(row, 6, self._score_item(float(item.get("cleanliness", 0.0))))

        row = len(details)
        total_name = QTableWidgetItem("所选区域整体")
        total_name.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.table.setItem(row, 0, total_name)
        self.table.setItem(row, 1, QTableWidgetItem("—"))
        totals = [
            report.get("total_unclean_surface_area", 0.0),
            report.get("total_cleaned_surface_area", 0.0),
            report.get("total_unclean_darkness_integral", 0.0),
            report.get("total_cleaned_darkness_integral", 0.0),
        ]
        for column, value in enumerate(totals, start=2):
            self.table.setItem(row, column, QTableWidgetItem(f"{float(value):.6f}"))
        self.table.setItem(row, 6, self._score_item(float(report.get("overall_cleanliness", 0.0))))

    def _run(self) -> None:
        if backend is None:
            QMessageBox.critical(self, "后端加载失败", BACKEND_IMPORT_ERROR)
            return
        try:
            paths = self._validate_paths()
        except Exception as exc:
            QMessageBox.warning(self, "输入错误", str(exc))
            return

        self.run_button.setEnabled(False)
        self.status_label.setText("正在执行自动配准和所选区域清洁度计算…")
        QApplication.processEvents()
        try:
            report = backend.calculate_selected_cleanliness(
                standard_model_path=paths["standard"],
                unclean_model_path=paths["unclean"],
                cleaned_model_path=paths["cleaned"],
                segmentation_folder=paths["segments"],
                segmentation_mode=self._mode(),
                selected_region_names=self._selected_names(),
                output_dir=paths["output"],
                show_matching_results=self.show_matching_box.isChecked(),
                show_point_cloud_comparison=self.show_gray_box.isChecked(),
                save_intermediate_results=self.save_intermediate_box.isChecked(),
            )
            self._render_report(report)
            score = float(report.get("overall_cleanliness", 0.0))
            self.status_label.setText(
                f"计算完成：有效区域 {report.get('total_count', 0)} 个，所选区域整体清洁度 {score:.2f}%"
            )
            QMessageBox.information(
                self, "计算完成",
                f"所选区域整体清洁度：{score:.2f}%\n结果已保存到：\n{paths['output']}",
            )
        except Exception as exc:
            traceback.print_exc()
            self.status_label.setText("计算失败")
            QMessageBox.critical(self, "计算失败", f"{exc}\n\n{traceback.format_exc()}")
        finally:
            self.run_button.setEnabled(True)
            QApplication.processEvents()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    window = SelectableCleanlinessWindow()
    window.show()
    sys.exit(app.exec_())
