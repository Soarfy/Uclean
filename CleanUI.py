# -*- coding: utf-8 -*-
"""GUI for the true-3D-area dental cleanliness workflow."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


BACKEND_IMPORT_ERROR = ""
try:
    import GeneratePathOffset66initAllnewdorobotsnewAreaWeighted as backend
except Exception:
    backend = None
    BACKEND_IMPORT_ERROR = traceback.format_exc()


# The table is always rendered in this anatomical order, never in score order.
FIXED_REGION_ORDER = ("baseface", "insideface", "outsideface", "tips", "upface")
REGION_DISPLAY_NAMES = {
    "baseface": "baseface（牙根区域）",
    "insideface": "insideface（内侧牙面）",
    "outsideface": "outsideface（外侧牙面）",
    "tips": "tips（牙缝区域）",
    "upface": "upface（上槽牙）",
}


class CleanlinessWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.project_dir = Path(__file__).resolve().parent
        self.path_edits = {}
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("牙模清洁度分析 - 3D真实面积加权")
        self.resize(1120, 760)
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        title = QLabel("牙模清洁度分析（自动 ICP + 分区 + 三维面积灰度积分）")
        title.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
        layout.addWidget(title)

        defaults = {
            "standard": self.project_dir / "pointsdata" / "LowerJawScans.ply",
            "unclean": self.project_dir / "pointsdata" / "UncleanLowerJawScan.ply",
            "cleaned": self.project_dir / "pointsdata" / "IO9-3 LowerJawScan.ply",
            "templates": self.project_dir / "segmentationsply",
            "output": self.project_dir / "cleanliness_results_area_weighted",
        }
        rows = (
            ("标准模型", "standard", False),
            ("没刷的模型", "unclean", False),
            ("刷过的模型", "cleaned", False),
            ("分割好的文件夹", "templates", True),
            ("结果输出文件夹", "output", True),
        )
        for label, key, is_directory in rows:
            row = QHBoxLayout()
            caption = QLabel(label + "：")
            caption.setFixedWidth(145)
            edit = QLineEdit(str(defaults[key]))
            button = QPushButton("选择…")
            button.setFixedWidth(90)
            button.clicked.connect(
                lambda _checked=False, k=key, d=is_directory: self._choose_path(k, d)
            )
            self.path_edits[key] = edit
            row.addWidget(caption)
            row.addWidget(edit, 1)
            row.addWidget(button)
            layout.addLayout(row)

        controls = QHBoxLayout()
        self.show_uv_box = QCheckBox("显示 UV")
        self.show_matching_box = QCheckBox("显示匹配结果")
        self.show_point_cloud_box = QCheckBox("显示点云对比结果")
        self.save_intermediate_box = QCheckBox("保存中间结果")
        option_boxes = (
            self.show_uv_box,
            self.show_matching_box,
            self.show_point_cloud_box,
            self.save_intermediate_box,
        )
        for box in option_boxes:
            box.setChecked(True)
            controls.addWidget(box)
        self.run_button = QPushButton("开始计算清洁度")
        self.run_button.setMinimumHeight(44)
        self.run_button.setStyleSheet(
            "QPushButton {background:#1677ff;color:white;font-weight:bold;font-size:14px;"
            "border-radius:5px;padding:8px 24px;}"
            "QPushButton:disabled {background:#9abff2;}"
        )
        self.run_button.clicked.connect(self._run)
        controls.addStretch(1)
        controls.addWidget(self.run_button)
        layout.addLayout(controls)

        hint = QLabel(
            "提示：ICP 完全自动，不需要选点。中间过程窗口关闭后会继续下一步；"
            "basedown 不计算，baseface 正常计算。"
        )
        hint.setStyleSheet("color:#666;")
        layout.addWidget(hint)

        self.status_label = QLabel("等待计算")
        self.status_label.setStyleSheet("font-weight:bold;color:#444;padding-top:6px;")
        layout.addWidget(self.status_label)

        self.table = QTableWidget(len(FIXED_REGION_ORDER) + 1, 7)
        self.table.setHorizontalHeaderLabels(
            [
                "固定顺序",
                "区域",
                "清洁前面积(mm²)",
                "清洁后面积(mm²)",
                "清洁前污渍积分",
                "清洁后污渍积分",
                "区域清洁度",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table, 1)
        self._render_empty_table()

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

    def _render_empty_table(self) -> None:
        for row, name in enumerate(FIXED_REGION_ORDER):
            self.table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.table.setItem(row, 1, QTableWidgetItem(REGION_DISPLAY_NAMES[name]))
            for column in range(2, 7):
                item = QTableWidgetItem("等待计算")
                item.setForeground(QColor("#999999"))
                self.table.setItem(row, column, item)
        total_row = len(FIXED_REGION_ORDER)
        self.table.setItem(total_row, 0, QTableWidgetItem("—"))
        self.table.setItem(total_row, 1, QTableWidgetItem("全部有效区域"))
        for column in range(2, 7):
            self.table.setItem(total_row, column, QTableWidgetItem("—"))

    def _validate_paths(self) -> dict:
        paths = {key: edit.text().strip() for key, edit in self.path_edits.items()}
        for key in ("standard", "unclean", "cleaned"):
            if not os.path.isfile(paths[key]):
                raise FileNotFoundError(f"模型文件不存在：{paths[key]}")
        if not os.path.isdir(paths["templates"]):
            raise FileNotFoundError(f"分割文件夹不存在：{paths['templates']}")
        if not list(Path(paths["templates"]).glob("*.ply")):
            raise FileNotFoundError("分割文件夹中没有找到 PLY 文件")
        return paths

    def _run(self) -> None:
        if backend is None:
            QMessageBox.critical(
                self,
                "后端加载失败",
                "无法加载 3D 面积加权后端。请检查 open3d、numpy、scipy、opencv-python。\n\n"
                + BACKEND_IMPORT_ERROR,
            )
            return
        try:
            paths = self._validate_paths()
        except Exception as exc:
            QMessageBox.warning(self, "路径错误", str(exc))
            return

        self.run_button.setEnabled(False)
        self.run_button.setText("正在计算，请按提示关闭中间窗口…")
        self.status_label.setText("正在执行自动 ICP 配准…")
        self._render_empty_table()
        QApplication.processEvents()
        try:
            report = backend.calculate_cleanliness(
                standard_model_path=paths["standard"],
                unclean_model_path=paths["unclean"],
                cleaned_model_path=paths["cleaned"],
                template_dir=paths["templates"],
                output_dir=paths["output"],
                show_uv=self.show_uv_box.isChecked(),
                show_matching_results=self.show_matching_box.isChecked(),
                show_point_cloud_comparison=self.show_point_cloud_box.isChecked(),
                save_intermediate_results=self.save_intermediate_box.isChecked(),
            )
            self._render_report(report)
            score = float(report.get("overall_cleanliness", 0.0))
            self.status_label.setText(
                f"计算完成：有效区域 {report.get('total_count', 0)} 个，整体清洁度 {score:.2f}%"
            )
            QMessageBox.information(
                self,
                "计算完成",
                f"整体清洁度：{score:.2f}%\n结果已保存到：\n{paths['output']}",
            )
        except Exception as exc:
            traceback.print_exc()
            self.status_label.setText("计算失败")
            QMessageBox.critical(self, "计算失败", f"{exc}\n\n{traceback.format_exc()}")
        finally:
            self.run_button.setEnabled(True)
            self.run_button.setText("开始计算清洁度")
            QApplication.processEvents()

    @staticmethod
    def _score_item(value: float) -> QTableWidgetItem:
        item = QTableWidgetItem(f"{value:.2f}%")
        item.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        if value < 60:
            item.setForeground(QColor("#d4380d"))
        elif value < 85:
            item.setForeground(QColor("#d48806"))
        else:
            item.setForeground(QColor("#389e0d"))
        return item

    def _render_report(self, report: dict) -> None:
        details = {str(item.get("name", "")).casefold(): item for item in report.get("details", [])}
        for row, name in enumerate(FIXED_REGION_ORDER):
            self.table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.table.setItem(row, 1, QTableWidgetItem(REGION_DISPLAY_NAMES[name]))
            item = details.get(name.casefold())
            if item is None:
                for column in range(2, 7):
                    missing = QTableWidgetItem("未匹配")
                    missing.setForeground(QColor("#999999"))
                    self.table.setItem(row, column, missing)
                continue
            before_3d = item.get("unclean_3d", {})
            after_3d = item.get("cleaned_3d", {})
            values = (
                float(item.get("before_surface_area", 0.0)),
                float(item.get("after_surface_area", 0.0)),
                float(before_3d.get("darkness_integral_3d", 0.0)),
                float(after_3d.get("darkness_integral_3d", 0.0)),
            )
            for column, value in enumerate(values, start=2):
                self.table.setItem(row, column, QTableWidgetItem(f"{value:.6f}"))
            self.table.setItem(row, 6, self._score_item(float(item.get("cleanliness", 0.0))))

        total_row = len(FIXED_REGION_ORDER)
        # Overall values are measured on the complete meshes with basedown
        # removed; they are intentionally not sums of the displayed regions.
        before_area = float(report.get(
            "total_unclean_surface_area",
            sum(float(x.get("before_surface_area", 0.0)) for x in details.values()),
        ))
        after_area = float(report.get(
            "total_cleaned_surface_area",
            sum(float(x.get("after_surface_area", 0.0)) for x in details.values()),
        ))
        self.table.setItem(total_row, 0, QTableWidgetItem("—"))
        total_name = QTableWidgetItem("全部有效区域")
        total_name.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.table.setItem(total_row, 1, total_name)
        self.table.setItem(total_row, 2, QTableWidgetItem(f"{before_area:.6f}"))
        self.table.setItem(total_row, 3, QTableWidgetItem(f"{after_area:.6f}"))
        self.table.setItem(total_row, 4, QTableWidgetItem(f"{float(report.get('total_unclean_darkness_integral', 0)):.6f}"))
        self.table.setItem(total_row, 5, QTableWidgetItem(f"{float(report.get('total_cleaned_darkness_integral', 0)):.6f}"))
        self.table.setItem(total_row, 6, self._score_item(float(report.get("overall_cleanliness", 0.0))))
        self.table.resizeRowsToContents()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    window = CleanlinessWindow()
    window.show()
    sys.exit(app.exec_())
