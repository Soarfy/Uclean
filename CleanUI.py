# -*- coding: utf-8 -*-
import sys
import os
import json
import glob
import builtins  # 显式全局导入，确保劫持 input 函数时、以及打包 EXE 后不闪退
import traceback

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLineEdit, QLabel,
                             QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont

# ==========================================
# 1. 尝试导入你的后端参考算法模块
# ==========================================
backend = None
backend_import_error = ""
try:
    import GeneratePathOffset66initAllnewdorobotsnew as backend
except Exception as _e:
    # 注意：这里捕获所有异常（不仅是 ImportError），
    # 因为打包后真正的失败往往是后端模块内部依赖（cv2 / skimage / scipy /
    # open3d / pyvista / pyvistaqt 等）未被收集，导致 import 时抛错。
    backend_import_error = traceback.format_exc()
    print("【警告】后端模块导入失败：\n" + backend_import_error)


# ==========================================
# 2. 牙模评估系统 GUI 核心主类
# ==========================================
class DentalGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('牙模自动化评估系统 - 稳定刷新确认版')
        self.setGeometry(100, 100, 950, 680)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # --- 路径选择配置表 ---
        self.paths = {}
        config = [
            ("训练路径 (Source):", "source", r"D:\UClean\IO9-3 LowerJawScan.ply", False),
            ("扫描路径 (Target):", "target", r"D:\UClean\LowerJawScans.ply", False),
            ("模板目录:", "temp", r"D:\UClean\segmentationsply", True),
            ("输出目录:", "out", r"D:\UClean\segementations", True)
        ]

        for label_text, key, default, is_dir in config:
            h_layout = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(160)
            edit = QLineEdit(default)
            self.paths[key] = edit
            btn = QPushButton("选择...")
            btn.clicked.connect(lambda checked, k=key, d=is_dir: self.browse_path(k, d))
            h_layout.addWidget(lbl)
            h_layout.addWidget(edit)
            h_layout.addWidget(btn)
            layout.addLayout(h_layout)

        # --- 核心控制按钮 ---
        self.run_btn = QPushButton("开始处理流程")
        self.run_btn.setFixedHeight(48)
        self.run_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; font-size: 14px;")
        self.run_btn.clicked.connect(self.execute_logic)
        layout.addWidget(self.run_btn)

        # --- 数据展示表格 ---
        layout.addWidget(QLabel("最新一轮采集评估数据："))
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["牙齿位置", "面积 (mm²)", "局部清洁度", "相对总面贡献"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

    def browse_path(self, key, is_dir):
        """处理路径点击浏览事件"""
        if is_dir:
            path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "3D Files (*.ply *.pcd)")
        if path:
            self.paths[key].setText(path)

    def ask_user_satisfaction(self):
        """核心交互：在 UI 界面弹出配准结果确认框"""
        reply = QMessageBox.question(self, '结果确认',
                                     "选点及配准结果是否满意？\n点击 'Yes' 将继续保存并分析，点击 'No' 将重新运行此步骤。",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        return reply == QMessageBox.Yes

    def execute_logic(self):
        """
        完整核心处理逻辑（带强行刷新与全自动点云补偿分析）
        """
        if backend is None:
            detail = backend_import_error or "（无详细堆栈）"
            QMessageBox.critical(
                self,
                "后端未加载",
                "未能导入后端模块 GeneratePathOffset66initAllnewdorobotsnew。\n"
                "通常是打包时缺少其依赖（cv2 / skimage / scipy / open3d / pyvista / pyvistaqt 等）。\n\n"
                "真实错误如下：\n" + detail
            )
            return

        source = self.paths["source"].text()
        target = self.paths["target"].text()
        temp_dir = self.paths["temp"].text()
        out_dir = self.paths["out"].text()

        # 1. 基础路径存在性校验
        if not all(os.path.exists(p) for p in [source, target, temp_dir]):
            QMessageBox.warning(self, "路径错误", "请检查输入路径、扫描文件及模板目录是否存在！")
            return

        # 2. 冻结按钮，并【强制彻底清空】上一轮的界面残余，避免视觉造成旧数据污染
        self.run_btn.setEnabled(False)
        self.run_btn.setText("后端算法正在高负载处理中，请稍候...")
        self.table.setRowCount(0)
        QApplication.processEvents()  # 🌟 强行令 Windows 刷新界面，使得表格立刻变空

        # 备份系统原始的 input 函数
        original_input = builtins.input

        try:
            # 3. 动态劫持后端的 input 阻塞，映射为前端 PyQt 确认弹窗
            def gui_input(prompt):
                prompt_lower = prompt.lower()
                if any(kw in prompt_lower for kw in ["满意", "satisfied", "confirm", "继续"]):
                    res = self.ask_user_satisfaction()
                    return "y" if res else "n"
                return "y"

            builtins.input = gui_input

            # 4. 【深度清洗内存】强行抹除后端常驻内存可能残留的历史统计数据
            if hasattr(backend, 'all_stats'): backend.all_stats = []
            if hasattr(backend, 'stats_list'): backend.stats_list = []

            # 5. 调用后端核心算法进行同步计算
            print("🚀 前端启动：开始调用后端核心分割、色彩空间预计算与配准算法...")
            backend.process_dental_mesh_registration(
                source_mesh_path=source,
                target_pcd_path=target,
                template_dir=temp_dir,
                output_base_dir=out_dir
            )
            print("🏁 后端运行结束：开始全面捕捉评估数据...")

            stats_list = []

            # 6. 第一优先：直接同步后端本轮运行的内存结果（最实时、必然对应本次选点）
            if hasattr(backend, 'all_stats') and backend.all_stats:
                print("【捕获成功】直接从后端运行内存变量中同步本轮真实数据。")
                stats_list = list(backend.all_stats)

            # 7. 第二优先：读取最新生成的 JSON 报告文件
            if not stats_list:
                possible_json_paths = [
                    os.path.join(out_dir, "mapped_meshes", "cleaned_results", "cleanliness_report.json"),
                    r"D:\UClean\segementations\mapped_meshes\cleaned_results\cleanliness_report.json"
                ]
                for path in possible_json_paths:
                    if os.path.exists(path):
                        print(f"【捕获成功】检测到 JSON 报告: {path}")
                        with open(path, 'r', encoding='utf-8') as f:
                            report_data = json.load(f)
                            stats_list = report_data.get("details", [])
                        break

            # 9. 整合最终捕获的数据流，渲染 UI 表格并进行数学计算
            if stats_list:
                total_area = sum(self._get_weight(s) for s in stats_list)
                if total_area > 0:
                    # 新计算逻辑：
                    # 全口加权总清洁度 = Σ(剩余浅色均值 × 剩余面积) / Σ(选中深色均值 × 分割总面积)
                    total_numerator = sum(
                        float(s.get('remaining_gray_mean', 0) or 0) * float(s.get('remaining_area', 0) or 0)
                        for s in stats_list
                    )
                    total_denominator = sum(
                        float(s.get('picked_gray_mean', 0) or 0) * float(s.get('original_area', 0) or 0)
                        for s in stats_list
                    )
                    total_cleanliness = (total_numerator / total_denominator * 100) if total_denominator > 0 else 0
                else:
                    total_cleanliness = 0

                # 强行刷新前端数据表组件
                self.update_results_table(stats_list)

                msg = (
                    f"全口牙齿评估完毕！\n"
                    f"--------------------------------\n"
                    f"处理牙齿分区：{len(stats_list)} 个核心片区\n"
                    f"总牙面面积：{total_area:.2f} mm²\n"
                    f"全口加权总清洁度：{total_cleanliness:.2f}%\n"
                    f"--------------------------------\n"
                    f"提示：最新一轮采集的数据已被强行绘制渲染至界面。"
                )
                QMessageBox.information(self, "评估完成", msg)
            else:
                QMessageBox.warning(self, "完成",
                                    "后端流程已结束，但在预设路径下未能自动提取到任何新一轮的数据或点云文件。")

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "运行出错", f"后端算法逻辑执行发生异常：\n{str(e)}")

        finally:
            # 10. 恢复标准终端输入流，解除按钮冻结
            builtins.input = original_input
            self.run_btn.setEnabled(True)
            self.run_btn.setText("开始处理流程")
            QApplication.processEvents()  # 再次强行释放界面死锁状态

    @staticmethod
    def _get_weight(s):
        """优先使用真实表面积加权；若旧数据缺少面积字段，退回用点数加权，避免表格空白"""
        area = float(s.get('original_area', 0) or 0)
        if area > 0:
            return area
        return float(s.get('original_points', 0) or 0)

    def update_results_table(self, data):
        """
        强行重绘与渲染数据表，确保视图即时同步
        """
        self.table.setRowCount(0)
        total_area = sum(self._get_weight(s) for s in data)
        if total_area <= 0: return

        # 按局部清洁度由低到高排序，方便医护人员一目了然看清哪里没刷干净
        sorted_data = sorted(data, key=lambda x: x.get('cleanliness', 0))

        # 计算全局分母：所有分区深色均值×面积之和
        total_denominator = sum(
            float(s.get('picked_gray_mean', 0) or 0) * float(s.get('original_area', 0) or 0)
            for s in data
        )

        overall_weighted_rate = 0

        for row, s in enumerate(sorted_data):
            self.table.insertRow(row)
            name = str(s.get('name', 'Unknown'))
            area = self._get_weight(s)
            local_rate = float(s.get('cleanliness', 0))

            # 贡献率计算：
            # 分子 = 单个区域剩余浅色均值 × 剩余面积
            # 分母 = 所有分区深色均值×面积之和（全局）
            # 贡献率 = 分子 / 分母
            picked_gray = float(s.get('picked_gray_mean', 0) or 0)
            remaining_gray = float(s.get('remaining_gray_mean', 0) or 0)
            remaining_area = float(s.get('remaining_area', 0) or 0)

            # 分子：单个区域剩余浅色均值 × 剩余面积
            numerator = remaining_gray * remaining_area

            if total_denominator > 0:
                contribution_rate = numerator / total_denominator
            else:
                contribution_rate = 0.0

            overall_weighted_rate += contribution_rate

            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(f"{area:.2f}"))

            # 局部清洁度（应用红/黄/绿报警色彩机制）；-1 表示面级数据不可用
            if local_rate < 0:
                local_item = QTableWidgetItem("--")
                local_item.setForeground(QColor("#999999"))
            else:
                local_item = QTableWidgetItem(f"{local_rate:.2f}%")
                self._apply_score_style(local_item, local_rate)
            self.table.setItem(row, 2, local_item)

            # 相对贡献
            contrib_item = QTableWidgetItem(f"{contribution_rate:.2f}%")
            contrib_item.setForeground(QColor("#666666"))  # 灰色文字，作为次要辅助参考数据
            self.table.setItem(row, 3, contrib_item)

        # 11. 追加一行全局加粗显眼的总计汇总行
        last_row = self.table.rowCount()
        self.table.insertRow(last_row)

        total_font = QFont("Microsoft YaHei", 10, QFont.Bold)

        label_item = QTableWidgetItem("★ 全口总计")
        label_item.setFont(total_font)
        label_item.setBackground(QColor("#f8f9fa"))

        area_sum_item = QTableWidgetItem(f"{total_area:.2f}")
        area_sum_item.setFont(total_font)

        placeholder_item = QTableWidgetItem("--")

        final_score_item = QTableWidgetItem(f"{overall_weighted_rate:.2f}%")
        final_score_item.setFont(total_font)
        self._apply_score_style(final_score_item, overall_weighted_rate)

        self.table.setItem(last_row, 0, label_item)
        self.table.setItem(last_row, 1, area_sum_item)
        self.table.setItem(last_row, 2, placeholder_item)
        self.table.setItem(last_row, 3, final_score_item)

        # 🌟 终极大招：通知 Windows 核心重新绘制该表格组件的视口，保证肉眼所见即所得
        self.table.viewport().update()
        QApplication.processEvents()

    def _apply_score_style(self, item, score):
        """辅助健康评分色彩控制函数"""
        if score < 60:
            item.setForeground(QColor("#d9534f"))  # 红色（不合格）
        elif score < 85:
            item.setForeground(QColor("#f0ad4e"))  # 橙色（合格但需改善）
        else:
            item.setForeground(QColor("#5cb85c"))  # 绿色（优秀）


# ==========================================
# 3. 应用程序入口点
# ==========================================
if __name__ == '__main__':
    # 注意：这里不要开启 Qt.AA_EnableHighDpiScaling！
    # 开启后 pyvista/VTK 渲染窗口的鼠标坐标会按 devicePixelRatio 偏移，
    # 导致后端 AdvancedCleanerWindow 的右键选点位置不准、点显示变小。
    app = QApplication(sys.argv)
    gui = DentalGui()
    gui.show()
    sys.exit(app.exec_())