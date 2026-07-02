import os
import sys
import numpy as np
import open3d as o3d
import pyvista as pv
from pyvistaqt import QtInteractor
from skimage import color
from scipy.spatial.distance import cdist
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QLabel, QPushButton, QMessageBox, QApplication)


class AdvancedCleanerWindow(QMainWindow):
    def __init__(self, input_path, output_dir, color_threshold=12.0):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.base_threshold = float(color_threshold)  # 主界面的原始阈值
        self.mesh_name = os.path.basename(input_path)
        self.result_data = None

        # 1. 加载数据：读取 PointCloud
        if not os.path.exists(input_path):
            print(f"未找到文件: {input_path}")
            return

        print("正在加载点云数据...")
        self.original_pcd = o3d.io.read_point_cloud(input_path)
        if not self.original_pcd.has_colors():
            print(f"跳过：{self.mesh_name} 缺少顶点颜色。")
            return

        # 提取顶点和颜色
        self.points = np.asarray(self.original_pcd.points)
        self.original_colors = np.asarray(self.original_pcd.colors)
        self.total_points_count = len(self.points)

        print(f"成功加载点云: {self.mesh_name}, 总点数: {self.total_points_count}")

        # 预计算 LAB 空间
        print("正在预计算色彩空间，请稍候...")
        rgb_view = self.original_colors.reshape(-1, 1, 3)
        self.pcd_lab = color.rgb2lab(rgb_view).reshape(-1, 3)
        # 根据当前点云颜色分布自适应阈值，降低不同来源数据的手感差异
        lab_std = np.std(self.pcd_lab, axis=0)
        std_mean = float(np.mean(lab_std))
        self.threshold_scale = float(np.clip(std_mean / 25.0, 0.55, 1.40))
        self.effective_base_threshold = self.base_threshold * self.threshold_scale

        # 根据点云稀疏度自适应点大小，尽量保持视觉密度一致
        nn_dists = np.asarray(self.original_pcd.compute_nearest_neighbor_distance())
        if nn_dists.size > 0:
            median_nn = float(np.median(nn_dists))
            self.display_point_size = float(np.clip(4.0 * (median_nn / 0.08), 2.5, 9.0))
        else:
            self.display_point_size = 4.0

        # 状态管理
        self.picked_labs = []
        self.picked_modes = []  # 记录每一次选点是在哪个界面操作的，以便匹配各自的阈值
        self.history = []  # 撤销历史
        self.is_previewing = False
        self._point_picking_enabled = False
        self.view_mode = "edit"  # edit / selected / remaining
        self._visible_indices = np.arange(len(self.points))

        # 初始化 PyVista 数据对象
        self.cloud = pv.PolyData(self.points)
        self._set_cloud_colors(self.original_colors)

        # 初始化 UI
        self.setWindowTitle(f"点云色彩交互清理工具: {self.mesh_name}")
        self.resize(1200, 800)
        self._init_ui()

    def _set_cloud_colors(self, colors_01):
        """同步点云显示颜色（映射到 0-255 字节范围）"""
        self.cloud.point_data["colors"] = (colors_01 * 255).astype(np.uint8)

    def _save_camera(self):
        try:
            return self.plotter.camera_position
        except Exception:
            return None

    def _restore_camera(self, camera_pos):
        if camera_pos is None: return
        try:
            self.plotter.camera_position = camera_pos
        except Exception:
            pass

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 左侧控制面板
        control_panel = QWidget()
        control_panel.setFixedWidth(260)
        panel_layout = QVBoxLayout(control_panel)

        self.info_label = QLabel(f"已选颜色样本: 0")
        self.info_label.setStyleSheet("font-weight: bold; color: #0078d7; font-size: 14px; margin: 15px 0;")
        panel_layout.addWidget(self.info_label)

        btn_style = "padding: 12px; font-size: 13px; margin-bottom: 5px;"

        self.btn_undo = QPushButton("返回上一步 (Undo)")
        self.btn_undo.setStyleSheet(btn_style)
        self.btn_undo.clicked.connect(self._undo)
        panel_layout.addWidget(self.btn_undo)

        self.btn_selected_view = QPushButton("已选点界面")
        self.btn_selected_view.setStyleSheet(btn_style)
        self.btn_selected_view.clicked.connect(self._switch_to_selected_mode)
        panel_layout.addWidget(self.btn_selected_view)

        self.btn_remaining_view = QPushButton("剩余点界面")
        self.btn_remaining_view.setStyleSheet(btn_style)
        self.btn_remaining_view.clicked.connect(self._switch_to_remaining_mode)
        panel_layout.addWidget(self.btn_remaining_view)

        self.btn_edit_view = QPushButton("返回选点界面")
        self.btn_edit_view.setStyleSheet(btn_style + "background-color: #0078d7; color: white;")
        self.btn_edit_view.clicked.connect(self._switch_to_edit_mode)
        panel_layout.addWidget(self.btn_edit_view)

        self.btn_save = QPushButton("最终确认并保存 (Save)")
        self.btn_save.setStyleSheet(btn_style + "background-color: #28a745; color: white; font-weight: bold;")
        self.btn_save.clicked.connect(self._save_to_disk)
        panel_layout.addWidget(self.btn_save)

        tip_label = QLabel(
            "💡 操作提示：\n"
            "· 【主界面右键】：选色过滤，变红。\n"
            "· 【已选点界面右键】：将误选的颜色“退回”给剩余点（精细小阈值）。\n"
            "· 【剩余点界面右键】：继续追加选色，“扔给”已选点（精细小阈值）。\n"
            "· 已选点/剩余点界面均显示点云【原色】。\n"
            "· 按住鼠标【左键】可拖动旋转 3D 视角。")
        tip_label.setStyleSheet("color: #444; font-size: 11px; margin-top: 20px; line-height: 1.5;")
        panel_layout.addWidget(tip_label)

        panel_layout.addStretch()
        main_layout.addWidget(control_panel)

        # 右侧渲染器
        self.plotter = QtInteractor(self)
        main_layout.addWidget(self.plotter)
        self.plotter.set_background([0.2, 0.2, 0.2])

        self._initial_render_view()

    def _bind_point_picking(self):
        if self._point_picking_enabled:
            return
        self.plotter.track_click_position(callback=self._on_point_picked, side='right')
        self._point_picking_enabled = True

    def _view_title(self):
        if self.view_mode == "selected":
            return "【已选点界面】（精细小阈值 & 显示原色）：右键点击可“退回”该颜色至剩余点"
        if self.view_mode == "remaining":
            return "【剩余点界面】（精细小阈值 & 显示原色）：右键点击可将该颜色“追加”至已选点"
        return "【主选点界面】：右键点击选色（选中区域变红）"

    def _get_selection_mask(self):
        if not self.picked_labs:
            return np.zeros(len(self.points), dtype=bool)

        dists = cdist(self.pcd_lab, np.array(self.picked_labs), metric='euclidean')

        # 核心逻辑：根据样本是在哪个界面被选取的，动态赋予其过滤时对应的阈值
        thresh_list = [
            self.effective_base_threshold * 0.5 if mode in ["selected", "remaining"] else self.effective_base_threshold
            for mode in self.picked_modes
        ]
        threshold_array = np.array(thresh_list).reshape(1, -1)
        return np.any(dists < threshold_array, axis=1)

    def _get_view_indices(self):
        selection_mask = self._get_selection_mask()
        if self.view_mode == "selected":
            return np.where(selection_mask)[0]
        if self.view_mode == "remaining":
            return np.where(~selection_mask)[0]
        return np.arange(len(self.points))

    def _get_view_colors(self, indices):
        return self.original_colors[indices].copy()

    def _apply_pick_colors_to_cloud(self):
        if not self.picked_labs:
            self._set_cloud_colors(self.original_colors)
            return

        mask = self._get_selection_mask()
        display_colors = self.original_colors.copy()
        display_colors[mask] = [1.0, 0.0, 0.0]  # 同步包含子视口追加操作后的全局红染区域
        self._set_cloud_colors(display_colors)

    def _initial_render_view(self):
        self.view_mode = "edit"
        self._visible_indices = np.arange(len(self.points))
        self.plotter.clear()
        self.plotter.add_text(self._view_title(), font_size=10, name="title_text")

        self._apply_pick_colors_to_cloud()

        self.plotter.add_mesh(
            self.cloud,
            scalars="colors",
            rgb=True,
            style='points',
            point_size=self.display_point_size,
            render_points_as_spheres=True,
            name="main_pcd",
            pickable=True
        )
        self._bind_point_picking()
        self.plotter.reset_camera()
        self.plotter.render()

    def _render_view(self, reset_view=False):
        self._visible_indices = self._get_view_indices()
        self.plotter.add_text(self._view_title(), font_size=10, name="title_text")

        if self.view_mode == "edit":
            self.plotter.remove_actor("main_pcd")
            self._apply_pick_colors_to_cloud()
            self.plotter.add_mesh(
                self.cloud,
                scalars="colors",
                rgb=True,
                style='points',
                point_size=self.display_point_size,
                render_points_as_spheres=True,
                name="main_pcd",
                pickable=True
            )
        else:
            self.plotter.remove_actor("main_pcd")
            if len(self._visible_indices) > 0:
                sub_cloud = pv.PolyData(self.points[self._visible_indices])
                view_colors = self._get_view_colors(self._visible_indices)
                sub_cloud.point_data["colors"] = (view_colors * 255).astype(np.uint8)
                self.plotter.add_mesh(
                    sub_cloud,
                    scalars="colors",
                    rgb=True,
                    style='points',
                    point_size=self.display_point_size,
                    render_points_as_spheres=True,
                    name="main_pcd",
                    pickable=True
                )
            else:
                self.plotter.add_mesh(pv.PolyData(), name="main_pcd")

        if reset_view:
            self.plotter.reset_camera()
        self.plotter.render()

    def _switch_to_edit_mode(self):
        self.view_mode = "edit"
        self._render_view(reset_view=False)

    def _switch_to_selected_mode(self):
        if not self.picked_labs or not np.any(self._get_selection_mask()):
            QMessageBox.information(self, "提示", "当前没有已选点，请先在选点界面右键选点。")
            return
        self.view_mode = "selected"
        self._render_view(reset_view=False)

    def _switch_to_remaining_mode(self):
        self.view_mode = "remaining"
        self._render_view(reset_view=False)

    def _pick_global_index(self, picked_point):
        if picked_point is None or len(self._visible_indices) == 0:
            return -1

        if isinstance(picked_point, (np.ndarray, list, tuple)):
            pos = picked_point
        elif hasattr(picked_point, "points"):
            pos = picked_point.points[0]
        else:
            pos = picked_point

        if self.view_mode == "edit":
            idx = self.cloud.find_closest_point(pos)
            return int(idx) if idx >= 0 else -1

        local_cloud = pv.PolyData(self.points[self._visible_indices])
        local_idx = local_cloud.find_closest_point(pos)
        if local_idx < 0:
            return -1
        return int(self._visible_indices[local_idx])

    def _on_point_picked(self, picked_point):
        if self.is_previewing:
            return

        idx = self._pick_global_index(picked_point)
        if idx < 0:
            return

        selected_lab = self.pcd_lab[idx].copy()
        self.history.append((list(self.picked_labs), list(self.picked_modes)))

        if self.view_mode == "selected":
            # 1. 在【已选点界面】右键：寻找之前是哪个样本把这个点变红的，并剔除它（使用0.5倍小阈值进行逆向匹配）
            if self.picked_labs:
                dists = cdist(selected_lab.reshape(1, -1), np.array(self.picked_labs), metric='euclidean')[0]
                thresh_list = [
                    self.effective_base_threshold * 0.5 if m in ["selected", "remaining"] else self.effective_base_threshold
                    for m in self.picked_modes
                ]
                matched_indices = np.where(dists < np.array(thresh_list))[0]
                if len(matched_indices) > 0:
                    target_idx = matched_indices[0]
                    self.picked_labs.pop(target_idx)
                    self.picked_modes.pop(target_idx)
                    print("已通过小阈值精细剔除该颜色，点云已回流至【剩余点】。")
                else:
                    self.picked_labs.pop()
                    self.picked_modes.pop()

            if not self.picked_labs:
                self.view_mode = "edit"
                QMessageBox.information(self, "提示", "已选点已全部退回，自动返回主选点界面。")
                self._update_display()
                self.plotter.reset_camera()
                return

        elif self.view_mode == "remaining":
            # 2. 在【剩余点界面】右键：以精细小阈值（0.5倍）追加该颜色至样本池，并扔给【已选点】
            self.picked_labs.append(selected_lab)
            self.picked_modes.append(self.view_mode)  # 存入 "remaining"，使其在 Mask 中触发 0.5 倍阈值
            print("已以精细小阈值追加过滤颜色，点云已递交给【已选点】。")
        else:
            # 3. 主选点界面右键：使用 base_threshold 标准阈值
            self.picked_labs.append(selected_lab)
            self.picked_modes.append(self.view_mode)

        self._update_display()

    def _update_display(self):
        self.info_label.setText(f"已选颜色样本: {len(self.picked_labs)}")
        camera_pos = self._save_camera()
        self._visible_indices = self._get_view_indices()
        self._render_view(reset_view=False)
        self._restore_camera(camera_pos)
        self.plotter.render()

    def _undo(self):
        if self.is_previewing:
            return
        if self.history:
            self.picked_labs, self.picked_modes = self.history.pop()
            if not self.picked_labs and self.view_mode == "selected":
                self.view_mode = "edit"
            self._update_display()

    def _save_to_disk(self):
        if not self.picked_labs:
            print("未选取样本，无法保存。")
            return

        keep_mask = ~self._get_selection_mask()
        keep_indices = np.where(keep_mask)[0]

        final_pcd = self.original_pcd.select_by_index(keep_indices)
        final_points_count = len(final_pcd.points)
        final_clean_rate = (final_points_count / self.total_points_count) * 100

        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, f"Cleaned_{self.mesh_name}")

        o3d.io.write_point_cloud(save_path, final_pcd)

        self.result_data = {
            "name": self.mesh_name,
            "original_points": int(self.total_points_count),
            "final_points": int(final_points_count),
            "clean_rate": float(final_clean_rate)
        }
        print(
            f"保存成功: {save_path}, 剩余点数占比: {final_clean_rate:.2f}% ({final_points_count}/{self.total_points_count})")
        self.close()

    def closeEvent(self, event):
        self.plotter.close()
        event.accept()


if __name__ == "__main__":
    import json

    # 两种用法：
    # 1) 无参数：使用下面的默认测试路径（原有行为）
    # 2) 被主流程作为子进程调用：
    #    python CleanSingletooth2.py <input_ply> <output_dir> [color_threshold] [result_json]
    if len(sys.argv) >= 3:
        input_path = sys.argv[1]
        output_dir = sys.argv[2]
        color_threshold = float(sys.argv[3]) if len(sys.argv) >= 4 else 12.0
        result_json = sys.argv[4] if len(sys.argv) >= 5 else ""
    else:
        input_path = r"D:\UClean\Seg_outsideface.ply"
        output_dir = "./output"
        color_threshold = 12.0
        result_json = ""

    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    if os.path.exists(input_path):
        cleaner = AdvancedCleanerWindow(input_path, output_dir, color_threshold)
        cleaner.show()
        app.exec_()
        # 将清理结果写入 JSON，供父进程（主流程）回收统计数据
        if result_json and cleaner.result_data:
            with open(result_json, "w", encoding="utf-8") as f:
                json.dump(cleaner.result_data, f, ensure_ascii=False)
        sys.exit(0)
    else:
        print("文件路径错误")
        sys.exit(1)