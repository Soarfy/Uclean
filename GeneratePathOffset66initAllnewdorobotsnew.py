# -*- coding: utf-8 -*-
# 初始扫描点云的牙模构建

import cv2
import copy
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R
import glob
import open3d as o3d
import numpy as np
import json
import os
from skimage import color
from scipy.spatial.distance import cdist
import os
import sys
import numpy as np
import open3d as o3d
import pyvista as pv
from pyvistaqt import QtInteractor
from skimage import color
from scipy.spatial.distance import cdist
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QLabel, QPushButton, QMessageBox, QApplication,
                             QDoubleSpinBox, QFormLayout)
from PyQt5.QtCore import QEventLoop


class BrushPathSaver:
    def __init__(self, XX=7, YY=11, L_horizontal=4, L_vertical=2):
        self.XX = XX
        self.YY = YY
        self.L_horizontal = L_horizontal
        self.L_vertical = L_vertical
        self.mtx = np.array([[4510.87, 0, 1238.18], [0, 4511.26, 1016.29], [0, 0, 1]], dtype=np.float32)
        self.dist = np.array([-0.0606931, 0.329977, -0.00188508, -0.00107541, -1.69546])

    def select_box_from_pointcloud(self, pcd, initial_size=10.0, step=1.0, angle_step=2.0):
        print("请在弹出的窗口中用 Shift+左键选择包围盒中心点，按 Q 完成选择。")
        vis_pick = o3d.visualization.VisualizerWithEditing()
        vis_pick.create_window(window_name="选择包围盒中心点")
        vis_pick.add_geometry(pcd)
        vis_pick.run()
        vis_pick.destroy_window()

        picked_points = vis_pick.get_picked_points()
        if not picked_points:
            print("未选择任何点，使用点云中心作为包围盒中心。")
            bbox_center = np.asarray(pcd.get_center())
        else:
            bbox_center = np.asarray(pcd.points)[picked_points[0]]

        print(f"包围盒中心点为: {bbox_center}")

        extent = np.array([initial_size] * 3)
        obb = o3d.geometry.OrientedBoundingBox(bbox_center, np.eye(3), extent)
        obb.color = (1, 0, 0)

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="调整矩形框，按 Q 确认")

        vis.add_geometry(pcd)
        vis.add_geometry(obb)

        def update_obb():
            vis.update_geometry(obb)

        # --- 缩放 ---
        def scale(dx=0, dy=0, dz=0):
            new_extent = obb.extent + np.array([dx, dy, dz])
            obb.extent = new_extent
            update_obb()

        # --- 旋转 ---
        def rotate(axis, direction=1):
            R = self.get_rotation_matrix(axis, angle_step * direction)
            obb.R = R @ obb.R
            update_obb()

        # --- 平移 ---
        def translate(dx=0, dy=0, dz=0):
            translation = np.array([dx, dy, dz])
            obb.center = obb.center + translation
            update_obb()

        # 平移绑定：WASDQE
        vis.register_key_callback(ord("D"), lambda vis: translate(dx=step))  # +X
        vis.register_key_callback(ord("A"), lambda vis: translate(dx=-step))  # -X
        vis.register_key_callback(ord("E"), lambda vis: translate(dy=step))  # +Y
        vis.register_key_callback(ord("Q"), lambda vis: translate(dy=-step))  # -Y
        vis.register_key_callback(ord("W"), lambda vis: translate(dz=step))  # +Z
        vis.register_key_callback(ord("S"), lambda vis: translate(dz=-step))  # -Z

        # 缩放绑定：Z/X/C/V/B/N
        vis.register_key_callback(ord("Z"), lambda vis: scale(dx=step))
        vis.register_key_callback(ord("X"), lambda vis: scale(dx=-step))
        vis.register_key_callback(ord("C"), lambda vis: scale(dy=step))
        vis.register_key_callback(ord("V"), lambda vis: scale(dy=-step))
        vis.register_key_callback(ord("B"), lambda vis: scale(dz=step))
        vis.register_key_callback(ord("N"), lambda vis: scale(dz=-step))

        # 旋转绑定：J/L/I/K/U/O
        vis.register_key_callback(ord("J"), lambda vis: rotate('z', direction=1))  # yaw +
        vis.register_key_callback(ord("L"), lambda vis: rotate('z', direction=-1))  # yaw -
        vis.register_key_callback(ord("I"), lambda vis: rotate('y', direction=1))  # pitch +
        vis.register_key_callback(ord("K"), lambda vis: rotate('y', direction=-1))  # pitch -
        vis.register_key_callback(ord("U"), lambda vis: rotate('x', direction=1))  # roll +
        vis.register_key_callback(ord("O"), lambda vis: rotate('x', direction=-1))  # roll -

        vis.run()
        vis.destroy_window()

        print(obb)

        cropped = pcd.crop(obb)
        return cropped

    def select_box_from_pointcloudbrushandtooth(self, pcd):
        # 固定 OBB 参数
        bbox_center = np.array([-9.17991, 5.67431, -116.632])  # 中心点
        extent = np.array([22, 47, 13])  # 宽、长、高
        rotation = np.eye(3)  # 如果没有旋转，保持单位矩阵

        bbox_centertooth = np.array([13.7611, -10.9384, -172.216])  # 中心点
        # extenttooth = np.array([38, 29, 16])  # 宽、长、高
        extenttooth = np.array([82, 93, 36])  # 宽、长、高
        rotationtooth = np.eye(3)  # 如果没有旋转，保持单位矩阵

        # 创建固定 OBB
        obb = o3d.geometry.OrientedBoundingBox(bbox_center, rotation, extent)
        obb.color = (1, 0, 0)

        obbtooth = o3d.geometry.OrientedBoundingBox(bbox_centertooth, rotationtooth, extenttooth)
        obbtooth.color = (0, 1, 0)

        # 显示原点云和 OBB（可选）
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(window_name="固定 OBB 框选结果")
        # vis.add_geometry(pcd)
        # vis.add_geometry(obb)
        # vis.add_geometry(obbtooth)
        # vis.run()
        # vis.destroy_window()

        # 裁剪并返回结果
        cropped = pcd.crop(obb)
        croppedtooth = pcd.crop(obbtooth)
        return cropped ,croppedtooth

    def load_board2cam_results_from_txt(self,path):
        # 构造文件路径
        rvecs_path = os.path.join(path, "rvecs.txt")
        tvecs_path = os.path.join(path, "tvecs.txt")
        indices_path = os.path.join(path, "valid_indices.txt")

        # 加载数据
        rvecs_arr = np.loadtxt(rvecs_path).reshape(-1, 3)
        tvecs_arr = np.loadtxt(tvecs_path).reshape(-1, 3)
        valid_indices = np.loadtxt(indices_path, dtype=int).tolist()

        # 转为列表形式，每个元素是 (3,1) 的 ndarray
        rvecs = [vec.reshape(3, 1) for vec in rvecs_arr]
        tvecs = [vec.reshape(3, 1) for vec in tvecs_arr]

        return rvecs, tvecs, valid_indices

    def get_board2cam_v2(self, image_dir, objp, w=7, h=11, show_result=True):
        import re
        rvecs, tvecs, valid_indices = [], [], []
        files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith('.bmp')],
            key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else -1
        )

        max_attempts = 18  # 最多尝试提升亮度5次
        brightness_step = 10  # 每次增加的亮度值

        for idx, fname in enumerate(files):
            img_path = os.path.join(image_dir, fname)
            print(f'处理图片: {img_path}')

            raw_gray = 250 - cv2.imread(img_path, 0)  # 原始反转灰度图
            gray = raw_gray.copy()

            ret, centers = cv2.findCirclesGrid(gray, (w, h), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            attempt = 0

            # 若初次未检测成功则尝试提升亮度
            while not ret and attempt < max_attempts:
                attempt += 1
                bright_gray = np.clip(gray + attempt * brightness_step, 0, 255).astype(np.uint8)
                bright_gray =cv2.convertScaleAbs(bright_gray,alpha=1.5,beta=20)
                ret, centers = cv2.findCirclesGrid(bright_gray, (w, h), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
                if ret:
                    gray = bright_gray  # 用增强后的图继续处理
                    print(f"{fname} 提升亮度后检测成功（第 {attempt} 次尝试）")
                    break

            if ret and centers.shape[0] == objp.shape[0]:
                if show_result:
                    show = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    cv2.drawChessboardCorners(show, (w, h), centers, ret)
                    cv2.imshow(f"Detected {idx}", show)
                    cv2.waitKey(300)
                    cv2.destroyWindow(f"Detected {idx}")
                success, rvec, tvec = cv2.solvePnP(objp, centers, self.mtx, self.dist)
                if success:
                    rvecs.append(rvec)
                    tvecs.append(tvec)
                    valid_indices.append(idx)
                    print(f"{fname} 检测成功")
            else:
                print(f"{fname} 检测失败")
        print(f"有效样本数: {len(rvecs)}")
        return rvecs, tvecs, valid_indices

    def manual_select_and_detect_on_resized(self, original_gray, pattern_size):
        # 将原图缩放至 640x480
        h, w = original_gray.shape
        resized_gray = cv2.resize(original_gray, (1280, 960))
        clone = cv2.cvtColor(resized_gray.copy(), cv2.COLOR_GRAY2BGR)
        selected_points = []

        def click_event(event, x, y, flags, param):
            nonlocal clone  # 提前声明
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_points.append((x, y))
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
            elif event == cv2.EVENT_RBUTTONDOWN and selected_points:
                selected_points.pop()
                clone = cv2.cvtColor(resized_gray.copy(), cv2.COLOR_GRAY2BGR)
                for pt in selected_points:
                    cv2.circle(clone, pt, 1, (0, 0, 255), -1)

        cv2.namedWindow("Manual Select")
        cv2.setMouseCallback("Manual Select", click_event)

        print("请点击标定点，右键撤销，按 Q 确认")

        while True:
            cv2.imshow("Manual Select", clone)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyWindow("Manual Select")

        if len(selected_points) != pattern_size[0] * pattern_size[1]:
            print("选择的点数不匹配")
            return None

        # 从 640x480 映射回原图大小
        scale_x = w / 1280
        scale_y = h / 960
        mapped_points = np.array([[[x * scale_x, y * scale_y]] for (x, y) in selected_points], dtype=np.float32)

        return mapped_points


    def get_board2cam_v3(self, image_dir, objp, w=7, h=11, show_result=True):
        import re

        rvecs, tvecs, valid_indices = [], [], []
        files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith('.bmp')],
            key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else -1
        )

        max_attempts = 38
        brightness_step = 6

        for idx, fname in enumerate(files):
            img_path = os.path.join(image_dir, fname)
            print(f'处理图片: {img_path}')

            raw_gray = 250 - cv2.imread(img_path, 0)
            gray = raw_gray.copy()

            ret, centers = cv2.findCirclesGrid(gray, (w, h), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            attempt = 0

            # 亮度增强
            while not ret and attempt < max_attempts:
                attempt += 1
                bright_gray = np.clip(gray + attempt * brightness_step, 0, 255).astype(np.uint8)
                bright_gray = cv2.convertScaleAbs(bright_gray, alpha=1.5, beta=10)
                ret, centers = cv2.findCirclesGrid(bright_gray, (w, h), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
                if ret:
                    gray = bright_gray
                    print(f"{fname} 提升亮度后检测成功（第 {attempt} 次尝试）")
                    break

            # 如果还失败，手动标注
            if not ret:
                print(f"{fname} 自动检测失败，进入人工标注")
                # raw_grayw = cv2.imread(img_path)
                centers = self.manual_select_and_detect_on_resized(raw_gray, (w, h))
                if centers is not None and centers.shape[0] == objp.shape[0]:
                    ret = True

            if ret and centers.shape[0] == objp.shape[0]:
                if show_result:
                    show = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                    cv2.drawChessboardCorners(show, (w, h), centers, ret)
                    show = cv2.resize(show, (640, 480))
                    cv2.imshow(f"Detected {idx}", show)
                    cv2.waitKey(600)
                    cv2.destroyWindow(f"Detected {idx}")
                success, rvec, tvec = cv2.solvePnP(objp, centers, self.mtx, self.dist)
                if success:
                    rvecs.append(rvec)
                    tvecs.append(tvec)
                    valid_indices.append(idx)
                    print(f"{fname} 检测成功")
            else:
                print(f"{fname} 检测失败")

        print(f"有效样本数: {len(rvecs)}")

        # === 保存结果到 txt 文件 ===
        # np.savetxt(os.path.join(image_dir, "rvecs.txt"), np.array(rvecs).reshape(len(rvecs), 3), fmt="%.6f")
        # np.savetxt(os.path.join(image_dir, "tvecs.txt"), np.array(tvecs).reshape(len(tvecs), 3), fmt="%.6f")
        # np.savetxt(os.path.join(image_dir, "valid_indices.txt"), np.array(valid_indices), fmt="%d")
        # print("结果已保存为 rvecs.txt, tvecs.txt, valid_indices.txt")

        return rvecs, tvecs, valid_indices

    def build_transform_matrix_from_pnp(self, rvec, tvec):
        R, _ = cv2.Rodrigues(rvec)
        H = np.eye(4)
        H[:3, :3] = R.T
        H[:3, 3] = (-R.T @ tvec).flatten()
        return H

    def scantobase(self, image_dir, ply_dir):
        objp = np.zeros((self.XX * self.YY, 3), np.float32)
        objp[:, :2] = np.array([[(2 * j + i % 2) * self.L_horizontal / 2, i * self.L_vertical]
                                for i in range(self.YY) for j in range(self.XX)])

        # rvecs, tvecs, valid_indices = self.get_board2cam_v2(image_dir, objp)
        # print("rvecs :",rvecs)
        # print("tvecs :",tvecs)
        # print("valid_indices :",valid_indices)

        # rvecs, tvecs, valid_indices = self.get_board2cam_v3(image_dir, objp)
        # print("rvecs :", rvecs)
        # print("tvecs :", tvecs)
        # print("valid_indices :", valid_indices)
        # rvecs, tvecs, valid_indices = self.load_board2cam_results_from_txt("D:\\hand_eye_calibration\\")

        rvecs = [np.array([[0.052025],
                       [0.11564863],
                       [-1.52599881]]), np.array([[0.51135969],
                                               [-0.32279626],
                                               [-1.47077935]]), np.array([[0.43161902],
                                                                       [-0.25718822],
                                                                       [-1.48645496]]), np.array([[0.38512335],
                                                                                               [-0.21983382],
                                                                                               [-1.49658353]]),
                np.array([[-0.56399725],
                       [0.59205233],
                       [-1.46159681]]), np.array([[-0.56208738],
                                               [0.5936893],
                                               [-1.46261156]]), np.array([[-0.5619906],
                                                                       [0.59378976],
                                                                       [-1.46291129]]), np.array([[0.67498363],
                                                                                               [-0.88772154],
                                                                                               [1.41028314]]),
                np.array([[0.04982151],
                       [0.119391],
                       [-1.42787529]]), np.array([[0.32149651],
                                               [0.40819927],
                                               [-1.39416184]])]
        tvecs = [np.array([[-27.70809509],
                       [11.75862643],
                       [183.65961567]]), np.array([[-31.47434612],
                                                [17.9947646],
                                                [166.47028778]]), np.array([[-12.92977982],
                                                                         [19.44917965],
                                                                         [166.97849319]]), np.array([[-2.81400374],
                                                                                                  [20.34740992],
                                                                                                  [166.49119001]]),
                np.array([[-17.01618127],
                       [18.1091849],
                       [161.03948607]]), np.array([[1.13095918],
                                                [27.8442349],
                                                [156.99667482]]), np.array([[4.1178226],
                                                                         [27.82822751],
                                                                         [154.31297594]]), np.array([[-8.0884848],
                                                                                                  [14.63001934],
                                                                                                  [145.05143674]]),
                np.array([[-43.17341761],
                       [17.24187208],
                       [182.61674778]]), np.array([[-46.4046467],
                                                [10.99776187],
                                                [178.17767086]])]
        valid_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]


        rvecs = rvecs[:-2]
        tvecs = tvecs[:-2]
        valid_indices = valid_indices[:-2]


        if not rvecs:
            print("没有有效PnP结果，退出")
            return None

        H_cam_list = [self.build_transform_matrix_from_pnp(r, t) for r, t in zip(rvecs, tvecs)]

        ply_files = sorted(
            [f for f in os.listdir(ply_dir) if f.endswith('.ply')],
            key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]) if x.split('_')[-1].split('.')[0].isdigit() else -1
        )
        selected_ply_files = [ply_files[i] for i in valid_indices]

        # 原始的所有 gripper pose
        gripper_poses = [
            [-9.278, 441.969, 422.757, -179.346, -0.653, 84.695],
            [-119.088, 450.759, 374.552, 148.328, -2.323, 84.687],
            [-119.087, 450.785, 374.559, 153.62, -2.324, 84.688],
            [-119.066, 450.825, 374.563, 156.657, -2.41, 84.552],
            [108.826, 431.026, 341.94, -140.27, -2.835, 84.538],
            [91.074, 440.293, 351.845, -140.274, -2.845, 84.531],
            [86.97, 440.335, 351.855, -140.273, -2.855, 84.524],
            # [35.91, 594.716, 484.448, 160.834, 60.901, -107.477],
            [35.964, 557.786, 446.559, 160.852, 60.924, -107.456],
            [36.791, 635.395, 498.368, 164.879, 52.32, -102.649],
            [25.411, 452.19, 422.686, -179.534, -0.215, 90.558],
            [25.468, 529.468, 459.959, -179.317, 20.414, 90.593]
        ]
        selected_gripper_poses = [gripper_poses[i] for i in valid_indices]

        if len(selected_ply_files) != len(H_cam_list):
            print("有效点云与变换数量不一致")
            return None

        all_pcds = []
        for ply_file, H in zip(selected_ply_files, H_cam_list):
            pcd = o3d.io.read_point_cloud(os.path.join(ply_dir, ply_file))
            if not pcd.has_points():
                continue
            pcd.transform(H)
            points = np.asarray(pcd.points)
            mask = (points[:, 2] >= -36) & (points[:, 2] <= 36)
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
            if pcd.has_colors():
                filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
            filtered_pcd, _ = filtered_pcd.remove_statistical_outlier(16, 1.0)
            if not filtered_pcd.has_colors():
                filtered_pcd.paint_uniform_color(np.random.rand(3))
            all_pcds.append(filtered_pcd)

        if not all_pcds:
            print("无有效点云")
            return None

        merged = all_pcds[0]
        merged_down = merged.voxel_down_sample(voxel_size=0.5)

        for i, source in enumerate(all_pcds[1:], 1):
            source_down = source.voxel_down_sample(0.25)
            reg = o3d.pipelines.registration.registration_icp(
                source_down, merged_down, 1.0, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            print(f"[ICP] 第{i}个点云: fitness={reg.fitness:.4f}, rmse={reg.inlier_rmse:.4f}")
            source.transform(reg.transformation)
            merged += source
            merged_down = merged.voxel_down_sample(0.5)

        o3d.io.write_point_cloud("D:/UsmileProject/hand_eye_calibration/mergeredoldboard.ply", merged)

        # 这里需要修改
        # tranindex = 1
        #
        # # ======= 新增：将merged旋转到机械臂基坐标系 =======
        # R_cam2board, _ = cv2.Rodrigues(rvecs[tranindex])


        # @@@@@@@@@@@@@@@@@@@@@@@@修改后的@@@@@@@@@@@@@@@@@@@@@@@@@@@
        tranindexnew = 0
        # rvecsnew = [np.array([[ 0.10021112],[-0.38141065],[ 1.05268607]])]
        rvecsnew = [np.array([[0.90280267],
         [0.68637767],
         [-1.4293294]])]

        R_cam2board, _ = cv2.Rodrigues(rvecsnew[tranindexnew])

        T_cam2board = np.eye(4)
        T_cam2board[:3, :3] = R_cam2board
        T_cam2board[:3, 3] = np.zeros(3)  # 不用平移

        # 手眼标定变换（你需要根据实际调整）
        # R_cam2gripper = np.array([[0.10257118, -0.99472467, 0.0014086],
        #                           [0.99471407, 0.10256304, -0.00497433],
        #                           [0.00480361, 0.00191138, 0.99998664]])

        R_cam2gripper = np.array([[0.09884656, - 0.99501953 ,- 0.0128642],
                                  [0.99358876,  0.09940076 ,- 0.05385968],
                                  [0.05487015 ,- 0.00745788 ,0.99846565]])
        t_cam2gripper = np.zeros(3)

        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = R_cam2gripper
        T_cam2gripper[:3, 3] = t_cam2gripper

        # 取第6号有效 gripper pose（与 rvecs[6] 对应）
        # gripper_pose = selected_gripper_poses[tranindex]
        #@@@@@@@@@@@@@@@@@@@@修改后@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # gripper_pose = np.array([25.468, 529.468, 459.959, -179.317, 20.414, 90.593])
        gripper_pose = np.array([35.964, 557.786, 446.559, 160.852, 60.924, -107.456])
        r = R.from_euler('xyz', [gripper_pose[3], gripper_pose[4], gripper_pose[5]], degrees=True)
        R_gripper2base = r.as_matrix()
        # t_gripper2base = np.array(gripper_pose[:3])
        t_gripper2base = np.zeros(3)

        T_gripper2base = np.eye(4)
        T_gripper2base[:3, :3] = R_gripper2base
        T_gripper2base[:3, 3] = t_gripper2base

        # 总变换矩阵
        T_board2base = T_gripper2base @ T_cam2gripper @ T_cam2board

        # 应用变换
        merged.transform(T_board2base)

        # ======= 结束 =======

        return merged

    def scantobasesingle(self, image_dir, ply_dir, base_ply_path):
        import copy

        objp = np.zeros((self.XX * self.YY, 3), np.float32)
        objp[:, :2] = np.array([[(2 * j + i % 2) * self.L_horizontal / 2, i * self.L_vertical]
                                for i in range(self.YY) for j in range(self.XX)])

        # rvecs, tvecs, valid_indices = self.get_board2cam_v2(image_dir, objp)
        ############################
        rvecs, tvecs, valid_indices = [], [], []

        max_attempts = 18  # 最多尝试提升亮度5次
        brightness_step = 10  # 每次增加的亮度值

        raw_gray = 250 - cv2.imread(image_dir, 0)  # 原始反转灰度图
        gray = raw_gray.copy()

        ret, centers = cv2.findCirclesGrid(gray, (7, 11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        attempt = 0

        # 若初次未检测成功则尝试提升亮度
        while not ret and attempt < max_attempts:
            attempt += 1
            bright_gray = np.clip(gray + attempt * brightness_step, 0, 255).astype(np.uint8)
            bright_gray = cv2.convertScaleAbs(bright_gray, alpha=1.5, beta=20)
            ret, centers = cv2.findCirclesGrid(bright_gray, (7, 11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            if ret:
                gray = bright_gray  # 用增强后的图继续处理
                print(f"（第 {attempt} 次尝试）")
                break

        if ret and centers.shape[0] == objp.shape[0]:
            success, rvec, tvec = cv2.solvePnP(objp, centers, self.mtx, self.dist)
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)
                valid_indices.append(0)
        else:
            print(f"检测失败")
        #################################

        if not rvecs:
            print("没有有效PnP结果，退出")
            return None

        H_cam_list = [self.build_transform_matrix_from_pnp(r, t) for r, t in zip(rvecs, tvecs)]


        selected_ply_files = [ply_dir]

        # 原始的所有 gripper pose
        gripper_poses = [
            # [-119.088, 450.759, 374.552, 148.328, -2.323, 84.687]
            # [25.468, 529.468, 459.959, -179.317, 20.414, 90.593]
            [35.964, 557.786, 446.559, 160.852, 60.924, -107.456]
        ]
        selected_gripper_poses = [gripper_poses[i] for i in valid_indices]

        if len(selected_ply_files) != len(H_cam_list):
            print("有效点云与变换数量不一致")
            return None

        all_pcds = []
        for ply_file, H in zip(selected_ply_files, H_cam_list):
            pcd = o3d.io.read_point_cloud(os.path.join(ply_dir, ply_file))
            if not pcd.has_points():
                continue
            pcd.transform(H)
            points = np.asarray(pcd.points)
            mask = (points[:, 2] >= -36) & (points[:, 2] <= 36)
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
            if pcd.has_colors():
                filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
            filtered_pcd, _ = filtered_pcd.remove_statistical_outlier(16, 1.0)
            if not filtered_pcd.has_colors():
                filtered_pcd.paint_uniform_color(np.random.rand(3))
            all_pcds.append(filtered_pcd)

        if not all_pcds:
            print("无有效点云")
            return None

        # 这里加载 base_ply_path 点云替代 all_pcds[0]
        base_pcd = o3d.io.read_point_cloud(base_ply_path)
        if not base_pcd.has_points():
            print("base_ply_path 加载点云无效")
            return None
        merged = copy.deepcopy(base_pcd)

        merged_down = merged.voxel_down_sample(voxel_size=0.5)

        # 从all_pcds[1]开始融合，跳过all_pcds[0]
        for i, source in enumerate(all_pcds[0:], 1):
            source_down = source.voxel_down_sample(0.25)
            reg = o3d.pipelines.registration.registration_icp(
                source_down, merged_down, 1.0, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            print(f"[ICP] 第{i}个点云: fitness={reg.fitness:.4f}, rmse={reg.inlier_rmse:.4f}")
            source.transform(reg.transformation)
            merged += source
            merged_down = merged.voxel_down_sample(0.5)

        tranindex = 0
        # rvecs = [np.array([[0.03923832], [-0.59975117], [0.04705051]])]
        # rvecs = [np.array([[ 0.10021112],[-0.38141065],[ 1.05268607]])]

        rvecs = [np.array([[0.90280267],
         [0.68637767],
         [-1.4293294]])]

        # ======= 新增：将merged旋转到机械臂基坐标系 =======
        R_cam2board, _ = cv2.Rodrigues(rvecs[tranindex])
        T_cam2board = np.eye(4)
        T_cam2board[:3, :3] = R_cam2board
        T_cam2board[:3, 3] = np.zeros(3)  # 不用平移

        # 手眼标定变换（你需要根据实际调整）
        R_cam2gripper = np.array([[0.10257118, -0.99472467, 0.0014086],
                                  [0.99471407, 0.10256304, -0.00497433],
                                  [0.00480361, 0.00191138, 0.99998664]])
        t_cam2gripper = np.zeros(3)

        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = R_cam2gripper
        T_cam2gripper[:3, 3] = t_cam2gripper

        # 取第6号有效 gripper pose（与 rvecs[6] 对应）
        gripper_pose = selected_gripper_poses[tranindex]
        r = R.from_euler('xyz', [gripper_pose[3], gripper_pose[4], gripper_pose[5]], degrees=True)
        R_gripper2base = r.as_matrix()
        # t_gripper2base = np.array(gripper_pose[:3])
        t_gripper2base = np.zeros(3)

        T_gripper2base = np.eye(4)
        T_gripper2base[:3, :3] = R_gripper2base
        T_gripper2base[:3, 3] = t_gripper2base

        # 总变换矩阵
        T_board2base = T_gripper2base @ T_cam2gripper @ T_cam2board

        # 应用变换
        merged.transform(T_board2base)

        # ======= 结束 =======
        return merged

    def scantobasesinglerobot(self, ply_dir):
        pcd = o3d.io.read_point_cloud(ply_dir)
        gripper_poses = [
            [264.8929,-285.1852,391.0669+40,-179.7725,-1.3507,-145.9055]
        ]

        tranindex = 0

        R_cam2gripper = np.array([[0.99793716, 0.06251409, 0.0146084],
                                  [-0.06295546, 0.99750287, 0.03200991],
                                  [-0.01257085, -0.03286356, 0.99938079]])
        t_cam2gripper = np.zeros(3)

        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = R_cam2gripper
        T_cam2gripper[:3, 3] = t_cam2gripper

        # 取第6号有效 gripper pose（与 rvecs[6] 对应）
        gripper_pose = gripper_poses[tranindex]
        r = R.from_euler('xyz', [gripper_pose[3], gripper_pose[4], gripper_pose[5]], degrees=True)
        R_gripper2base = r.as_matrix()
        t_gripper2base = np.zeros(3)

        T_gripper2base = np.eye(4)
        T_gripper2base[:3, :3] = R_gripper2base
        T_gripper2base[:3, 3] = t_gripper2base

        # 总变换矩阵
        T_board2base = T_gripper2base @ T_cam2gripper

        # 应用变换
        pcd.transform(T_board2base)

        # ======= 结束 =======
        return pcd

    def get_rotation_matrix(self,axis, angle_degrees):
        angle = np.radians(angle_degrees)
        c = np.cos(angle)
        s = np.sin(angle)
        if axis == 'x':
            return np.array([[1, 0, 0],
                             [0, c, -s],
                             [0, s, c]])
        elif axis == 'y':
            return np.array([[c, 0, s],
                             [0, 1, 0],
                             [-s, 0, c]])
        elif axis == 'z':
            return np.array([[c, -s, 0],
                             [s, c, 0],
                             [0, 0, 1]])
        else:
            raise ValueError("Invalid axis")

    def interactive_path_on_pointcloudv1(self, pcd, num_samples=28):
        print("请选择多个点用于地测线（Shift+左键），按 Q 完成")

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="选择路径点")
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

        picked_ids = vis.get_picked_points()
        if len(picked_ids) < 2:
            print("至少需要选择两个点！")
            return None

        print(f"选中的点索引: {picked_ids}")
        points_np = np.asarray(pcd.points)
        picked_path = points_np[picked_ids]

        # === 在这里插入第一个点和最后一个点 ===
        picked_path = list(picked_path)  # 转为列表便于插入

        # 在第一个点和第二个点之间添加新点，沿y轴负方向8单位
        second_point = picked_path[1]
        new_first_point = second_point + np.array([0.0, -3.0, 0.0])
        picked_path.insert(1, new_first_point)

        # 在最后一个点后添加一个点，沿y轴正方向8单位
        last_point = picked_path[-1]
        new_last_point = last_point + np.array([0.0, 15.0, 0.0])
        picked_path.append(new_last_point)

        # 转回 numpy 数组
        picked_path = np.array(picked_path)

        # === 拟合三维样条曲线 ===
        path_pts = picked_path.T  # 转置为 (3, N) 形式
        tck, u = splprep(path_pts, s=0)  # s=0 表示严格通过所有点

        # 均匀采样 num_samples 个点
        u_fine = np.linspace(0, 1, num_samples)
        sampled_points = np.array(splev(u_fine, tck)).T  # 转回 (N, 3)

        return sampled_points

    def interactive_path_on_pointcloud(self, pcd, num_samples=28):
        print("请选择多个点用于地测线（Shift+左键），按 Q 完成")

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="选择路径点")
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

        picked_ids = vis.get_picked_points()
        if len(picked_ids) < 2:
            print("至少需要选择两个点！")
            return None

        print(f"选中的点索引: {picked_ids}")
        points_np = np.asarray(pcd.points)
        picked_path = points_np[picked_ids]

        # === 插入第一个和最后一个扩展点 ===
        picked_path = list(picked_path)  # 转为列表
        second_point = picked_path[1]
        new_first_point = second_point + np.array([0.0, -3.0, 0.0])
        picked_path.insert(1, new_first_point)

        last_point = picked_path[-1]
        new_last_point = last_point + np.array([0.0, 15.0, 0.0])
        picked_path.append(new_last_point)
        picked_path = np.array(picked_path)

        # === 拆分：前2个点保留，中间段拟合，最后1个点保留 ===
        fixed_start = picked_path[:2]  # 前两个原点
        fixed_end = picked_path[-1:]  # 最后一个原点
        fit_segment = picked_path[2:-1]  # 第3个到倒数第1个点

        # 曲线拟合
        path_pts = fit_segment.T
        tck, u = splprep(path_pts, s=0)

        # 计算各段分配的点数
        n_fixed_start = len(fixed_start)
        n_fixed_end = len(fixed_end)
        n_fit = num_samples - n_fixed_start - n_fixed_end
        if n_fit < 2:
            print("样本数太少，无法拟合中间段！")
            return None

        u_fine = np.linspace(0, 1, n_fit)
        fit_points_sampled = np.array(splev(u_fine, tck)).T

        # 拼接完整路径
        sampled_points = np.vstack([fixed_start, fit_points_sampled, fixed_end])

        return sampled_points

    def interactive_path_on_pointcloudv2(self, pcd, num_samples=28, smooth_factor=0.1):
        """
        一次性拟合经过所有选中点的平滑曲线
        :param pcd: Open3D 点云
        :param num_samples: 采样点数量
        :param smooth_factor: 平滑因子，0 表示严格经过所有点，值越大越平滑
        """
        print("请选择多个点用于路径（Shift+左键），按 Q 完成")

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="选择路径点")
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

        picked_ids = vis.get_picked_points()
        if len(picked_ids) < 2:
            print("至少需要选择两个点！")
            return None

        print(f"选中的点索引: {picked_ids}")
        points_np = np.asarray(pcd.points)
        picked_path = points_np[picked_ids]

        # 样条拟合
        path_pts = picked_path.T
        tck, u = splprep(path_pts, s=smooth_factor)

        # 采样
        u_fine = np.linspace(0, 1, num_samples)
        fit_points_sampled = np.array(splev(u_fine, tck)).T

        return fit_points_sampled

    def interactive_translate_path(self, pcd, path_pts, step=1.0):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="地测线可移动")

        # 主点云
        vis.add_geometry(pcd)

        # 初始化路径线
        path_line = o3d.geometry.LineSet()
        path_line.points = o3d.utility.Vector3dVector(path_pts)
        path_line.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(path_pts) - 1)])
        path_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * (len(path_pts) - 1))
        vis.add_geometry(path_line)

        # 定义回调函数
        def move(dx=0, dy=0, dz=0):
            nonlocal path_pts
            translation = np.array([dx, dy, dz])
            path_pts[1:] += translation  # 只平移第1个之后的点
            path_line.points = o3d.utility.Vector3dVector(path_pts)
            vis.update_geometry(path_line)
            print(f"平移: dx={dx}, dy={dy}, dz={dz}")

        vis.register_key_callback(ord("S"), lambda v: move(dz=-step))
        vis.register_key_callback(ord("W"), lambda v: move(dz=step))
        vis.register_key_callback(ord("A"), lambda v: move(dx=-step))
        vis.register_key_callback(ord("D"), lambda v: move(dx=step))
        vis.register_key_callback(ord("Q"), lambda v: move(dy=step))
        vis.register_key_callback(ord("E"), lambda v: move(dy=-step))
        vis.register_key_callback(256, lambda v: vis.close())  # ESC退出

        vis.run()
        vis.destroy_window()
        return path_pts


class BrushPathProcessor:
    def __init__(self):
        pass

    @staticmethod
    def load_pointcloud(filename):
        """
        只加载点云(ply)
        返回 np.ndarray 的点坐标和 open3d 点云对象
        """
        pcd = o3d.io.read_point_cloud(filename)
        V = np.asarray(pcd.points)
        return V, pcd

    @staticmethod
    def pick_vertex_indices_from_pointcloud(points):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="选择路径点")
        vis.add_geometry(points)
        vis.run()
        vis.destroy_window()

        picked_ids = vis.get_picked_points()
        if len(picked_ids) < 2:
            print("至少需要选择两个点！")
            return None

        print(f"选中的点索引: {picked_ids}")
        points_np = np.asarray(points.points)
        picked_path = points_np[picked_ids]
        return picked_path, picked_ids

    @staticmethod
    def pick_vertex_indices_from_pointcloud2(points):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="选择MarKer的中心点")
        vis.add_geometry(points)
        vis.run()
        vis.destroy_window()

        picked_ids = vis.get_picked_points()

        print(f"选中的MarKer点索引: {picked_ids}")
        points_np = np.asarray(points.points)
        picked_path = points_np[picked_ids]
        return picked_path, picked_ids

    @staticmethod
    def fit_arc_and_sample_points(path_pts, num_samples=30):
        if len(path_pts) < 3:
            raise ValueError("至少需要三个点来拟合圆弧")
        centroid = path_pts.mean(axis=0)
        pts_centered = path_pts - centroid
        U, S, Vt = np.linalg.svd(pts_centered)
        normal = Vt[2]
        plane_basis = Vt[:2]
        pts_2d = pts_centered @ plane_basis.T
        A = np.hstack([2 * pts_2d, np.ones((len(pts_2d), 1))])
        b = np.sum(pts_2d ** 2, axis=1).reshape(-1, 1)
        x = np.linalg.lstsq(A, b, rcond=None)[0].flatten()
        cx, cy, c = x
        radius = np.sqrt(c + cx ** 2 + cy ** 2)
        angles = np.arctan2(pts_2d[:, 1] - cy, pts_2d[:, 0] - cx)
        angles = np.unwrap(angles)
        start_angle = angles[0]
        end_angle = angles[-1]
        sampled_angles = np.linspace(start_angle, end_angle, num_samples)
        sampled_pts_2d = np.stack([
            cx + radius * np.cos(sampled_angles),
            cy + radius * np.sin(sampled_angles)
        ], axis=1)
        sampled_pts_3d = sampled_pts_2d @ plane_basis + centroid
        return sampled_pts_3d

    @staticmethod
    def visualize_path_and_pointcloud(points, path_pts):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

        path_line = o3d.geometry.LineSet()
        path_line.points = o3d.utility.Vector3dVector(path_pts)
        path_line.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(path_pts) - 1)])
        path_line.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(path_pts) - 1)])

        o3d.visualization.draw_geometries([pcd, path_line])

    @staticmethod
    def compute_directions(path_pts):
        directions = path_pts[1:] - path_pts[:-1]
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / (norms + 1e-8)
        directions = np.vstack([directions, directions[-1]])
        return directions

    @staticmethod
    def rotation_from_vector_to_vector(v1, v2):
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        axis = np.cross(v1, v2)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            return np.eye(3)
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R

    @staticmethod
    def rotate_pointcloud(pcd: o3d.geometry.PointCloud, R: np.ndarray, center: np.ndarray):
        pts = np.asarray(pcd.points)
        pts = pts - center
        pts = pts @ R.T
        pts = pts + center
        pcd_rot = o3d.geometry.PointCloud()
        pcd_rot.points = o3d.utility.Vector3dVector(pts)
        if pcd.has_colors():
            pcd_rot.colors = pcd.colors
        return pcd_rot

    @staticmethod
    def translate_pointcloud(pcd: o3d.geometry.PointCloud, translation: np.ndarray):
        pts = np.asarray(pcd.points)
        pts = pts + translation
        pcd_t = o3d.geometry.PointCloud()
        pcd_t.points = o3d.utility.Vector3dVector(pts)
        if pcd.has_colors():
            pcd_t.colors = pcd.colors
        return pcd_t

    def save_pointcloud_along_path_as_ply(self, pcd, path_pts, mesh_direction_pts, output_folder="saved_ply"):
        os.makedirs(output_folder, exist_ok=True)
        original_pcd = copy.deepcopy(pcd)
        v1 = np.asarray(mesh_direction_pts[0])
        v2 = np.asarray(mesh_direction_pts[1])
        model_forward = v2 - v1
        model_forward /= np.linalg.norm(model_forward)
        mesh_center = np.asarray(mesh_direction_pts[2])
        directions = self.compute_directions(path_pts)

        for i, (pos, dir_vec) in enumerate(zip(path_pts, directions)):
            tmp_pcd = copy.deepcopy(original_pcd)
            if i == 0:
                filename = os.path.join(output_folder, f"frame_{i:04d}.ply")
                o3d.io.write_point_cloud(filename, tmp_pcd)
                print(f"Saved {filename}")
                continue
            R = self.rotation_from_vector_to_vector(model_forward, dir_vec)
            tmp_pcd = self.rotate_pointcloud(tmp_pcd, R, mesh_center)
            translation = pos - mesh_center
            tmp_pcd = self.translate_pointcloud(tmp_pcd, translation)
            filename = os.path.join(output_folder, f"frame_{i:04d}.ply")
            o3d.io.write_point_cloud(filename, tmp_pcd)
            print(f"Saved {filename}")

    def save_model_along_path_as_objs_custom(self, mesh, path_pts, mesh_direction_pts, output_folder="saved_objs"):
        os.makedirs(output_folder, exist_ok=True)
        original_mesh = copy.deepcopy(mesh)
        v1 = np.asarray(mesh_direction_pts[0])
        v2 = np.asarray(mesh_direction_pts[1])
        model_forward = v2 - v1
        model_forward /= np.linalg.norm(model_forward)
        mesh_center = np.asarray(mesh_direction_pts[2])
        directions = self.compute_directions(path_pts)

        for i, (pos, dir_vec) in enumerate(zip(path_pts, directions)):
            tmp_mesh = copy.deepcopy(original_mesh)
            if i == 0:
                filename = os.path.join(output_folder, f"frame_{i:04d}.obj")
                o3d.io.write_triangle_mesh(filename, tmp_mesh)
                print(f"Saved {filename}")
                continue
            R = self.rotation_from_vector_to_vector(model_forward, dir_vec)
            tmp_mesh.rotate(R, center=mesh_center)
            tmp_mesh.translate(pos - mesh_center, relative=True)
            filename = os.path.join(output_folder, f"frame_{i:04d}.obj")
            o3d.io.write_triangle_mesh(filename, tmp_mesh)
            print(f"Saved {filename}")


def pick_points(pcd, window_name="Pick Points"):
    print(f"\nPick points in window: {window_name}")
    print("1) Please pick points using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    picked_ids = vis.get_picked_points()
    print("Picked indices:", picked_ids)
    return picked_ids


def pick_points_thread(pcd, window_name, save_path, result_holder):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    picked_ids = vis.get_picked_points()
    print(f"{window_name} Picked indices:", picked_ids)
    picked_pts = np.asarray(pcd.points)[picked_ids]
    np.savetxt(save_path, picked_pts)
    print(f"{window_name} 已保存点到 {save_path}")
    result_holder.extend(picked_ids)


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # 源点云为黄色
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # 目标点云为蓝色
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name="ICP Result")


def register_via_correspondences(source, target, source_points, target_points):
    corr = np.zeros((len(source_points), 2), dtype=int)
    corr[:, 0] = np.arange(len(source_points))
    corr[:, 1] = np.arange(len(target_points))
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source_points, target_points, o3d.utility.Vector2iVector(corr))

    print("Perform point-to-point ICP refinement")
    threshold = 3
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    print("Transformation Matrix:")
    print(reg_p2p.transformation)
    draw_registration_result(source, target, reg_p2p.transformation)
    return reg_p2p.transformation

class FaithfulPCDRegistrar:
    """
    支持 p1, p2 都为点云；
    - p1 使用沿某轴的渐变色（默认沿 z 轴），
    - p2 使用部分染色（前 color_n 个点），
    - 保持交互选点流程与之前一致。

    用法:
      r = FaithfulPCDRegistrar(pcd1, pcd2, save_debug_dir="debug")
      r.run_interactive()
    """

    def __init__(self, pcd1, pcd2, save_debug_dir=None,
                 nb_neighbors=16, std_ratio=2.0, color_n=10000,
                 icp_thresh=0.30, refine_thresh=1.0, gradient_axis=2):
        self.pcd1 = pcd1  # ptoothscan 点云
        self.pcd2 = pcd2  # ptooth 点云
        self.save_debug_dir = save_debug_dir
        if save_debug_dir:
            os.makedirs(save_debug_dir, exist_ok=True)

        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
        self.color_n = color_n
        self.icp_thresh = float(icp_thresh)
        self.refine_thresh = float(refine_thresh)
        # gradient_axis: 0=x,1=y,2=z
        self.gradient_axis = int(gradient_axis)

        # placeholders
        self.ind1 = None
        self.ind2 = None
        self.shift_to_align = None
        self.pcd2_shifted = None

    # ---------- minimal helper: color first n points ----------
    @staticmethod
    def color_partial(pcd, color, n=1000):
        points = np.asarray(pcd.points)
        N = len(points)
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            if len(colors) != N:
                colors = np.ones((N, 3), dtype=float) * 0.7
        else:
            colors = np.ones((N, 3), dtype=float) * 0.7
        colors[:min(n, N)] = color
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    # ---------- gradient color for a pointcloud ----------
    def apply_gradient(self, pcd, axis=2):
        pts = np.asarray(pcd.points)
        if pts.size == 0:
            return pcd
        vals = pts[:, axis]
        vmin = vals.min()
        vmax = vals.max()
        if np.isclose(vmin, vmax):
            t = np.zeros_like(vals)
        else:
            t = (vals - vmin) / (vmax - vmin)
        # simple colormap: blue -> cyan -> yellow -> red (approx)
        # map t in [0,1] to RGB
        r = np.minimum(2 * t, 1.0)
        g = np.minimum(2 * np.abs(t - 0.5), 1.0)
        b = np.minimum(2 * (1 - t), 1.0)
        colors = np.vstack([r, g, b]).T
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    # ---------- load and filter points directly from pcd objects ----------
    def load_and_filter(self):
        # print("处理点云 p1, p2 ...")
        # print(f"原始点数 pcd1: {len(self.pcd1.points)}, pcd2: {len(self.pcd2.points)}")

        # 对 p1 使用渐变色（沿指定轴）
        self.apply_gradient(self.pcd2, axis=self.gradient_axis)

        # 对 p2 使用前 color_n 个点染色为蓝色
        # self.color_partial(self.pcd1, [0, 0, 1], n=self.color_n)

    # ---------- initial shift ----------
    def make_initial_shift(self):
        center1 = self.pcd1.get_center()
        center2 = self.pcd2.get_center()
        self.shift_to_align = center1 - center2
        self.pcd2_shifted = o3d.geometry.PointCloud(self.pcd2)
        self.pcd2_shifted.translate(self.shift_to_align, relative=True)
        self.pcd2_shifted.translate([30, 0, 0], relative=True)
        # print("center1:", center1)
        # print("center2:", center2)
        # print("shift_to_align:", self.shift_to_align)

        # if self.save_debug_dir:
        #     np.savetxt(os.path.join(self.save_debug_dir, "center1.txt"), np.array(center1).reshape(1, 3))
        #     np.savetxt(os.path.join(self.save_debug_dir, "center2.txt"), np.array(center2).reshape(1, 3))
        #     np.savetxt(os.path.join(self.save_debug_dir, "shift_to_align.txt"),
        #                np.array(self.shift_to_align).reshape(1, 3))

    def make_initial_shiftzero(self):
        center1 = self.pcd1.get_center()
        center2 = self.pcd2.get_center()
        self.shift_to_align = center1 - center2
        self.pcd2_shifted = o3d.geometry.PointCloud(self.pcd2)
        self.pcd2_shifted.translate(self.shift_to_align, relative=True)
        self.pcd2_shifted.translate([0, 0, 0], relative=True)

    # ---------- interactive pick ----------
    def pick_correspondences(self, window_name="Pick Correspondences"):
        # fused used for picking should be pcd1 + pcd2_shifted so indices are coherent
        fused = self.pcd1 + self.pcd2_shifted

        print("打开点云选择窗口，请按原来习惯选点（先 pcd1 的点，然后 pcd2 的点），完成后按 Q。")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=window_name)
        vis.add_geometry(fused)
        vis.run()
        vis.destroy_window()
        picked_ids = vis.get_picked_points()
        print("Picked indices:", picked_ids)
        if self.save_debug_dir:
            np.savetxt(os.path.join(self.save_debug_dir, "picked_indices.txt"), np.array(picked_ids, dtype=np.int32),
                       fmt="%d")
        return picked_ids, fused

    # ---------- compute initial transform ----------
    def compute_trans_init_from_picks(self, picked_ids, fused,mode):
        if len(picked_ids) == 0:
            raise RuntimeError("未选取点 (picked_ids 为空)。")

        N = len(picked_ids) // 2
        ids1 = picked_ids[:N]
        ids2 = picked_ids[N:2 * N]
        points = np.asarray(fused.points)

        target_points = points[ids1]
        # 还原 pcd2 的原始坐标：减去 X 轴平移(100) 与 center 对齐移动(shift_to_align)
        if(mode ==0):
            source_points = points[ids2] - np.array([30, 0, 0]) - self.shift_to_align
        else:
            source_points = points[ids2] - np.array([0, 0, 0]) - self.shift_to_align
        src_corr = o3d.geometry.PointCloud()
        tgt_corr = o3d.geometry.PointCloud()
        src_corr.points = o3d.utility.Vector3dVector(source_points)
        tgt_corr.points = o3d.utility.Vector3dVector(target_points)

        corr = np.zeros((N, 2), dtype=np.int32)
        corr[:, 0] = np.arange(N, dtype=np.int32)
        corr[:, 1] = np.arange(N, dtype=np.int32)

        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        trans_init = p2p.compute_transformation(src_corr, tgt_corr, o3d.utility.Vector2iVector(corr))

        # if self.save_debug_dir:
        #     np.savetxt(os.path.join(self.save_debug_dir, "trans_init.txt"), trans_init.reshape(4, 4))
        return trans_init

    # ---------- ICP ----------
    def run_icp_with_init(self, trans_init):
        reg_p2p = o3d.pipelines.registration.registration_icp(
            self.pcd2, self.pcd1, self.icp_thresh, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        pcd2_aligned = copy.deepcopy(self.pcd2)
        pcd2_aligned.transform(reg_p2p.transformation)

        # 不再使用这个后面的优化
        reg_refine = o3d.pipelines.registration.registration_icp(
            pcd2_aligned, self.pcd1, self.refine_thresh, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        pcd2_aligned_refined = copy.deepcopy(pcd2_aligned)
        pcd2_aligned_refined.transform(reg_refine.transformation)

        # fused_final = pcd1_aligned_refined + self.pcd2
        # o3d.visualization.draw_geometries([fused_final], window_name="Fused Point Cloud (After ICP Refinement)")

        return pcd2_aligned_refined
        # return pcd2_aligned

    def run_multiple_icp(self, trans_init, num_iterations=3):
        pcd2_aligned = copy.deepcopy(self.pcd2)

        # 初始配准
        reg_p2p = o3d.pipelines.registration.registration_icp(
            self.pcd2, self.pcd1, self.icp_thresh, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        pcd2_aligned.transform(reg_p2p.transformation)
        last_transformation = reg_p2p.transformation

        for i in range(num_iterations):
            print(f"Running ICP iteration {i + 1}...")

            # 每次迭代都用上次的变换结果作为初始化
            reg_refine = o3d.pipelines.registration.registration_icp(
                pcd2_aligned, self.pcd1, self.refine_thresh, last_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

            pcd2_aligned.transform(reg_refine.transformation)
            last_transformation = reg_refine.transformation

        # 最终配准后的点云
        pcd2_aligned_refined = copy.deepcopy(pcd2_aligned)
        pcd2_aligned_refined.transform(last_transformation)

        # 返回最终配准的点云
        return pcd2_aligned_refined

    # ---------- one-line runner ----------
    def run_interactive(self,mode=0):
        self.load_and_filter()
        if(mode == 0):
            self.make_initial_shift()
        else:
            self.make_initial_shiftzero()
        picked_ids, fused = self.pick_correspondences()
        trans_init = self.compute_trans_init_from_picks(picked_ids, fused,mode)

        # print("brush and brush trans init",trans_init)

        reg_refine = self.run_icp_with_init(trans_init)

        return reg_refine


def register_pointclouds_interactive(pcd1, pcd2, translation_step=0.05, rotation_step=0.05):
    """
    通过交互式操作将pcd2配准到pcd1

    参数:
    pcd1: 目标点云
    pcd2: 需要变换的点云
    translation_step: 平移步长
    rotation_step: 旋转步长(角度)

    返回:
    transformed_pcd2: 变换后的pcd2
    transformation: 应用的变换矩阵
    """

    # 复制点云以避免修改原始数据
    pcd1_copy = pcd1
    pcd2_copy = pcd2

    # 为区分两个点云，设置不同颜色
    # pcd1_copy.paint_uniform_color([1, 0, 0])  # 红色 - 目标点云
    # pcd2_copy.paint_uniform_color([0, 1, 0])  # 绿色 - 待变换点云

    # 初始变换矩阵（单位矩阵）
    transformation = np.eye(4)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="点云配准 - 按ESC退出")

    # 添加点云到可视化
    vis.add_geometry(pcd1_copy)
    vis.add_geometry(pcd2_copy)

    def update_pcd2():
        """更新pcd2的变换"""
        pcd2_copy.transform(transformation)
        vis.update_geometry(pcd2_copy)
        # 重置变换矩阵为单位矩阵，因为点云已经应用了变换
        transformation[:] = np.eye(4)

    def get_rotation_matrix(axis, angle_degrees):
        """获取绕指定轴的旋转矩阵"""
        angle_rad = np.radians(angle_degrees)
        if axis == 'x':
            R = np.array([[1, 0, 0],
                          [0, np.cos(angle_rad), -np.sin(angle_rad)],
                          [0, np.sin(angle_rad), np.cos(angle_rad)]])
        elif axis == 'y':
            R = np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                          [0, 1, 0],
                          [-np.sin(angle_rad), 0, np.cos(angle_rad)]])
        elif axis == 'z':
            R = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                          [np.sin(angle_rad), np.cos(angle_rad), 0],
                          [0, 0, 1]])
        return R

    # --- 平移函数 ---
    def translate(dx=0, dy=0, dz=0):
        nonlocal transformation
        T = np.eye(4)
        T[:3, 3] = [dx, dy, dz]
        transformation = T @ transformation
        update_pcd2()

    # --- 旋转函数 ---
    def rotate(axis, direction=1):
        nonlocal transformation
        R = get_rotation_matrix(axis, rotation_step * direction)
        T = np.eye(4)
        T[:3, :3] = R
        transformation = T @ transformation
        update_pcd2()

    # --- 重置函数 ---
    def reset_transform():
        nonlocal transformation
        # 重置pcd2到初始状态
        pcd2_copy.points = pcd2.points
        transformation = np.eye(4)
        vis.update_geometry(pcd2_copy)
        print("已重置变换")

    # --- 键盘绑定 ---

    # 平移绑定：WASD + RF (上下)
    vis.register_key_callback(ord("D"), lambda vis: translate(dx=translation_step))  # +X
    vis.register_key_callback(ord("A"), lambda vis: translate(dx=-translation_step))  # -X
    vis.register_key_callback(ord("W"), lambda vis: translate(dy=translation_step))  # +Y
    vis.register_key_callback(ord("S"), lambda vis: translate(dy=-translation_step))  # -Y
    vis.register_key_callback(ord("Q"), lambda vis: translate(dz=translation_step))  # +Z
    vis.register_key_callback(ord("E"), lambda vis: translate(dz=-translation_step))  # -Z

    # 旋转绑定：J/L (绕Z), I/K (绕Y), U/O (绕X)
    vis.register_key_callback(ord("J"), lambda vis: rotate('z', direction=1))  # 绕Z+
    vis.register_key_callback(ord("L"), lambda vis: rotate('z', direction=-1))  # 绕Z-
    vis.register_key_callback(ord("I"), lambda vis: rotate('y', direction=1))  # 绕Y+
    vis.register_key_callback(ord("K"), lambda vis: rotate('y', direction=-1))  # 绕Y-
    vis.register_key_callback(ord("U"), lambda vis: rotate('x', direction=1))  # 绕X+
    vis.register_key_callback(ord("O"), lambda vis: rotate('x', direction=-1))  # 绕X-

    # 重置绑定：T
    vis.register_key_callback(ord("T"), lambda vis: reset_transform())

    print("=" * 50)
    print("点云配准控制说明:")
    print("平移: D/A (X轴), W/S (Y轴), R/F (Z轴)")
    print("旋转: J/L (绕Z轴), I/K (绕Y轴), U/O (绕X轴)")
    print("重置: T")
    print("退出: ESC")
    print("=" * 50)

    vis.run()
    vis.destroy_window()

    # 应用最终的变换到原始pcd2
    final_pcd2 = pcd2
    final_pcd2.transform(transformation)

    return final_pcd2, transformation

def rotate_point_cloud_with_selected_axis(ply_path, save_prefix="rotated"):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
       print("Error: empty point cloud.")
       return
    print("Select exactly 2 points in the popup window (Shift+LeftClick).")
    print("Press 'Q' to finish picking.")

    # Point picking
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window("选择两个点作为旋转轴向量")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    picked_ids = vis.get_picked_points()
    print("Picked point indices:", picked_ids)

    # 保存到txt
    np.savetxt("D:/UsmileProject/hand_eye_calibration/picked_rotation_directionid.txt", np.array(picked_ids, dtype=int), fmt="%d")
    print("Picked point indices saved to picked_rotation_directionid.txt")

    return picked_ids
def pick_three_paths(processor, selected):
    """
    连续执行三次点选 + 拟合 + 保存
    保存路径分别为 outer / upper / inner
    """

    names = ["outer", "upper", "inner"]
    base_dir = "D:/UsmileProject/hand_eye_calibration/"

    for name in names:
        print(f"\n====== 现在开始选择: {name} 点 ======")

        # 1. 选点
        raw_path_pts, picked_ids = processor.pick_vertex_indices_from_pointcloud(selected)
        print(f"[{name}] 选点数量: {len(raw_path_pts)}")

        if len(raw_path_pts) < 3:
            print(f"[{name}] 选点太少，跳过拟合！")
            continue

        # 3. 保存路径
        save_path = f"{base_dir}/{name}/picked_path.npy"
        np.save(save_path, raw_path_pts)
        print(f"[{name}] 已保存到: {save_path}")

    print("\n=== 已完成 outer / upper / inner 三条路径的选择并保存 ===")

def save_mesh_direction_ids(mesh_direction_ids, save_path="mesh_direction_ids.txt"):
    mesh_direction_ids = np.array(mesh_direction_ids, dtype=np.int32)
    np.savetxt(save_path, mesh_direction_ids, fmt="%d")
    print(f"[INFO] mesh_direction_ids 已保存到 {save_path}")

def load_mesh_direction_ids(load_path="mesh_direction_ids.txt"):
    mesh_direction_np = np.loadtxt(load_path, dtype=np.int32)
    print(f"[INFO] 已从 {load_path} 加载 mesh_direction_ids")
    return mesh_direction_np

def load_picked_path_by_keyboard():
    print("请选择要加载的轨迹点路径:")
    print("1: 牙外侧")
    print("2: 牙上侧")
    print("3: 牙内侧")

    choice = input("请输入数字 (1/2/3): ").strip()

    # ===== 三个轨迹路径 =====
    base_dir = "D:/UsmileProject/hand_eye_calibration"

    path_map = {
        "1": os.path.join(base_dir, "outer", "picked_path.npy"),     # 牙外侧
        "2": os.path.join(base_dir, "upper", "picked_path.npy"),     # 牙上侧
        "3": os.path.join(base_dir, "inner", "picked_path.npy")      # 牙内侧
    }

    if choice not in path_map:
        raise ValueError("输入错误！只能输入 1 / 2 / 3")

    path_file = path_map[choice]

    if not os.path.exists(path_file):
        raise FileNotFoundError(f"文件不存在: {path_file}")

    print(f"加载轨迹文件: {path_file}")

    picked_path = np.load(path_file)
    return picked_path



def register_and_segment(target, source, template_path, output_path, distance_threshold=0.8):
    """
    target: 目标点云对象 (pcd2)
    source: 待分割的源点云对象 (pcd1_final)
    template_path: 模板点云路径
    output_path: 分割出的点云保存路径
    distance_threshold: 距离阈值(mm)

    Returns:
        segmented_part: 提取出来的点云 (红色)
        remaining_part: 剩余的点云 (灰色)
    """
    # 1. 加载模板
    if not os.path.exists(template_path):
        print(f"错误：找不到模板文件 {template_path}")
        return None, None

    print("正在加载模板点云...")
    template = o3d.io.read_point_cloud(template_path)

    # 2. 建立 KDTree 进行最近邻搜索
    print(f"正在基于搜索半径 {distance_threshold}mm 提取点云...")
    template_tree = o3d.geometry.KDTreeFlann(template)

    points = np.asarray(source.points)
    selected_indices = []

    # 3. 遍历 source 的每个点，检查它到 template 的距离
    for i, point in enumerate(points):
        # search_radius_vector_3d 返回 (点的个数, 索引, 距离平方)
        [k, _, _] = template_tree.search_radius_vector_3d(point, distance_threshold)
        if k > 0:
            selected_indices.append(i)

    # 4. 根据索引提取点
    # 提取匹配到的部分
    segmented_part = source.select_by_index(selected_indices)
    # 提取剩余的部分 (使用 invert=True)
    remaining_part = source.select_by_index(selected_indices, invert=True)

    # 5. 保存结果 (仅保存提取出来的部分)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    o3d.io.write_point_cloud(output_path, segmented_part)
    print(f"分割完成！")
    print(f" - 提取点数: {len(selected_indices)}")
    print(f" - 剩余点数: {len(points) - len(selected_indices)}")
    print(f" - 保存至: {output_path}")

    # 返回两个部分
    return segmented_part, remaining_part


def segment_by_mesh_surface(source, template_mesh_path, output_path, distance_threshold=0.5):
    """
    将 Mesh 采样为点云，提取其表面附近的所有点
    """
    # 1. 加载并采样 Mesh
    source = o3d.io.read_triangle_mesh(source)
    mesh = o3d.io.read_triangle_mesh(template_mesh_path)
    # 根据面积采样足够多的点（如 50000 个），使其形成密集的“面覆盖”
    template_pcd = mesh.sample_points_uniformly(number_of_points=50000)

    # 2. 建立 KDTree
    template_tree = o3d.geometry.KDTreeFlann(template_pcd)
    source_points = np.asarray(source.points)
    selected_indices = []

    print(f"正在进行面覆盖提取，阈值: {distance_threshold}mm...")
    for i, pt in enumerate(source_points):
        # 寻找距离采样面在阈值范围内的点
        [k, _, _] = template_tree.search_radius_vector_3d(pt, distance_threshold)
        if k > 0:
            selected_indices.append(i)

    # 3. 提取并保存
    segmented_part = source.select_by_index(selected_indices)
    o3d.io.write_point_cloud(output_path, segmented_part)

    # 可视化
    template_pcd.paint_uniform_color([1, 0, 0])  # 模板采样显示为红色
    segmented_part.paint_uniform_color([0, 1, 0])  # 提取结果绿色
    o3d.visualization.draw_geometries([source, segmented_part, template_pcd])


def segment_by_mesh_volume(source, template_mesh_path, output_path):
    """
    使用 Mesh 的体积覆盖来分割点云
    """
    # 1. 加载 Mesh 并转换为 Tensor 场景进行射线追踪
    print("正在加载并处理 Mesh 模板...")
    mesh = o3d.io.read_triangle_mesh(template_mesh_path)

    # 必须是 Tensor 形式才能使用 RaycastingScene
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(t_mesh)

    # 2. 将源点云转换为 Tensor
    print("正在计算点与 Mesh 的空间关系...")
    points_tensor = o3d.core.Tensor(np.asarray(source.points), dtype=o3d.core.Dtype.Float32)

    # 3. 计算 Occupancy (占用率)
    # 对于闭合 Mesh，返回 1 表示在内部，0 表示在外部
    # 对于非闭合 Mesh，它会根据最近表面法线判定
    occupancy = scene.compute_occupancy(points_tensor)

    # 转换为 numpy 布尔索引
    inside_mask = occupancy.numpy().astype(bool)
    inside_indices = np.where(inside_mask)[0]

    # 4. 提取并保存
    segmented_part = source.select_by_index(inside_indices)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    o3d.io.write_point_cloud(output_path, segmented_part)
    print(f"分割完成！边界内提取了 {len(inside_indices)} 个点。")

    # 可视化对比
    source.paint_uniform_color([0.5, 0.5, 0.5])
    segmented_part.paint_uniform_color([0, 1, 0])  # 内部点绿色
    o3d.visualization.draw_geometries([source, segmented_part], window_name="Mesh 边界内提取")

    return segmented_part



def segment_and_visualize(target_mesh_path, source_pcd_path, output_path, distance_threshold=0.1):
    """
    加载 Mesh 和 点云，进行精准投影分割，保存并显示结果
    """
    # 1. 加载数据
    # print("正在加载数据...")
    # if not os.path.exists(target_mesh_path) or not os.path.exists(source_pcd_path):
    #     print("错误：文件路径不存在，请检查 path2new 和 source 路径")
    #     return

    # 加载为 Mesh 对象 (pcd2)
    # pcd2_mesh = o3d.io.read_triangle_mesh(target_mesh_path)
    pcd2_mesh = target_mesh_path

    # 加载待分割的点云 (pcd1_final)
    source_pcd = o3d.io.read_point_cloud(source_pcd_path)

    if pcd2_mesh.is_empty() or source_pcd.is_empty():
        print("错误：读取的文件内容为空")
        return

    # 2. 建立 Raycasting 场景进行精准投影
    print("正在进行 Mesh 投影计算...")
    scene = o3d.t.geometry.RaycastingScene()
    # 转换 legacy mesh 到 tensor mesh
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(pcd2_mesh)
    _ = scene.add_triangles(t_mesh)

    # 将点云转为 Tensor
    points_t = o3d.core.Tensor(np.asarray(source_pcd.points), dtype=o3d.core.Dtype.Float32)

    # 计算距离 (点到 Mesh 表面最近处的距离)
    distances = scene.compute_distance(points_t).numpy()

    # 3. 筛选并提取
    selected_indices = np.where(distances <= distance_threshold)[0]
    segmented_part = source_pcd.select_by_index(selected_indices.tolist())

    # 4. 保存结果
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    o3d.io.write_point_cloud(output_path, segmented_part)
    print(f"--- 分割完成 ---")
    print(f"原始点数: {len(source_pcd.points)}")
    print(f"提取点数: {len(segmented_part.points)}")
    print(f"已保存至: {output_path}")

    # 5. 可视化对比
    print("正在打开可视化窗口...")

    # 原始点云设为灰色
    source_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    # 提取出来的点云设为红色，方便观察轮廓是否精准
    segmented_part.paint_uniform_color([1.0, 0, 0])

    # 可选：将 Mesh 也显示出来（设为半透明蓝色轮廓）
    pcd2_mesh.compute_vertex_normals()
    pcd2_mesh.paint_uniform_color([0, 0.6, 0.9])

    # 窗口显示
    # 如果你想看点云和 Mesh 的重合度，把 pcd2_mesh 加上
    o3d.visualization.draw_geometries([source_pcd, segmented_part],
                                      window_name="精准分割结果 (灰色:原始, 红色:提取)",
                                      width=1200, height=800)

def get_samples_interactively(pcd):
    """
    弹出窗口供用户选点
    按 'K' 键锁定选点，按 'Q' 退出窗口
    """
    print("\n[交互提示]：")
    print("1. 按住 'Shift + 左键' 选择要剔除的颜色区域（选取几个点即可）")
    print("2. 选完后按 'K' 确认选点")
    print("3. 按 'Q' 关闭窗口并开始计算")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="选取要剔除的颜色点")
    vis.add_geometry(pcd)
    vis.run()  # 用户在这里进行交互
    vis.destroy_window()

    picked_indices = vis.get_picked_points()
    if not picked_indices:
        return None

    # 获取选中点的 RGB 并转为 LAB
    picked_colors = np.asarray(pcd.colors)[picked_indices].reshape(-1, 1, 3)
    picked_lab = color.rgb2lab(picked_colors).reshape(-1, 3)
    return picked_lab

def interactive_segmentation(input_path):
    if not os.path.exists(input_path):
        print(f"未找到文件: {input_path}")
        return

    # 1. 加载 Mesh
    original_mesh = o3d.io.read_triangle_mesh(input_path)
    if not original_mesh.has_vertex_colors():
        print("错误：该 Mesh 缺少顶点颜色。")
        return

    original_total_area = original_mesh.get_surface_area()
    current_mesh = original_mesh

    # 用于存储每一轮被剔除的面积信息
    removal_history = []

    print(f"模型加载成功！")
    print(f"原始总面积: {original_total_area:.4f}")

    while True:
        # 2. 转换为 PointCloud 用于交互选点
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = current_mesh.vertices
        temp_pcd.colors = current_mesh.vertex_colors

        sampled_lab = get_samples_interactively(temp_pcd)

        if sampled_lab is None:
            print("未选取任何点，请重新尝试。")
            continue

        # 3. 颜色距离计算 (LAB 空间)
        current_rgb = np.asarray(current_mesh.vertex_colors).reshape(-1, 1, 3)
        current_lab = color.rgb2lab(current_rgb).reshape(-1, 3)

        color_threshold = 12.0
        dists = cdist(current_lab, sampled_lab, metric='euclidean')
        min_dists = np.min(dists, axis=1)

        # 4. 执行分割
        keep_indices = np.where(min_dists >= color_threshold)[0]
        temp_remaining_mesh = current_mesh.select_by_index(keep_indices)

        # --- 核心修改：计算本轮剔除的数值 ---
        before_area = current_mesh.get_surface_area()
        after_area = temp_remaining_mesh.get_surface_area()
        removed_area = before_area - after_area
        # 避免浮点数误差出现负数
        removed_area = max(0, removed_area)

        current_round_clean_rate = (removed_area / original_total_area) * 100

        print(f"\n--- 本轮处理预览 ---")
        print(f"本轮剔除面积: {removed_area:.4f}")
        print(f"本轮贡献清洁度: {current_round_clean_rate:.2f}%")
        print(f"剩余总面积: {after_area:.4f}")

        o3d.visualization.draw_geometries([temp_remaining_mesh], window_name="预览：剔除后的效果")

        # 5. 用户决策
        user_input = input(
            "满意本轮效果吗？\n(y: 确定并结束 / n: 确定并继续下一轮 / r: 撤销本轮重选): "
        ).lower().strip()

        if user_input == 'y' or user_input == 'n':
            # 确认保留本轮操作
            removal_history.append({
                "removed_area": removed_area,
                "percentage": current_round_clean_rate
            })
            current_mesh = temp_remaining_mesh

            if user_input == 'y':
                break
            else:
                print(f"--- 进度已保存。当前累计已剔除: {sum(h['percentage'] for h in removal_history):.2f}% ---")

        elif user_input == 'r':
            print("--- 已重置本轮操作 ---")
            continue
        else:
            print("无效输入，请重新选择。")

    # 6. 最终结算
    final_area = current_mesh.get_surface_area()
    total_removed_rate = sum(h['percentage'] for h in removal_history)
    # 最终清洁度（剩余部分占比）
    final_cleanliness_rate = (final_area / original_total_area) * 100

    print("\n" + "=" * 45)
    print("【清洗任务最终报告】")
    print(f"1. 原始总面积:   {original_total_area:.4f}")
    print(f"2. 最终剩余面积: {final_area:.4f}")
    print("-" * 20)
    for i, history in enumerate(removal_history):
        print(f"第 {i + 1} 轮剔除占比: {history['percentage']:.2f}%")
    print("-" * 20)
    print(f"总计剔除比例:    {total_removed_rate:.2f}%")
    print(f"最终清洁度得分:  {final_cleanliness_rate:.2f}% (剩余面积占比)")
    print("=" * 45)

    o3d.io.write_triangle_mesh("final_cleaned_mesh.ply", current_mesh)

# 利用分割好的面片来获取点云
def register_and_segment2(source, template_path, output_path, distance_threshold=0.8):
    """
    source: 变换后的完整源点云 (pcd1_final)
    template_path: 牙齿/牙龈模板路径
    output_path: 单个分割结果保存路径
    """
    print(f"正在处理模板: {os.path.basename(template_path)}")
    template = o3d.io.read_point_cloud(template_path)

    # 计算源点云中每个点到模板点云的最短距离
    # 这是向量化操作，比手动循环 KDTree 快非常多
    dists = source.compute_point_cloud_distance(template)
    dists = np.asarray(dists)

    # 获取距离小于阈值的顶点索引
    selected_indices = np.where(dists < distance_threshold)[0]

    if len(selected_indices) == 0:
        print(f"警告：模板 {os.path.basename(template_path)} 未匹配到任何点。")
        return None, []

    # 提取分割的点云用于实时预览或保存
    segmented_part = source.select_by_index(selected_indices)

    # 保存单个分割的点云结果
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    o3d.io.write_point_cloud(output_path, segmented_part)

    print(f"成功分割：提取 {len(selected_indices)} 个点。")
    return segmented_part, selected_indices

def map_mesh_and_color_to_pcd(mesh_path, pcd_obj, output_path, smoothing_iterations=3):
    """
    mesh_path: 原始模板 Mesh 的路径 (用于获取拓扑)
    pcd_obj: 已经分割出来的点云对象 (已经带有颜色信息)
    output_path: 保存后的 Mesh 路径
    """
    print(f"正在映射 Mesh 拓扑: {os.path.basename(mesh_path)}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    if not mesh.has_triangles():
        print(f"错误：{mesh_path} 不包含面片！")
        return None

    mesh.compute_vertex_normals()
    # 使用传入的点云对象建立 KDTree
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_obj)
    mesh_vertices = np.asarray(mesh.vertices)
    pcd_points = np.asarray(pcd_obj.points)
    pcd_colors = np.asarray(pcd_obj.colors) if pcd_obj.has_colors() else None

    new_vertices = np.zeros_like(mesh_vertices)
    new_colors = np.zeros_like(mesh_vertices)

    for i in range(len(mesh_vertices)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(mesh_vertices[i], 1)
        nearest_idx = idx[0]
        new_vertices[i] = pcd_points[nearest_idx]
        if pcd_colors is not None:
            new_colors[i] = pcd_colors[nearest_idx]

    mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    if pcd_colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)

    if smoothing_iterations > 0:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=smoothing_iterations)

    mesh.compute_vertex_normals()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_triangle_mesh(output_path, mesh)
    return mesh

class AdvancedCleanerWindow(QMainWindow):
    def __init__(self, input_path, output_dir, color_threshold=12.0):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.base_threshold = float(color_threshold)  # 主选点界面的阈值
        self.fine_threshold = float(color_threshold) * 0.5  # 已选点 / 剩余点界面的精细阈值（可单独调整）
        self.display_contrast = 1.45  # 仅用于显示增强：白更白、黑更黑（不影响选点计算）
        self.mesh_name = os.path.basename(input_path)
        self.result_data = None

        self._loaded_mesh = None  # 必须先初始化，后面的代码依赖 self._loaded_mesh

        # 1. 加载数据：优先作为三角网格加载（PLY 文件通常由 map_mesh_and_color_to_pcd
        # 用 write_triangle_mesh 保存，包含完整面拓扑），若无面则降级为点云。
        if not os.path.exists(input_path):
            print(f"未找到文件: {input_path}")
            return

        print("正在加载点云数据...")
        _tmp_mesh = o3d.io.read_triangle_mesh(input_path)
        if _tmp_mesh.has_triangles():
            # 有三角面：用 mesh 的顶点 + 颜色构造点云用于显示
            self.original_pcd = o3d.geometry.PointCloud()
            self.original_pcd.points = _tmp_mesh.vertices
            if _tmp_mesh.has_vertex_colors():
                self.original_pcd.colors = _tmp_mesh.vertex_colors
            self._loaded_mesh = _tmp_mesh  # 保存 mesh 引用供面级清洁度计算使用
            print(f"  已作为三角 Mesh 加载（面数={len(_tmp_mesh.triangles)}）")
        else:
            # 无三角面，降级为点云
            self.original_pcd = o3d.io.read_point_cloud(input_path)
            self._loaded_mesh = None
            print("  降级为点云加载（无三角面）")

        if not self.original_pcd.has_colors():
            print(f"跳过：{self.mesh_name} 缺少顶点颜色。")
            return

        # 提取顶点和颜色
        self.points = np.asarray(self.original_pcd.points)
        self.original_colors = np.asarray(self.original_pcd.colors)
        self.total_points_count = len(self.points)

        # 读取原始三角网格以获得真实表面积（mm²）和面级灰度数据
        # 清洁度 = Σ(remaining_face_area × grayscale) / Σ(segment_face_area × grayscale)
        # 其中 grayscale = 1 - luminance（黑色=1.0，白色=0.0），颜色越深权重越大
        self._has_mesh_data = False
        self.original_area = 0.0
        self._mesh_verts = None
        self._mesh_faces = None
        self._face_grays = None

        # 优先使用已加载的 mesh（PLY 包含三角面的情况）
        if self._loaded_mesh is not None:
            mesh_verts = np.asarray(self._loaded_mesh.vertices)
            mesh_faces = np.asarray(self._loaded_mesh.triangles)
            self.original_area = float(self._loaded_mesh.get_surface_area())
            if self._loaded_mesh.has_vertex_colors():
                mesh_rgb = np.asarray(self._loaded_mesh.vertex_colors)
            else:
                mesh_rgb = self.original_colors

            n_faces = len(mesh_faces)
            face_grays = np.zeros(n_faces, dtype=np.float64)
            for fi, tri in enumerate(mesh_faces):
                colors_for_face = mesh_rgb[tri]
                lum = (colors_for_face[:, 0] * 0.299 +
                       colors_for_face[:, 1] * 0.587 +
                       colors_for_face[:, 2] * 0.114)
                face_grays[fi] = 1.0 - float(np.mean(lum))

            self._mesh_verts = mesh_verts
            self._mesh_faces = mesh_faces
            self._face_grays = face_grays
            self._has_mesh_data = True
            print(f"  [清洁度] 直接使用 Mesh 数据: 面积={self.original_area:.4f} mm², "
                  f"面数={n_faces}, gray范围=[{face_grays.min():.3f}, {face_grays.max():.3f}]")
        else:
            # 无三角面：尝试从文件名反推模板 mesh 路径
            fname = os.path.basename(input_path)
            if fname.startswith("Mesh_"):
                fname_in_template = fname.replace("Mesh_", "", 1)
                search_dirs = [
                    os.path.join(os.path.dirname(self.output_dir), "template_ply"),
                    os.path.join(os.path.dirname(self.output_dir), "templates"),
                    os.path.join(os.path.dirname(os.path.dirname(self.output_dir)), "segmentationsply"),
                ]
                template_mesh = None
                for d in search_dirs:
                    candidate = os.path.join(d, fname_in_template)
                    if os.path.exists(candidate):
                        template_mesh = o3d.io.read_triangle_mesh(candidate)
                        print(f"  [清洁度] 找到模板 Mesh: {candidate}")
                        break
                    if fname_in_template.startswith("Seg_"):
                        alt = os.path.join(d, fname_in_template.replace("Seg_", "", 1))
                        if os.path.exists(alt):
                            template_mesh = o3d.io.read_triangle_mesh(alt)
                            print(f"  [清洁度] 找到模板 Mesh: {alt}")
                            break

                if template_mesh is not None and template_mesh.has_triangles():
                    self.original_area = float(template_mesh.get_surface_area())
                    self._mesh_verts = np.asarray(template_mesh.vertices)
                    self._mesh_faces = np.asarray(template_mesh.triangles)
                    if template_mesh.has_vertex_colors():
                        mesh_rgb = np.asarray(template_mesh.vertex_colors)
                    else:
                        mesh_rgb = self.original_colors
                    n_faces = len(self._mesh_faces)
                    face_grays = np.zeros(n_faces, dtype=np.float64)
                    for fi, tri in enumerate(self._mesh_faces):
                        colors_for_face = mesh_rgb[tri]
                        lum = (colors_for_face[:, 0] * 0.299 +
                               colors_for_face[:, 1] * 0.587 +
                               colors_for_face[:, 2] * 0.114)
                        face_grays[fi] = 1.0 - float(np.mean(lum))
                    self._face_grays = face_grays
                    self._has_mesh_data = True
                    print(f"  [清洁度] 使用模板 Mesh: 面积={self.original_area:.4f} mm², "
                          f"面数={n_faces}, gray范围=[{face_grays.min():.3f}, {face_grays.max():.3f}]")
                else:
                    print(f"  [清洁度] 未能找到对应模板 Mesh，跳过面级清洁度计算")

        print(f"成功加载点云: {self.mesh_name}, 总点数: {self.total_points_count}")

        # 预计算 LAB 空间
        print("正在预计算色彩空间，请稍候...")
        rgb_view = self.original_colors.reshape(-1, 1, 3)
        self.pcd_lab = color.rgb2lab(rgb_view).reshape(-1, 3)

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

    @staticmethod
    def _compute_face_areas(verts, faces):
        """逐三角面片计算面积，返回 (n_faces,)"""
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        ab, ac = v1 - v0, v2 - v0
        cross = np.cross(ab, ac)
        return np.linalg.norm(cross, axis=1) * 0.5

    def _set_cloud_colors(self, colors_01):
        """同步点云显示颜色（映射到 0-255 字节范围）"""
        enhanced = self._enhance_display_colors(colors_01)
        self.cloud.point_data["colors"] = (enhanced * 255).astype(np.uint8)

    def _enhance_display_colors(self, colors_01):
        """
        显示增强（不改原始数据）：提升黑白对比度，便于筛选时观察边界。
        线性对比度公式：out = (in - 0.5) * contrast + 0.5
        """
        c = np.asarray(colors_01, dtype=np.float64)
        c = (c - 0.5) * self.display_contrast + 0.5
        return np.clip(c, 0.0, 1.0)

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

        # --- 阈值调整区 ---
        thresh_title = QLabel("阈值调整")
        thresh_title.setStyleSheet("font-weight: bold; color: #333; font-size: 13px; margin-top: 18px;")
        panel_layout.addWidget(thresh_title)

        form_layout = QFormLayout()

        self.spin_base = QDoubleSpinBox()
        self.spin_base.setRange(0.1, 100.0)
        self.spin_base.setSingleStep(0.5)
        self.spin_base.setDecimals(2)
        self.spin_base.setValue(self.base_threshold)
        self.spin_base.valueChanged.connect(self._on_base_threshold_changed)
        form_layout.addRow("主选点界面:", self.spin_base)

        self.spin_fine = QDoubleSpinBox()
        self.spin_fine.setRange(0.1, 100.0)
        self.spin_fine.setSingleStep(0.5)
        self.spin_fine.setDecimals(2)
        self.spin_fine.setValue(self.fine_threshold)
        self.spin_fine.valueChanged.connect(self._on_fine_threshold_changed)
        form_layout.addRow("已选/剩余界面:", self.spin_fine)

        panel_layout.addLayout(form_layout)

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

    def _on_base_threshold_changed(self, value):
        """主选点界面阈值变化：实时重算红染范围并刷新视图"""
        self.base_threshold = float(value)
        print(f"主选点界面阈值已调整为: {self.base_threshold:.2f}")
        self._refresh_after_threshold_change()

    def _on_fine_threshold_changed(self, value):
        """已选点 / 剩余点界面精细阈值变化：实时重算并刷新视图"""
        self.fine_threshold = float(value)
        print(f"已选/剩余界面精细阈值已调整为: {self.fine_threshold:.2f}")
        self._refresh_after_threshold_change()

    def _refresh_after_threshold_change(self):
        """阈值改变后在保持视角的前提下重绘当前视图"""
        if not hasattr(self, "plotter"):
            return
        camera_pos = self._save_camera()
        self._visible_indices = self._get_view_indices()
        self._render_view(reset_view=False)
        self._restore_camera(camera_pos)
        self.plotter.render()

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
            self.fine_threshold if mode in ["selected", "remaining"] else self.base_threshold
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
        return self._enhance_display_colors(self.original_colors[indices].copy())

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
            point_size=4.0,
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
                point_size=4.0,
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
                    point_size=4.0,
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
                    self.fine_threshold if m in ["selected", "remaining"] else self.base_threshold
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

        # 基于颜色均值的清洁度计算
        # 用户选深色点（脏的，要剔除）→ 剩余的是浅色点（干净的）
        # 分母 = 分割区域总面积 × 选中深色点的颜色均值
        # 分子 = 剩余面积 × 剩余浅色点的颜色均值
        # 清洁度 = 分子 / 分母
        # 颜色：白=0, 黑=1（灰度 = 1 - luminance）
        if getattr(self, "_has_mesh_data", False):
            if (getattr(self, "_loaded_mesh", None) is not None
                    and self._loaded_mesh.has_vertex_colors()):
                mesh_rgb = np.asarray(self._loaded_mesh.vertex_colors)
            else:
                mesh_rgb = self.original_colors  # pcd 颜色（扫描后脏颜色）
            mesh_verts = self._mesh_verts
            mesh_faces = self._mesh_faces
            face_grays = self._face_grays
            face_areas = self._compute_face_areas(mesh_verts, mesh_faces)

            # 获取选中点（深色）和剩余点（浅色）的掩码
            picked_mask = self._get_selection_mask()  # 选中的是深色点
            keep_mask = ~picked_mask  # 剩余的是浅色点

            # 计算选中点和剩余点的颜色均值（基于点的颜色）
            all_point_grays = 1.0 - (
                mesh_rgb[:, 0] * 0.299 +
                mesh_rgb[:, 1] * 0.587 +
                mesh_rgb[:, 2] * 0.114
            )  # 白=0, 黑=1

            # 选中点的颜色均值
            picked_point_indices = np.where(picked_mask)[0]
            if len(picked_point_indices) > 0:
                picked_gray_mean = float(np.mean(all_point_grays[picked_point_indices]))
            else:
                picked_gray_mean = 0.0

            # 剩余点的颜色均值
            remaining_point_indices = np.where(keep_mask)[0]
            if len(remaining_point_indices) > 0:
                remaining_gray_mean = float(np.mean(all_point_grays[remaining_point_indices]))
            else:
                remaining_gray_mean = 0.0

            # 计算面级掩码
            keep_set = set(int(i) for i in keep_indices)
            remaining_mask = np.array([
                all(int(v) in keep_set for v in tri)
                for tri in mesh_faces
            ])
            remaining_areas = face_areas * remaining_mask

            total_area = float(np.sum(face_areas))
            remaining_area = float(np.sum(remaining_areas))

            if total_area > 0 and picked_gray_mean > 0:
                # 分母 = 分割区域总面积 × 选中深色点的颜色均值
                denominator = total_area * picked_gray_mean
                # 分子 = 剩余面积 × 剩余浅色点的颜色均值
                numerator = remaining_area * remaining_gray_mean

                cleanliness = numerator / denominator if denominator > 0 else 0.0
                dirty_area = float(denominator - numerator)
            else:
                numerator = 0.0
                denominator = 0.0
                cleanliness = -1.0
                dirty_area = 0.0
        else:
            # 无面级数据时：无法计算真实清洁度，设为 -1（表示无效/未知），
            # 避免兜底时用点级保留率（final_clean_rate = final/original*100）产生误导性的极高值。
            # 前端 _apply_score_style 按 score<60 红/<85 黄/>=85 绿判断，-1 统一显示红色。
            numerator   = 0.0
            denominator = 0.0
            cleanliness = -1.0  # 特殊标记：面级清洁度不可用
            dirty_area  = 0.0

        self.result_data = {
            "name": self.mesh_name,
            "original_points": int(self.total_points_count),
            "final_points": int(final_points_count),
            "original_area": float(getattr(self, "original_area", 0.0)),
            "remaining_area": float(remaining_area) if total_area > 0 else 0.0,
            "cleanliness": float(cleanliness * 100.0),
            "numerator": numerator,
            "denominator": denominator,
            "dirty_area": max(0.0, dirty_area),
            "picked_gray_mean": float(picked_gray_mean),
            "remaining_gray_mean": float(remaining_gray_mean),
        }
        has_mesh = getattr(self, "_has_mesh_data", False)
        if cleanliness < 0:
            print_str = (f"保存成功: {save_path}\n"
                         f"  [清洁度] 面级数据不可用（_has_mesh_data={has_mesh}），"
                         f"清洁度显示为 '--'")
        else:
            print_str = (
                f"保存成功: {save_path}\n"
                f"  [清洁度] 新计算逻辑: 分母=选中深色均值×总面积, 分子=剩余浅色均值×剩余面积\n"
                f"  分割总面积: {total_area:.4f} mm², 剩余面积: {remaining_area:.4f} mm²\n"
                f"  选中深色点颜色均值: {picked_gray_mean:.4f} (白=0, 黑=1)\n"
                f"  剩余浅色点颜色均值: {remaining_gray_mean:.4f} (白=0, 黑=1)\n"
                f"  分母: {denominator:.4f}, 分子: {numerator:.4f}\n"
                f"  清洁度: {cleanliness * 100:.2f}%\n"
                f"  剩余点数: {final_points_count}/{self.total_points_count}")
        print(print_str)
        self.close()

    def closeEvent(self, event):
        plotter = getattr(self, "plotter", None)
        if plotter is not None:
            plotter.close()
        event.accept()
        # 通知外部的局部事件循环退出（支持被其他 Qt 程序嵌套调用）
        loop = getattr(self, "_wait_loop", None)
        if loop is not None and loop.isRunning():
            loop.quit()


# --- 2. 批量处理逻辑接口 ---
def process_single_mesh_cleanliness_advanced(input_path, output_dir, color_threshold=12.0):
    app = QApplication.instance() or QApplication(sys.argv)
    window = AdvancedCleanerWindow(input_path, output_dir, color_threshold)
    # 加载失败（文件缺失 / 无顶点颜色）时直接跳过，避免空窗口阻塞流程
    if not hasattr(window, "points"):
        window.close()
        return None
    window.show()
    # 使用局部 QEventLoop 而不是 app.exec_()：
    # app.exec_() 在事件循环已运行时（例如从 CleanUI 调用）会立即返回，
    # 导致 result_data 为 None、JSON 报告无法生成。QEventLoop 可安全嵌套。
    loop = QEventLoop()
    window._wait_loop = loop
    loop.exec_()
    return window.result_data


# 模块级结果缓存：前端（CleanUI）在 JSON 文件读取失败时可直接读取 backend.all_stats
all_stats = []


def batch_clean_mapped_meshes(mesh_dir, cleaned_result_dir=None):
    """
    处理模型并将结果直接保存为 JSON 报告。
    """
    global all_stats
    all_stats = []
    if not os.path.exists(mesh_dir):
        return []

    if cleaned_result_dir is None:
        cleaned_result_dir = os.path.join(mesh_dir, "cleaned_results")
    os.makedirs(cleaned_result_dir, exist_ok=True)

    mesh_files = glob.glob(os.path.join(mesh_dir, "*.ply"))
    exclude_keywords = ["basedown"]  # 仅排除 basedown，其余分区全部参与清洁度运算

    for f_path in mesh_files:
        file_name = os.path.basename(f_path)
        if any(k in file_name.lower() for k in exclude_keywords):
            continue

        stats = process_single_mesh_cleanliness_advanced(f_path, cleaned_result_dir)
        if stats:
            all_stats.append(stats)

    # 保存 JSON
    if all_stats:
        json_path = os.path.join(cleaned_result_dir, "cleanliness_report.json")
        avg_clean = sum(s.get('cleanliness', 0) for s in all_stats) / len(all_stats)
        total_area = sum(s.get('denominator', 0) for s in all_stats if s.get('denominator', 0) > 0)
        if total_area > 0:
            weighted_clean = sum(
                s.get('numerator', 0) for s in all_stats
            ) / total_area * 100.0
        else:
            weighted_clean = avg_clean
        report_data = {
            "average_cleanliness": weighted_clean,
            "total_count": len(all_stats),
            "formula": "cleanliness = sum(face_area * grayscale) / sum(segment_face_area * grayscale)",
            "details": sorted(all_stats, key=lambda x: x.get('cleanliness', 0), reverse=True)
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
        print(f"√ 汇总报告已保存至: {json_path}")

    return all_stats


def batch_clean_segmented_pcds(output_base_dir, cleaned_result_dir=None):
    """
    对 mapped_meshes/Mesh_*.ply（register_and_segment2 映射回 Mesh 拓扑后的结果）做交互清理。
    这样所有后续选点都统一在 mapped_meshes 的点云上进行，不再使用 Seg_*.ply 分割结果。
    """
    global all_stats
    all_stats = []
    if not os.path.exists(output_base_dir):
        return []

    mesh_dir = os.path.join(output_base_dir, "mapped_meshes")
    if cleaned_result_dir is None:
        cleaned_result_dir = os.path.join(mesh_dir, "cleaned_results")
    os.makedirs(cleaned_result_dir, exist_ok=True)

    mesh_files = glob.glob(os.path.join(mesh_dir, "Mesh_*.ply"))
    exclude_keywords = ["basedown"]  # 仅排除 basedown，其余分区全部参与清洁度运算

    for f_path in mesh_files:
        file_name = os.path.basename(f_path)
        if any(k in file_name.lower() for k in exclude_keywords):
            continue

        stats = process_single_mesh_cleanliness_advanced(f_path, cleaned_result_dir)
        if stats:
            all_stats.append(stats)

    if all_stats:
        json_path = os.path.join(cleaned_result_dir, "cleanliness_report.json")
        avg_clean = sum(s.get('cleanliness', 0) for s in all_stats) / len(all_stats)
        total_area = sum(s.get('denominator', 0) for s in all_stats if s.get('denominator', 0) > 0)
        if total_area > 0:
            weighted_clean = sum(
                s.get('numerator', 0) for s in all_stats
            ) / total_area * 100.0
        else:
            weighted_clean = avg_clean
        report_data = {
            "average_cleanliness": weighted_clean,
            "total_count": len(all_stats),
            "formula": "cleanliness = sum(face_area * grayscale) / sum(segment_face_area * grayscale)",
            "details": sorted(all_stats, key=lambda x: x.get('cleanliness', 0), reverse=True)
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
        print(f"√ 汇总报告已保存至: {json_path}")

    return all_stats



def process_dental_mesh_registration(
        source_mesh_path,
        target_pcd_path,
        template_dir,
        output_base_dir,
        dist_threshold=0.8,
        smoothing_iters=2
):
    """
    牙模配准与自动化分割处理流程
    :param source_mesh_path: 待配准的原始 PLY Mesh 路径
    :param target_pcd_path: 目标位置的点云路径
    :param template_dir: 存放牙齿分割模板的文件夹
    :param output_base_dir: 结果输出根目录
    """

    # 1. 初始化路径与环境
    mesh_output_dir = os.path.join(output_base_dir, "mapped_meshes")
    os.makedirs(mesh_output_dir, exist_ok=True)

    # --- 读取数据 ---
    mesh1 = o3d.io.read_triangle_mesh(source_mesh_path)
    if not mesh1.has_triangles():
        print(f"错误：{source_mesh_path} 不包含有效的 Mesh 面片信息！")
        return

    # 从 Mesh 提取点云用于配准
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = mesh1.vertices
    pcd1.colors = mesh1.vertex_colors
    pcd1.normals = mesh1.vertex_normals

    pcd2 = o3d.io.read_point_cloud(target_pcd_path)
    if pcd2.is_empty():
        print(f"错误：无法读取目标点云 {target_pcd_path}")
        return

    # --- 初始对齐与交互选点 ---
    center1 = pcd1.get_center()
    center2 = pcd2.get_center()
    shift_to_align = center1 - center2
    # 将 pcd2 平移并平移 100 单位以便侧向对比观察
    pcd2_shifted = copy.deepcopy(pcd2).translate(shift_to_align).translate([100, 0, 0])

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="选点：左侧Source(1)，右侧Target(2)")
    vis.add_geometry(pcd1 + pcd2_shifted)
    vis.run()
    vis.destroy_window()

    picked_ids = vis.get_picked_points()
    if len(picked_ids) < 2:
        print("配准失败：选点数量不足（至少需要一对点）。")
        return

    # --- 精确配准计算 ---
    N = len(picked_ids) // 2
    ids1, ids2 = picked_ids[:N], picked_ids[N:]
    all_points = np.asarray((pcd1 + pcd2_shifted).points)

    # 还原 Target 选点的真实坐标（减去之前的展示平移量）
    source_points = all_points[ids1]
    target_points = all_points[ids2] - np.array([100, 0, 0]) - shift_to_align

    src_corr = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
    tgt_corr = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_points))

    corr = np.zeros((N, 2), dtype=int)
    corr[:, 0], corr[:, 1] = np.arange(N), np.arange(N)

    # 初始变换矩阵计算
    p2p_est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p_est.compute_transformation(src_corr, tgt_corr, o3d.utility.Vector2iVector(corr))

    # ICP 优化：粗对齐 -> 精对齐
    reg_p2p = o3d.pipelines.registration.registration_icp(pcd1, pcd2, 3.0, trans_init, p2p_est)
    reg_refine = o3d.pipelines.registration.registration_icp(
        pcd1.transform(reg_p2p.transformation), pcd2, 1.0, np.eye(4), p2p_est
    )

    # 计算总矩阵并作用于原始 Mesh
    total_transform = reg_refine.transformation @ reg_p2p.transformation
    mesh1.transform(total_transform)

    # 同步更新点云以进行后续分割
    pcd1_final = o3d.geometry.PointCloud()
    pcd1_final.points = mesh1.vertices
    pcd1_final.colors = mesh1.vertex_colors

    # --- 批量分割逻辑 ---
    template_files = glob.glob(os.path.join(template_dir, "*.ply"))
    all_tooth_indices = []

    for t_path in template_files:
        file_name = os.path.basename(t_path)
        pcd_out_path = os.path.join(output_base_dir, f"Seg_{file_name}")

        # 调用外部定义的分割函数 register_and_segment2
        segmented_pcd, indices = register_and_segment2(
            source=pcd1_final,
            template_path=t_path,
            output_path=pcd_out_path,
            distance_threshold=dist_threshold
        )

        if segmented_pcd is not None:
            # 映射回 Mesh 拓扑
            mapped_mesh_path = os.path.join(mesh_output_dir, f"Mesh_{file_name}")
            map_mesh_and_color_to_pcd(
                mesh_path=t_path,
                pcd_obj=segmented_pcd,
                output_path=mapped_mesh_path,
                smoothing_iterations=smoothing_iters
            )

            # 汇总非牙龈部分的索引
            if "牙龈" not in file_name:
                all_tooth_indices.extend(indices)

    # --- 合并牙齿 Mesh 部分 ---
    if all_tooth_indices:
        unique_indices = list(set(all_tooth_indices))
        combined_tooth_mesh = mesh1.select_by_index(unique_indices)

        combined_output_path = os.path.join(output_base_dir, "toothpart.ply")
        o3d.io.write_triangle_mesh(combined_output_path, combined_tooth_mesh)
        print(f"\n任务完成！合并后的牙齿已保存至: {combined_output_path}")

        # 4. 后处理阶段
        # 交互式手动调整
        # interactive_segmentation(combined_output_path)
        # 交互清理使用 mapped_meshes/Mesh_*.ply（映射回 Mesh 拓扑后的点云）
        batch_clean_mapped_meshes(mesh_output_dir)

        # 可视化结果
        o3d.visualization.draw_geometries([combined_tooth_mesh], window_name="Final Tooth Mesh")
    else:
        print("警告：未发现任何牙齿分割索引。")




# --- 调用示例 ---
if __name__ == "__main__":
    process_dental_mesh_registration(
        source_mesh_path=r"D:\UClean\IO9-3 LowerJawScan.ply",
        target_pcd_path=r"D:\UClean\LowerJawScans.ply",
        template_dir=r"D:\UClean\segmentationsply",
        output_base_dir=r"D:\UClean\segementations"
    )





