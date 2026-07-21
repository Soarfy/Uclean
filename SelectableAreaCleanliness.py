# -*- coding: utf-8 -*-
"""Selectable standard/detailed dental-region cleanliness backend.

This module reuses the registration, colour conversion, exposure
normalisation and true-3D triangle integration implementation from
``GeneratePathOffset66initAllnewdorobotsnewAreaWeighted.py``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import GeneratePathOffset66initAllnewdorobotsnewAreaWeighted as core


SEGMENTATION_MODES = ("standard", "detailed")
REGION_GROUP_ORDER = ("insideface", "outsideface", "upface", "tips", "baseface")
REGION_GROUP_NAMES_ZH = {
    "insideface": "内侧面", "outsideface": "外侧面",
    "upface": "上槽牙（咬合面）", "tips": "牙缝",
    "baseface": "牙根区域（龈沟）",
}


@dataclass(frozen=True)
class SegmentationRegion:
    """One selectable segmentation module."""

    name: str
    group: str
    path: Path


def _geometry_points(path: Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    points = np.asarray(mesh.vertices)
    if not len(points):
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    if not len(points):
        raise ValueError(f"分割文件没有有效几何点：{path}")
    return points.copy()


def discover_regions(folder: str | Path, mode: str) -> List[SegmentationRegion]:
    """Discover large regions or numbered detailed regions.

    ``standard`` uses only ``group/group.ply``. ``detailed`` uses only the
    numbered files such as ``group/group.001.ply``.
    """
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"分割目录不存在：{root}")
    if mode not in SEGMENTATION_MODES:
        raise ValueError(f"分割模式必须是：{', '.join(SEGMENTATION_MODES)}")

    regions: List[SegmentationRegion] = []
    group_dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.casefold())
    for group_dir in group_dirs:
        standard_path = group_dir / f"{group_dir.name}.ply"
        if mode == "standard":
            candidates = [standard_path] if standard_path.is_file() else []
        else:
            candidates = [
                path for path in sorted(group_dir.glob("*.ply"))
                if path.name.casefold() != standard_path.name.casefold()
            ]
        for path in candidates:
            try:
                _geometry_points(path)
            except ValueError:
                continue
            regions.append(SegmentationRegion(path.stem, group_dir.name, path))
    if not regions:
        mode_name = "标准大区域" if mode == "standard" else "细分区域"
        raise FileNotFoundError(f"{root} 中没有可用的{mode_name} PLY 文件")
    return regions


def build_region_trees(regions: Sequence[SegmentationRegion]) -> List[cKDTree]:
    return [cKDTree(_geometry_points(region.path)) for region in regions]


def preview_geometries(
    regions: Sequence[SegmentationRegion],
) -> List[o3d.geometry.PointCloud]:
    """Combine all modules as individually coloured point clouds."""
    palette = np.array([
        [0.90, 0.12, 0.15], [0.16, 0.48, 0.85], [0.18, 0.70, 0.35],
        [0.62, 0.28, 0.78], [1.00, 0.50, 0.05], [0.05, 0.70, 0.70],
        [0.95, 0.75, 0.08], [0.85, 0.30, 0.58], [0.45, 0.52, 0.60],
    ])
    clouds: List[o3d.geometry.PointCloud] = []
    for index, region in enumerate(regions):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(_geometry_points(region.path)))
        cloud.paint_uniform_color(palette[index % len(palette)])
        clouds.append(cloud)
    return clouds


def show_region_preview(regions: Sequence[SegmentationRegion], title: str) -> None:
    if not regions:
        raise ValueError("没有可预览的分割区域")
    o3d.visualization.draw_geometries(
        preview_geometries(regions), window_name=title, width=1280, height=800
    )


def _label_faces(
    mesh: o3d.geometry.TriangleMesh,
    trees: Sequence[cKDTree],
    max_distance: float,
) -> np.ndarray:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    centroids = vertices[faces].mean(axis=1)
    distances = np.column_stack([
        tree.query(centroids, k=1, workers=-1)[0] for tree in trees
    ])
    labels = np.argmin(distances, axis=1).astype(np.int32)
    labels[np.min(distances, axis=1) > max_distance] = -1
    return labels


def calculate_selected_cleanliness(
    standard_model_path: str | Path,
    unclean_model_path: str | Path,
    cleaned_model_path: str | Path,
    segmentation_folder: str | Path,
    segmentation_mode: str,
    selected_region_names: Sequence[str],
    output_dir: str | Path,
    voxel_size: Optional[float] = None,
    max_label_distance: Optional[float] = None,
    show_matching_results: bool = True,
    show_point_cloud_comparison: bool = False,
    save_intermediate_results: bool = True,
) -> dict:
    """Calculate cleanliness for exactly the selected segmentation modules."""
    all_regions = discover_regions(segmentation_folder, segmentation_mode)
    selected_keys = {name.casefold() for name in selected_region_names}
    selected_ids = [i for i, region in enumerate(all_regions) if region.name.casefold() in selected_keys]
    missing = selected_keys - {region.name.casefold() for region in all_regions}
    if missing:
        raise ValueError(f"所选区域不存在：{', '.join(sorted(missing))}")
    if not selected_ids:
        raise ValueError("请至少选择一个参与计算的区域")

    out = Path(output_dir)
    crop_dir = out / "selected_cropped_meshes"
    gray_dir = out / "selected_gray3d"
    out.mkdir(parents=True, exist_ok=True)
    if save_intermediate_results:
        crop_dir.mkdir(parents=True, exist_ok=True)
        gray_dir.mkdir(parents=True, exist_ok=True)

    standard = core._load_mesh(standard_model_path)
    before = core._load_mesh(unclean_model_path)
    after = core._load_mesh(cleaned_model_path)
    print("[1/4] 自动配准刷前模型和刷后模型 ...")
    before_reg = core.register_mesh_to_standard(before, standard, voxel_size)
    after_reg = core.register_mesh_to_standard(after, standard, voxel_size)
    after_reg = core.refine_cleaned_to_unclean(after_reg, before_reg, standard, voxel_size)
    if show_matching_results:
        core.show_registration(standard, before_reg.mesh, "刷前模型 -> 标准模型")
        core.show_registration(standard, after_reg.mesh, "刷后模型 -> 标准模型")

    print("[2/4] 映射完整分割方案并提取所选区域 ...")
    # Label against the complete scheme first, so an unselected neighbouring
    # module cannot be mistakenly absorbed by a selected module.
    trees = build_region_trees(all_regions)
    diagonal = float(np.linalg.norm(np.ptp(np.asarray(standard.vertices), axis=0)))
    label_distance = float(max_label_distance or max(diagonal / 80.0, 0.20))
    before_labels = _label_faces(before_reg.mesh, trees, label_distance)
    after_labels = _label_faces(after_reg.mesh, trees, label_distance)
    before_selected_faces = np.flatnonzero(np.isin(before_labels, selected_ids))
    after_selected_faces = np.flatnonzero(np.isin(after_labels, selected_ids))
    if not len(before_selected_faces) or not len(after_selected_faces):
        raise RuntimeError("所选区域未映射到扫描模型，请检查模型配准和分割坐标")

    before_raw = core.rgb_to_luminance(np.asarray(before_reg.mesh.vertex_colors))
    after_raw = core.rgb_to_luminance(np.asarray(after_reg.mesh.vertex_colors))
    before_vertices = np.unique(np.asarray(before_reg.mesh.triangles)[before_selected_faces])
    after_vertices = np.unique(np.asarray(after_reg.mesh.triangles)[after_selected_faces])
    before_lum, after_lum, exposure = core.normalise_luminance_pair(
        before_raw, after_raw, before_raw[before_vertices], after_raw[after_vertices]
    )

    print("[3/4] 计算所选区域的真实三维面积颜色积分 ...")
    details = []
    valid_before_parts: List[np.ndarray] = []
    valid_after_parts: List[np.ndarray] = []
    group_before_parts = {}
    group_after_parts = {}
    for region_id in selected_ids:
        region = all_regions[region_id]
        bf = np.flatnonzero(before_labels == region_id)
        af = np.flatnonzero(after_labels == region_id)
        if not len(bf) or not len(af):
            print(f"      跳过未同时匹配区域：{region.name}")
            continue
        b3d = core.integrate_region_3d(before_reg.mesh, bf, before_lum)
        a3d = core.integrate_region_3d(after_reg.mesh, af, after_lum)
        ratio = a3d["darkness_integral_3d"] / max(b3d["darkness_integral_3d"], core.EPS)
        score = float(np.clip((1.0 - ratio) * 100.0, -100.0, 100.0))
        details.append({
            "name": region.name,
            "group": region.group,
            "segmentation_file": str(region.path),
            "cleanliness": score,
            "integral_ratio_after_before": float(ratio),
            "before_surface_area": b3d["surface_area"],
            "after_surface_area": a3d["surface_area"],
            "before_faces": int(len(bf)),
            "after_faces": int(len(af)),
            "unclean_3d": b3d,
            "cleaned_3d": a3d,
        })
        valid_before_parts.append(bf)
        valid_after_parts.append(af)
        group_before_parts.setdefault(region.group.casefold(), []).append(bf)
        group_after_parts.setdefault(region.group.casefold(), []).append(af)
        if save_intermediate_results:
            o3d.io.write_triangle_mesh(
                str(crop_dir / f"{region.name}_unclean.ply"),
                core.extract_face_mesh(before_reg.mesh, bf),
            )
            o3d.io.write_triangle_mesh(
                str(crop_dir / f"{region.name}_cleaned.ply"),
                core.extract_face_mesh(after_reg.mesh, af),
            )
        if show_point_cloud_comparison:
            core.show_region_gray_comparison(
                before_reg.mesh, bf, before_lum,
                after_reg.mesh, af, after_lum,
                region.name, gray_dir, save_files=save_intermediate_results,
            )
    if not details:
        raise RuntimeError("没有同时匹配到刷前和刷后模型的所选区域")

    # Combine selected detailed modules by their owning large region.  The
    # score is calculated from the union's before/after integrals, never by
    # averaging the percentages of individual small modules.
    group_details = []
    for group in REGION_GROUP_ORDER:
        before_parts = group_before_parts.get(group, [])
        after_parts = group_after_parts.get(group, [])
        if not before_parts or not after_parts:
            continue
        group_bf = np.unique(np.concatenate(before_parts))
        group_af = np.unique(np.concatenate(after_parts))
        group_before = core.integrate_region_3d(before_reg.mesh, group_bf, before_lum)
        group_after = core.integrate_region_3d(after_reg.mesh, group_af, after_lum)
        ratio = group_after["darkness_integral_3d"] / max(
            group_before["darkness_integral_3d"], core.EPS
        )
        group_details.append({
            "group": group,
            "name_zh": REGION_GROUP_NAMES_ZH[group],
            "selected_modules": [
                item["name"] for item in details
                if item["group"].casefold() == group
            ],
            "cleanliness": float(np.clip((1.0 - ratio) * 100.0, -100.0, 100.0)),
            "integral_ratio_after_before": float(ratio),
            "before_surface_area": group_before["surface_area"],
            "after_surface_area": group_after["surface_area"],
            "before_faces": int(len(group_bf)),
            "after_faces": int(len(group_af)),
            "unclean_3d": group_before,
            "cleaned_3d": group_after,
        })

    # Integrate the selected union directly, preserving the reference method.
    overall_bf = np.unique(np.concatenate(valid_before_parts))
    overall_af = np.unique(np.concatenate(valid_after_parts))
    overall_before = core.integrate_region_3d(before_reg.mesh, overall_bf, before_lum)
    overall_after = core.integrate_region_3d(after_reg.mesh, overall_af, after_lum)
    total_before = overall_before["darkness_integral_3d"]
    total_after = overall_after["darkness_integral_3d"]
    overall_score = float(np.clip(
        (1.0 - total_after / max(total_before, core.EPS)) * 100.0, -100.0, 100.0
    ))
    report = {
        "method": "selected segmentation modules + true-3D triangle-area weighted colour integral",
        "segmentation_mode": segmentation_mode,
        "segmentation_folder": str(Path(segmentation_folder)),
        "selected_regions": [all_regions[i].name for i in selected_ids],
        "formula": "cleanliness(%) = (1 - cleaned_selected_darkness_integral / unclean_selected_darkness_integral) * 100",
        "exposure_normalisation": exposure,
        "max_label_distance": label_distance,
        "registration": {
            "unclean": {"fitness": before_reg.fitness, "inlier_rmse": before_reg.inlier_rmse},
            "cleaned": {"fitness": after_reg.fitness, "inlier_rmse": after_reg.inlier_rmse},
        },
        "overall_cleanliness": overall_score,
        "total_unclean_darkness_integral": total_before,
        "total_cleaned_darkness_integral": total_after,
        "total_unclean_surface_area": overall_before["surface_area"],
        "total_cleaned_surface_area": overall_after["surface_area"],
        "overall_unclean_3d": overall_before,
        "overall_cleaned_3d": overall_after,
        "total_count": len(details),
        "selected_group_count": len(group_details),
        "group_details": group_details,
        "details": details,
    }
    print("[4/4] 保存清洁度报告 ...")
    with (out / "selected_cleanliness_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    with (out / "selected_cleanliness_report.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "region", "group", "cleanliness_percent", "unclean_darkness_integral",
            "cleaned_darkness_integral", "unclean_area_mm2", "cleaned_area_mm2",
        ])
        for item in details:
            writer.writerow([
                item["name"], item["group"], item["cleanliness"],
                item["unclean_3d"]["darkness_integral_3d"],
                item["cleaned_3d"]["darkness_integral_3d"],
                item["before_surface_area"], item["after_surface_area"],
            ])
    with (out / "selected_cleanliness_by_group.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "group", "group_zh", "selected_modules", "cleanliness_percent",
            "unclean_darkness_integral", "cleaned_darkness_integral",
            "unclean_area_mm2", "cleaned_area_mm2",
        ])
        for item in group_details:
            writer.writerow([
                item["group"], item["name_zh"], ";".join(item["selected_modules"]),
                item["cleanliness"], item["unclean_3d"]["darkness_integral_3d"],
                item["cleaned_3d"]["darkness_integral_3d"],
                item["before_surface_area"], item["after_surface_area"],
            ])
    return report
