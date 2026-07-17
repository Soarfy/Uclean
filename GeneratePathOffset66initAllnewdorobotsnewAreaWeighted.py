# -*- coding: utf-8 -*-
"""Dental cleanliness measured by true 3-D triangle-area weighted greyscale.

Pipeline
--------
1. Register the unclean and cleaned meshes independently to a standard mesh.
2. Transfer region labels from ``segmentationsply`` to both registered meshes.
3. Ignore basedown, convert vertex colour to grey and integrate every triangle
   as ``true 3-D area * mean triangle darkness``.
4. UV images are generated only for visual inspection and never affect scores.

The module is intentionally non-interactive.  It can be called from Python or
run directly.  Open3D, NumPy, SciPy and OpenCV are required.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


EPS = 1.0e-12
# In this project only basedown is excluded.  baseface and every other mesh
# region must be measured.
DEFAULT_GINGIVA_WORDS = ("basedown",)

# Kept for compatibility with CleanUI.py, which reads this module variable.
all_stats: List[dict] = []


@dataclass
class RegistrationResult:
    mesh: o3d.geometry.TriangleMesh
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float


def _load_mesh(path: os.PathLike | str) -> o3d.geometry.TriangleMesh:
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=False)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"PLY is not a triangle mesh or is empty: {path}")
    if not mesh.has_vertex_colors():
        raise ValueError(f"PLY has no vertex RGB colours: {path}")
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def _pcd_for_registration(mesh: o3d.geometry.TriangleMesh, voxel: float) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices).copy())
    pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3.0, max_nn=50))
    return pcd


def _pca_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centre = np.median(points, axis=0)
    _, _, vh = np.linalg.svd(points - centre, full_matrices=False)
    frame = vh.T
    if np.linalg.det(frame) < 0:
        frame[:, -1] *= -1
    return centre, frame


def _pca_initialisations(source: np.ndarray, target: np.ndarray) -> Iterable[np.ndarray]:
    """Yield rigid PCA initialisations, including axis/sign ambiguity."""
    cs, fs = _pca_frame(source)
    ct, ft = _pca_frame(target)
    for perm in itertools.permutations(range(3)):
        p = np.eye(3)[:, perm]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            q = p @ np.diag(signs)
            r = ft @ q @ fs.T
            if np.linalg.det(r) < 0:
                continue
            t = ct - r @ cs
            transform = np.eye(4)
            transform[:3, :3] = r
            transform[:3, 3] = t
            yield transform


def register_mesh_to_standard(
    moving: o3d.geometry.TriangleMesh,
    standard: o3d.geometry.TriangleMesh,
    voxel_size: Optional[float] = None,
) -> RegistrationResult:
    """Rigid PCA + multi-scale point-to-plane ICP; returns moving -> standard."""
    sv = np.asarray(standard.vertices)
    mv = np.asarray(moving.vertices)
    diagonal = float(np.linalg.norm(np.ptp(sv, axis=0)))
    voxel = float(voxel_size or max(diagonal / 120.0, 0.05))

    # Coarse clouds keep the initialisation search fast.
    coarse_voxel = voxel * 4.0
    src_c = _pcd_for_registration(moving, coarse_voxel)
    dst_c = _pcd_for_registration(standard, coarse_voxel)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=35)

    best = None
    for init in _pca_initialisations(np.asarray(src_c.points), np.asarray(dst_c.points)):
        reg = o3d.pipelines.registration.registration_icp(
            src_c, dst_c, coarse_voxel * 3.0, init, estimation, criteria
        )
        score = (float(reg.fitness), -float(reg.inlier_rmse))
        if best is None or score > best[0]:
            best = (score, reg.transformation)
    if best is None:
        raise RuntimeError("Could not create an ICP initialisation")

    transform = best[1]
    final_reg = None
    for scale, iterations in ((2.0, 60), (1.0, 80), (0.5, 100)):
        level_voxel = voxel * scale
        src = _pcd_for_registration(moving, level_voxel)
        dst = _pcd_for_registration(standard, level_voxel)
        final_reg = o3d.pipelines.registration.registration_icp(
            src,
            dst,
            level_voxel * 2.5,
            transform,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations),
        )
        transform = final_reg.transformation
        print(
            f"      ICP scale={scale:g}, fitness={final_reg.fitness:.5f}, "
            f"RMSE={final_reg.inlier_rmse:.5f}"
        )

    aligned = o3d.geometry.TriangleMesh(moving)
    aligned.transform(transform)
    aligned.compute_vertex_normals()
    return RegistrationResult(
        aligned, transform, float(final_reg.fitness), float(final_reg.inlier_rmse)
    )


def _is_gingiva(name: str, words: Sequence[str]) -> bool:
    lowered = name.casefold()
    return any(word.casefold() in lowered for word in words)


def build_standard_region_labels(
    standard: o3d.geometry.TriangleMesh,
    template_dir: os.PathLike | str,
    gingiva_words: Sequence[str] = DEFAULT_GINGIVA_WORDS,
) -> Tuple[np.ndarray, List[str], List[bool]]:
    """Label each standard vertex by its nearest segmentation template."""
    paths = sorted(Path(template_dir).glob("*.ply"))
    if not paths:
        raise FileNotFoundError(f"No PLY templates in {template_dir}")
    names, trees, gingiva = [], [], []
    for path in paths:
        template = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
        points = np.asarray(template.vertices)
        if len(points) == 0:
            pcd = o3d.io.read_point_cloud(str(path))
            points = np.asarray(pcd.points)
        if len(points) == 0:
            continue
        names.append(path.stem)
        trees.append(cKDTree(points))
        gingiva.append(_is_gingiva(path.stem, gingiva_words))
    if not trees:
        raise ValueError("All segmentation templates are empty")

    vertices = np.asarray(standard.vertices)
    distance_columns = [tree.query(vertices, k=1, workers=-1)[0] for tree in trees]
    labels = np.argmin(np.column_stack(distance_columns), axis=1).astype(np.int32)
    return labels, names, gingiva


def load_template_region_references(
    template_dir: os.PathLike | str,
    gingiva_words: Sequence[str] = DEFAULT_GINGIVA_WORDS,
) -> Tuple[List[str], List[cKDTree], List[bool]]:
    """Load each segmentation mesh as a direct geometric region reference.

    Unlike ``build_standard_region_labels``, this does not first paint labels
    onto the vertices of the complete standard mesh.  Consequently a region
    boundary is governed by the actual meshes in ``template_dir`` rather than
    by the vertex density/topology of the standard mesh.
    """
    paths = sorted(Path(template_dir).glob("*.ply"))
    if not paths:
        raise FileNotFoundError(f"No PLY templates in {template_dir}")

    names: List[str] = []
    trees: List[cKDTree] = []
    gingiva: List[bool] = []
    for path in paths:
        template = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
        points = np.asarray(template.vertices)
        if len(points) == 0:
            points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
        if len(points) == 0:
            print(f"      skip empty template: {path.name}")
            continue
        names.append(path.stem)
        trees.append(cKDTree(points))
        gingiva.append(_is_gingiva(path.stem, gingiva_words))
    if not trees:
        raise ValueError("All segmentation templates are empty")
    return names, trees, gingiva


def label_mesh_faces_from_templates(
    mesh: o3d.geometry.TriangleMesh,
    template_trees: Sequence[cKDTree],
    max_distance: float,
) -> np.ndarray:
    """Crop/label aligned scan faces directly against segmentation meshes."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    centroids = vertices[faces].mean(axis=1)
    distance_columns = [tree.query(centroids, k=1, workers=-1)[0] for tree in template_trees]
    distances = np.column_stack(distance_columns)
    labels = np.argmin(distances, axis=1).astype(np.int32)
    labels[np.min(distances, axis=1) > max_distance] = -1
    return labels


def extract_face_mesh(
    mesh: o3d.geometry.TriangleMesh,
    face_indices: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """Create a compact triangle mesh from selected faces, preserving RGB."""
    source_vertices = np.asarray(mesh.vertices)
    selected_faces = np.asarray(mesh.triangles)[face_indices]
    used_vertices, inverse = np.unique(selected_faces.reshape(-1), return_inverse=True)
    cropped = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(source_vertices[used_vertices].copy()),
        o3d.utility.Vector3iVector(inverse.reshape(-1, 3).astype(np.int32)),
    )
    if mesh.has_vertex_colors():
        cropped.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertex_colors)[used_vertices].copy()
        )
    cropped.compute_vertex_normals()
    return cropped


def label_mesh_faces(
    mesh: o3d.geometry.TriangleMesh,
    standard_vertices: np.ndarray,
    standard_vertex_labels: np.ndarray,
    max_distance: float,
) -> np.ndarray:
    """Transfer standard labels to aligned mesh faces through face centroids."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    centroids = vertices[faces].mean(axis=1)
    distances, nearest = cKDTree(standard_vertices).query(centroids, workers=-1)
    labels = standard_vertex_labels[nearest].copy()
    labels[distances > max_distance] = -1
    return labels


def rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    """sRGB -> linear-light relative luminance (IEC 61966-2-1)."""
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear @ np.array([0.2126, 0.7152, 0.0722])


def normalise_luminance_pair(
    before: np.ndarray,
    after: np.ndarray,
    before_sample: Optional[np.ndarray] = None,
    after_sample: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Shared percentile normalisation reduces scanner exposure differences."""
    # Only tooth samples should determine the exposure range; gingiva colours
    # must not influence either the UV integral or its normalisation.
    joined = np.concatenate([
        before if before_sample is None else before_sample,
        after if after_sample is None else after_sample,
    ])
    low, high = np.percentile(joined, [2.0, 98.0])
    if high - low < EPS:
        low, high = float(joined.min()), float(joined.max() + EPS)
    return (
        np.clip((before - low) / (high - low), 0.0, 1.0),
        np.clip((after - low) / (high - low), 0.0, 1.0),
        {"luminance_p02": float(low), "luminance_p98": float(high)},
    )


def _pca_uv(vertices: np.ndarray) -> np.ndarray:
    """Stable 2-D UV projection for one anatomical region."""
    centre, frame = _pca_frame(vertices)
    uv = (vertices - centre) @ frame[:, :2]
    # Resolve sign deterministically in standard coordinates.
    for axis in range(2):
        dominant_xyz = int(np.argmax(np.abs(frame[:, axis])))
        if frame[dominant_xyz, axis] < 0:
            uv[:, axis] *= -1
    return uv


def _cylindrical_arch_uv(vertices: np.ndarray) -> np.ndarray:
    """Unroll a curved inside/outside dental-arch wall to (arc, height).

    PCA first finds the dental-arch plane.  A least-squares circle is fitted in
    that plane; polar angle becomes horizontal U and height outside that plane
    becomes V.  The seam is placed in the largest angular gap, avoiding a cut
    through the measured tooth surface whenever possible.
    """
    centre, frame = _pca_frame(vertices)
    local = (vertices - centre) @ frame
    x, y, height = local[:, 0], local[:, 1], local[:, 2]

    # Algebraic least-squares circle: x^2+y^2 = 2*cx*x+2*cy*y+c.
    design = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(x))))
    rhs = x * x + y * y
    cx, cy, _ = np.linalg.lstsq(design, rhs, rcond=None)[0]
    dx, dy = x - cx, y - cy
    radius = float(np.median(np.hypot(dx, dy)))
    if radius < EPS:
        return _pca_uv(vertices)

    angle = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
    ordered = np.sort(angle)
    wrapped = np.r_[ordered, ordered[0] + 2.0 * np.pi]
    seam_index = int(np.argmax(np.diff(wrapped)))
    seam = float(wrapped[seam_index + 1] % (2.0 * np.pi))
    unwrapped_angle = np.mod(angle - seam, 2.0 * np.pi)

    # Remove a possible unused angular interval at either end and express U in
    # approximate millimetres of arc length.
    unwrapped_angle -= unwrapped_angle.min()
    return np.column_stack((unwrapped_angle * radius, height))


def unwrap_region_uv(vertices: np.ndarray, region_name: str) -> Tuple[np.ndarray, str]:
    """Choose an anatomical UV parameterisation for the named region."""
    name = region_name.casefold()
    curved_words = ("inside", "outside", "baseface", "内侧", "外侧")
    if any(word in name for word in curved_words):
        return _cylindrical_arch_uv(vertices), "dental_arch_cylindrical"
    return _pca_uv(vertices), "pca_planar"


def rasterise_region_uv(
    mesh: o3d.geometry.TriangleMesh,
    face_indices: np.ndarray,
    luminance: np.ndarray,
    resolution: int,
    region_name: str,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Rasterise triangle greys to UV pixels and return image, mask, metrics.

    Triangle colour is the mean of its three vertex greys.  Accumulation and a
    coverage counter make overlapping projected triangles deterministic.
    """
    vertices = np.asarray(mesh.vertices)
    faces_all = np.asarray(mesh.triangles)
    faces = faces_all[face_indices]
    used = np.unique(faces)
    uv_all, unwrap_method = unwrap_region_uv(vertices[used], region_name)
    local = np.full(len(vertices), -1, dtype=np.int64)
    local[used] = np.arange(len(used))
    uv = uv_all[local[faces]]

    uv_min = uv_all.min(axis=0)
    uv_max = uv_all.max(axis=0)
    extent = np.maximum(uv_max - uv_min, EPS)
    margin = 3
    scale = (resolution - 2 * margin - 1) / float(extent.max())
    pix = (uv - uv_min) * scale + margin

    accum = np.zeros((resolution, resolution), np.float64)
    count = np.zeros((resolution, resolution), np.uint16)
    face_lum = luminance[faces].mean(axis=1)
    for tri, grey in zip(pix, face_lum):
        points = np.rint(tri).astype(np.int32)
        x0 = max(int(points[:, 0].min()), 0)
        x1 = min(int(points[:, 0].max()), resolution - 1)
        y0 = max(int(points[:, 1].min()), 0)
        y1 = min(int(points[:, 1].max()), resolution - 1)
        if x1 < x0 or y1 < y0:
            continue
        local_mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        local_points = points - np.array([x0, y0], dtype=np.int32)
        cv2.fillConvexPoly(local_mask, local_points, 1, lineType=cv2.LINE_8)
        selected = local_mask.astype(bool)
        accum_roi = accum[y0:y1 + 1, x0:x1 + 1]
        count_roi = count[y0:y1 + 1, x0:x1 + 1]
        accum_roi[selected] += grey
        count_roi[selected] += 1

    mask = count > 0
    image = np.full_like(accum, np.nan)
    image[mask] = accum[mask] / count[mask]
    pixel_area = 1.0 / (scale * scale)
    grey_integral = float(np.nansum(image) * pixel_area)
    darkness_integral = float(np.nansum(1.0 - image[mask]) * pixel_area)
    return image, mask, {
        "uv_valid_pixels": int(mask.sum()),
        "uv_pixel_area": float(pixel_area),
        "gray_integral": grey_integral,
        "darkness_integral": darkness_integral,
        "mean_gray": float(np.nanmean(image)),
        "mean_darkness": float(np.nanmean(1.0 - image[mask])),
        "unwrap_method": unwrap_method,
    }


def _save_uv_png(path: Path, image: np.ndarray, mask: np.ndarray) -> None:
    output = np.zeros((*image.shape, 4), dtype=np.uint8)
    grey = np.zeros(image.shape, dtype=np.uint8)
    grey[mask] = np.rint(np.clip(image[mask], 0, 1) * 255).astype(np.uint8)
    output[..., :3] = grey[..., None]
    output[..., 3] = mask.astype(np.uint8) * 255
    cv2.imwrite(str(path), output)


def _mesh_as_display_cloud(
    mesh: o3d.geometry.TriangleMesh,
    voxel: float,
) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices).copy())
    if mesh.has_vertex_colors():
        cloud.colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors).copy())
    cloud = cloud.voxel_down_sample(max(voxel, EPS))
    return cloud


def show_registration(
    standard: o3d.geometry.TriangleMesh,
    aligned: o3d.geometry.TriangleMesh,
    title: str,
) -> None:
    """Show automatic ICP overlap. This window never asks the user to pick points."""
    diagonal = float(np.linalg.norm(np.ptp(np.asarray(standard.vertices), axis=0)))
    voxel = max(diagonal / 250.0, 0.03)
    standard_cloud = _mesh_as_display_cloud(standard, voxel)
    aligned_cloud = _mesh_as_display_cloud(aligned, voxel)
    print("      标准模型和扫描模型均显示各自原始RGB颜色；关闭窗口继续。")
    o3d.visualization.draw_geometries(
        [standard_cloud, aligned_cloud],
        window_name=title,
        width=1280,
        height=800,
    )


def refine_cleaned_to_unclean(
    cleaned_registration: RegistrationResult,
    unclean_registration: RegistrationResult,
    standard: o3d.geometry.TriangleMesh,
    voxel_size: Optional[float] = None,
) -> RegistrationResult:
    """Small pairwise ICP correction after both scans entered standard space.

    The cleaned scan is the moving cloud and the unclean scan is fixed.  The
    correspondence threshold is deliberately tighter than global ICP so this
    stage cannot make a large jump away from the standard coordinate system.
    """
    diagonal = float(np.linalg.norm(np.ptp(np.asarray(standard.vertices), axis=0)))
    voxel = float(voxel_size or max(diagonal / 160.0, 0.04))
    moving = _pcd_for_registration(cleaned_registration.mesh, voxel)
    fixed = _pcd_for_registration(unclean_registration.mesh, voxel)
    result = o3d.pipelines.registration.registration_icp(
        moving,
        fixed,
        voxel * 1.25,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
    )
    correction = result.transformation
    translation = float(np.linalg.norm(correction[:3, 3]))
    rotation_cos = np.clip((np.trace(correction[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    rotation_deg = float(np.degrees(np.arccos(rotation_cos)))
    # Reject an implausible pairwise correction rather than silently damaging
    # the already valid standard registrations.
    if translation > voxel * 2.0 or rotation_deg > 5.0:
        print(
            f"      Pairwise ICP rejected: translation={translation:.4f}, "
            f"rotation={rotation_deg:.3f} deg"
        )
        return cleaned_registration
    corrected_mesh = o3d.geometry.TriangleMesh(cleaned_registration.mesh)
    corrected_mesh.transform(correction)
    corrected_mesh.compute_vertex_normals()
    combined = correction @ cleaned_registration.transformation
    print(
        f"      Pairwise ICP accepted: fitness={result.fitness:.5f}, "
        f"RMSE={result.inlier_rmse:.5f}, translation={translation:.4f}, "
        f"rotation={rotation_deg:.3f} deg"
    )
    return RegistrationResult(
        corrected_mesh, combined, float(result.fitness), float(result.inlier_rmse)
    )


def show_pairwise_registration(
    unclean: o3d.geometry.TriangleMesh,
    cleaned: o3d.geometry.TriangleMesh,
    standard: o3d.geometry.TriangleMesh,
) -> None:
    diagonal = float(np.linalg.norm(np.ptp(np.asarray(standard.vertices), axis=0)))
    voxel = max(diagonal / 250.0, 0.03)
    print("      未刷和刷后模型均显示原始RGB，二者应在同一位置重合；关闭窗口继续。")
    o3d.visualization.draw_geometries(
        [_mesh_as_display_cloud(unclean, voxel), _mesh_as_display_cloud(cleaned, voxel)],
        window_name="Pairwise ICP - Unclean vs Cleaned (original RGB)",
        width=1280,
        height=800,
    )


def show_segmentation(
    mesh: o3d.geometry.TriangleMesh,
    face_labels: np.ndarray,
    region_names: Sequence[str],
    gingiva_flags: Sequence[bool],
    title: str,
) -> None:
    """Show face-centre region colours; gingiva is grey."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    centres = vertices[faces].mean(axis=1)
    palette = np.array([
        [0.90, 0.20, 0.20], [0.20, 0.65, 0.95], [0.95, 0.70, 0.15],
        [0.55, 0.25, 0.85], [0.15, 0.80, 0.55], [0.95, 0.35, 0.65],
        [0.35, 0.75, 0.20], [0.20, 0.80, 0.85],
    ])
    colours = np.full((len(faces), 3), 0.35, dtype=np.float64)
    for region_id, name in enumerate(region_names):
        selected = face_labels == region_id
        colours[selected] = 0.45 if gingiva_flags[region_id] else palette[region_id % len(palette)]
        print(f"      {name}: {int(selected.sum())} faces" + ("（牙龈，不积分）" if gingiva_flags[region_id] else ""))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(centres)
    cloud.colors = o3d.utility.Vector3dVector(colours)
    print("      不同颜色代表不同区域，灰色为牙龈/未匹配区域；关闭窗口继续。")
    o3d.visualization.draw_geometries([cloud], window_name=title, width=1280, height=800)


def make_uv_comparison(
    before_image: np.ndarray,
    before_mask: np.ndarray,
    after_image: np.ndarray,
    after_mask: np.ndarray,
    name: str,
) -> np.ndarray:
    """Build a labelled BGR image containing unclean and cleaned UV maps."""
    def panel(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        grey = np.full(image.shape, 245, dtype=np.uint8)
        grey[mask] = np.rint(np.clip(image[mask], 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    left, right = panel(before_image, before_mask), panel(after_image, after_mask)
    gap = np.full((left.shape[0], 12, 3), 210, dtype=np.uint8)
    result = np.concatenate([left, gap, right], axis=1)
    cv2.putText(result, f"{name} - UNCLEAN", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    x = left.shape[1] + gap.shape[1] + 20
    cv2.putText(result, f"{name} - CLEANED", (x, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 150, 0), 2)
    return result


def make_region_gray_cloud(
    mesh: o3d.geometry.TriangleMesh,
    face_indices: np.ndarray,
    luminance: np.ndarray,
) -> o3d.geometry.PointCloud:
    """Create a face-centre point cloud coloured with triangle mean grey."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)[face_indices]
    centres = vertices[faces].mean(axis=1)
    face_gray = luminance[faces].mean(axis=1)
    colors = np.repeat(face_gray[:, None], 3, axis=1)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(centres)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def show_region_gray_comparison(
    before_mesh: o3d.geometry.TriangleMesh,
    before_faces: np.ndarray,
    before_luminance: np.ndarray,
    after_mesh: o3d.geometry.TriangleMesh,
    after_faces: np.ndarray,
    after_luminance: np.ndarray,
    region_name: str,
    output_dir: Path,
    save_files: bool = True,
) -> None:
    """Show true 3-D region greys side by side and save both gray clouds."""
    before_cloud = make_region_gray_cloud(before_mesh, before_faces, before_luminance)
    after_cloud = make_region_gray_cloud(after_mesh, after_faces, after_luminance)
    if save_files:
        o3d.io.write_point_cloud(str(output_dir / f"{region_name}_unclean_gray3d.ply"), before_cloud)
        o3d.io.write_point_cloud(str(output_dir / f"{region_name}_cleaned_gray3d.ply"), after_cloud)

    before_points = np.asarray(before_cloud.points)
    after_display = o3d.geometry.PointCloud(after_cloud)
    width = float(np.ptp(before_points, axis=0).max()) if len(before_points) else 1.0
    # Both scans are registered to the same standard coordinates.  Moving only
    # the display copy makes a clear left/right comparison; saved PLYs remain in
    # their aligned standard coordinates.
    after_display.translate((width * 1.35, 0.0, 0.0))
    print(f"      3D灰度对比 {region_name}: 左=清洁前，右=清洁后；关闭窗口继续。")
    o3d.visualization.draw_geometries(
        [before_cloud, after_display],
        window_name=f"3D gray comparison - {region_name} (left unclean / right cleaned)",
        width=1400,
        height=800,
    )


def _surface_area(mesh: o3d.geometry.TriangleMesh, face_indices: np.ndarray) -> float:
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)[face_indices]
    return float(np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum() * 0.5)


def integrate_region_3d(
    mesh: o3d.geometry.TriangleMesh,
    face_indices: np.ndarray,
    luminance: np.ndarray,
) -> dict:
    """Integrate greyscale on the actual 3-D surface, independent of UV.

    Vertex luminance is averaged on each triangle.  The triangle contribution
    is its real 3-D area multiplied by grey (or darkness).  ICP is rigid, so it
    does not alter these areas.
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)[face_indices]
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    face_gray = luminance[faces].mean(axis=1)
    face_darkness = 1.0 - face_gray
    total_area = float(areas.sum())
    gray_integral = float(np.dot(areas, face_gray))
    darkness_integral = float(np.dot(areas, face_darkness))
    return {
        "surface_area": total_area,
        "face_count": int(len(faces)),
        "gray_integral_3d": gray_integral,
        "darkness_integral_3d": darkness_integral,
        "mean_gray_area_weighted": gray_integral / max(total_area, EPS),
        "mean_darkness_area_weighted": darkness_integral / max(total_area, EPS),
    }


def calculate_cleanliness(
    standard_model_path: os.PathLike | str,
    unclean_model_path: os.PathLike | str,
    cleaned_model_path: os.PathLike | str,
    template_dir: os.PathLike | str,
    output_dir: os.PathLike | str,
    uv_resolution: int = 1024,
    voxel_size: Optional[float] = None,
    max_label_distance: Optional[float] = None,
    gingiva_words: Sequence[str] = DEFAULT_GINGIVA_WORDS,
    visualize: Optional[bool] = None,
    show_uv: bool = True,
    show_matching_results: bool = True,
    show_point_cloud_comparison: bool = True,
    save_intermediate_results: bool = True,
) -> dict:
    """Run the full three-model measurement and write all results."""
    global all_stats
    if visualize is not None:
        show_uv = bool(visualize)
        show_matching_results = bool(visualize)
        show_point_cloud_comparison = bool(visualize)
    if uv_resolution < 256:
        raise ValueError("uv_resolution must be at least 256")
    out = Path(output_dir)
    uv_dir = out / "uv"
    gray3d_dir = out / "gray3d"
    cropped_dir = out / "cropped_meshes"
    out.mkdir(parents=True, exist_ok=True)
    if save_intermediate_results:
        uv_dir.mkdir(parents=True, exist_ok=True)
        gray3d_dir.mkdir(parents=True, exist_ok=True)
        cropped_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading meshes ...")
    standard = _load_mesh(standard_model_path)
    before = _load_mesh(unclean_model_path)
    after = _load_mesh(cleaned_model_path)

    print("[2/5] Registering unclean mesh to standard (PCA + ICP) ...")
    before_reg = register_mesh_to_standard(before, standard, voxel_size)
    print(f"      fitness={before_reg.fitness:.5f}, RMSE={before_reg.inlier_rmse:.5f}")
    if show_matching_results:
        show_registration(standard, before_reg.mesh, "ICP - Unclean vs Standard (automatic, no picking)")
    print("[3/5] Registering cleaned mesh to standard (PCA + ICP) ...")
    after_reg = register_mesh_to_standard(after, standard, voxel_size)
    print(f"      fitness={after_reg.fitness:.5f}, RMSE={after_reg.inlier_rmse:.5f}")
    if show_matching_results:
        show_registration(standard, after_reg.mesh, "ICP - Cleaned vs Standard (automatic, no picking)")

    print("[3b/5] Pairwise refinement: cleaned scan -> unclean scan ...")
    after_reg = refine_cleaned_to_unclean(after_reg, before_reg, standard, voxel_size)
    if show_matching_results:
        show_pairwise_registration(before_reg.mesh, after_reg.mesh, standard)

    if save_intermediate_results:
        o3d.io.write_triangle_mesh(str(out / "unclean_registered.ply"), before_reg.mesh)
        o3d.io.write_triangle_mesh(str(out / "cleaned_registered.ply"), after_reg.mesh)
        np.savetxt(out / "unclean_to_standard_transform.txt", before_reg.transformation, fmt="%.12g")
        np.savetxt(out / "cleaned_to_standard_transform.txt", after_reg.transformation, fmt="%.12g")

    print("[4/5] Cropping registered meshes directly with segmentation meshes ...")
    region_names, template_trees, gingiva_flags = load_template_region_references(
        template_dir, gingiva_words
    )
    std_v = np.asarray(standard.vertices)
    diagonal = float(np.linalg.norm(np.ptp(std_v, axis=0)))
    label_distance = float(max_label_distance or max(diagonal / 80.0, 0.20))
    before_labels = label_mesh_faces_from_templates(before_reg.mesh, template_trees, label_distance)
    after_labels = label_mesh_faces_from_templates(after_reg.mesh, template_trees, label_distance)
    if show_matching_results:
        show_segmentation(before_reg.mesh, before_labels, region_names, gingiva_flags, "Segments - Unclean")
        show_segmentation(after_reg.mesh, after_labels, region_names, gingiva_flags, "Segments - Cleaned")

    before_lum_raw = rgb_to_luminance(np.asarray(before_reg.mesh.vertex_colors))
    after_lum_raw = rgb_to_luminance(np.asarray(after_reg.mesh.vertex_colors))
    tooth_ids = np.array([i for i, flag in enumerate(gingiva_flags) if not flag], dtype=np.int32)
    before_tooth_faces = np.flatnonzero(np.isin(before_labels, tooth_ids))
    after_tooth_faces = np.flatnonzero(np.isin(after_labels, tooth_ids))
    if len(before_tooth_faces) == 0 or len(after_tooth_faces) == 0:
        raise RuntimeError("No non-gingiva faces were transferred; check ICP and template coordinates")
    before_tooth_vertices = np.unique(np.asarray(before_reg.mesh.triangles)[before_tooth_faces])
    after_tooth_vertices = np.unique(np.asarray(after_reg.mesh.triangles)[after_tooth_faces])
    before_lum, after_lum, exposure = normalise_luminance_pair(
        before_lum_raw,
        after_lum_raw,
        before_lum_raw[before_tooth_vertices],
        after_lum_raw[after_tooth_vertices],
    )

    # Integrate overall cleanliness from each complete scan instead of adding
    # cropped regions. Unmatched faces remain included; only basedown is out.
    excluded_ids = np.flatnonzero(np.asarray(gingiva_flags, dtype=bool))
    before_overall_faces = np.flatnonzero(~np.isin(before_labels, excluded_ids))
    after_overall_faces = np.flatnonzero(~np.isin(after_labels, excluded_ids))
    if len(before_overall_faces) == 0 or len(after_overall_faces) == 0:
        raise RuntimeError("No mesh faces remain after excluding basedown")
    overall_before_3d = integrate_region_3d(
        before_reg.mesh, before_overall_faces, before_lum
    )
    overall_after_3d = integrate_region_3d(
        after_reg.mesh, after_overall_faces, after_lum
    )

    print("[5/5] UV rasterisation and pixel integrals ...")
    details = []
    for region_id, name in enumerate(region_names):
        bf = np.flatnonzero(before_labels == region_id)
        af = np.flatnonzero(after_labels == region_id)

        # Save every available crop (including excluded gingiva and a region
        # present in only one scan) so segmentation errors remain inspectable.
        if save_intermediate_results and len(bf):
            o3d.io.write_triangle_mesh(
                str(cropped_dir / f"{name}_unclean_cropped.ply"),
                extract_face_mesh(before_reg.mesh, bf),
            )
        if save_intermediate_results and len(af):
            o3d.io.write_triangle_mesh(
                str(cropped_dir / f"{name}_cleaned_cropped.ply"),
                extract_face_mesh(after_reg.mesh, af),
            )
        if gingiva_flags[region_id]:
            print(f"      skip gingiva: {name}")
            continue
        if len(bf) == 0 or len(af) == 0:
            print(f"      skip empty region: {name}")
            continue
        b_img, b_mask, bm = rasterise_region_uv(
            before_reg.mesh, bf, before_lum, uv_resolution, name
        )
        a_img, a_mask, am = rasterise_region_uv(
            after_reg.mesh, af, after_lum, uv_resolution, name
        )
        comparison = make_uv_comparison(b_img, b_mask, a_img, a_mask, name)
        if save_intermediate_results:
            _save_uv_png(uv_dir / f"{name}_unclean.png", b_img, b_mask)
            _save_uv_png(uv_dir / f"{name}_cleaned.png", a_img, a_mask)
            cv2.imwrite(str(uv_dir / f"{name}_comparison.png"), comparison)
        if show_uv:
            preview_width = min(comparison.shape[1], 1600)
            preview_height = max(1, round(comparison.shape[0] * preview_width / comparison.shape[1]))
            preview = cv2.resize(comparison, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
            cv2.imshow(f"UV comparison - {name} (press any key)", preview)
            cv2.waitKey(0)
            cv2.destroyWindow(f"UV comparison - {name} (press any key)")

        # UV above is display-only. The score below uses true 3-D triangle area.
        b3d = integrate_region_3d(before_reg.mesh, bf, before_lum)
        a3d = integrate_region_3d(after_reg.mesh, af, after_lum)
        # Always save the 3-D greys; open a side-by-side window in visual mode.
        if show_point_cloud_comparison:
            show_region_gray_comparison(
                before_reg.mesh, bf, before_lum,
                after_reg.mesh, af, after_lum,
                name, gray3d_dir, save_files=save_intermediate_results,
            )
        elif save_intermediate_results:
            o3d.io.write_point_cloud(
                str(gray3d_dir / f"{name}_unclean_gray3d.ply"),
                make_region_gray_cloud(before_reg.mesh, bf, before_lum),
            )
            o3d.io.write_point_cloud(
                str(gray3d_dir / f"{name}_cleaned_gray3d.ply"),
                make_region_gray_cloud(after_reg.mesh, af, after_lum),
            )
        ratio = a3d["darkness_integral_3d"] / max(b3d["darkness_integral_3d"], EPS)
        cleanliness = float(np.clip((1.0 - ratio) * 100.0, -100.0, 100.0))
        before_area = b3d["surface_area"]
        after_area = a3d["surface_area"]
        item = {
            "name": name,
            "cleanliness": cleanliness,
            "integral_ratio_after_before": float(ratio),
            "before_surface_area": before_area,
            "after_surface_area": after_area,
            "original_area": before_area,
            "before_faces": int(len(bf)),
            "after_faces": int(len(af)),
            "unclean_3d": b3d,
            "cleaned_3d": a3d,
            "uv_unclean_display_only": bm,
            "uv_cleaned_display_only": am,
            # Compatibility fields expected by the current CleanUI table.
            "picked_gray_mean": b3d["mean_darkness_area_weighted"],
            "remaining_gray_mean": a3d["mean_darkness_area_weighted"],
            "remaining_area": after_area,
            "denominator": b3d["darkness_integral_3d"],
            "numerator": a3d["darkness_integral_3d"],
        }
        details.append(item)
        print(f"      {name}: {cleanliness:.2f}%")

    total_before = overall_before_3d["darkness_integral_3d"]
    total_after = overall_after_3d["darkness_integral_3d"]
    total_cleanliness = float(np.clip((1.0 - total_after / max(total_before, EPS)) * 100.0, -100.0, 100.0))
    report = {
        "method": "independent ICP + complete scan meshes excluding basedown + true 3-D triangle-area weighted darkness integral",
        "cropped_mesh_directory": str(cropped_dir) if save_intermediate_results else None,
        "intermediate_results_saved": bool(save_intermediate_results),
        "formula": "cleanliness(%) = (1 - cleaned_mesh_darkness_integral_excluding_basedown / unclean_mesh_darkness_integral_excluding_basedown) * 100",
        "uv_role": "visualisation only; UV values do not participate in cleanliness",
        "gingiva_excluded": [n for n, flag in zip(region_names, gingiva_flags) if flag],
        "uv_resolution": int(uv_resolution),
        "max_label_distance": label_distance,
        "exposure_normalisation": exposure,
        "registration": {
            "unclean": {"fitness": before_reg.fitness, "inlier_rmse": before_reg.inlier_rmse},
            "cleaned": {"fitness": after_reg.fitness, "inlier_rmse": after_reg.inlier_rmse},
        },
        "overall_cleanliness": total_cleanliness,
        "total_unclean_darkness_integral": float(total_before),
        "total_cleaned_darkness_integral": float(total_after),
        "total_unclean_surface_area": float(overall_before_3d["surface_area"]),
        "total_cleaned_surface_area": float(overall_after_3d["surface_area"]),
        "overall_unclean_3d": overall_before_3d,
        "overall_cleaned_3d": overall_after_3d,
        "total_count": len(details),
        "details": details,
    }
    all_stats = details
    with open(out / "cleanliness_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    with open(out / "cleanliness_report.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["region", "cleanliness_percent", "after_before_ratio", "unclean_darkness_integral", "cleaned_darkness_integral", "unclean_area_mm2", "cleaned_area_mm2"])
        for x in details:
            writer.writerow([x["name"], x["cleanliness"], x["integral_ratio_after_before"], x["unclean_3d"]["darkness_integral_3d"], x["cleaned_3d"]["darkness_integral_3d"], x["before_surface_area"], x["after_surface_area"]])
    print(f"Overall cleanliness: {total_cleanliness:.2f}%")
    print(f"Report: {out / 'cleanliness_report.json'}")
    return report


def process_dental_mesh_registration(
    source_mesh_path: os.PathLike | str,
    target_pcd_path: os.PathLike | str,
    template_dir: os.PathLike | str,
    output_base_dir: os.PathLike | str,
    cleaned_model_path: Optional[os.PathLike | str] = None,
    **kwargs,
) -> dict:
    """Compatibility wrapper for older callers.

    Old argument names mean: source=unclean, target=standard.  The cleaned path
    must now be supplied because a before/after ratio cannot be inferred from a
    single scan.
    """
    if cleaned_model_path is None:
        default = Path(target_pcd_path).parent / "IO9-3 LowerJawScan.ply"
        if not default.exists():
            raise ValueError("cleaned_model_path is required for before/after cleanliness")
        cleaned_model_path = default
    return calculate_cleanliness(
        standard_model_path=target_pcd_path,
        unclean_model_path=source_mesh_path,
        cleaned_model_path=cleaned_model_path,
        template_dir=template_dir,
        output_dir=output_base_dir,
        **kwargs,
    )


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Calculate dental cleanliness using true 3-D triangle-area weighted greyscale")
    parser.add_argument("--standard", default=str(here / "pointsdata" / "LowerJawScans.ply"))
    parser.add_argument("--unclean", default=str(here / "pointsdata" / "UncleanLowerJawScan.ply"))
    parser.add_argument("--cleaned", default=str(here / "pointsdata" / "IO9-3 LowerJawScan.ply"))
    parser.add_argument("--templates", default=str(here / "segmentationsply"))
    parser.add_argument("--output", default=str(here / "cleanliness_results_area_weighted"))
    parser.add_argument("--uv-resolution", type=int, default=1024)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--max-label-distance", type=float, default=None)
    parser.add_argument("--no-visualize", action="store_true", help="Do not open ICP/segment/UV windows")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    calculate_cleanliness(
        args.standard,
        args.unclean,
        args.cleaned,
        args.templates,
        args.output,
        uv_resolution=args.uv_resolution,
        voxel_size=args.voxel_size,
        max_label_distance=args.max_label_distance,
        visualize=not args.no_visualize,
    )
