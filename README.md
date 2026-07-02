# UClean - 牙模自动化评估系统

UClean 是一套面向牙模扫描结果的 3D 清洁度评估工具，支持配准、交互分割、颜色选点评估与结果汇总。

## 核心流程

1. 使用源 mesh 与目标点云进行配准，并支持交互选点确认。
2. 基于模板对牙模进行自动分割，输出映射后的 mesh 结果。
3. 通过 `mapped_meshes` 下的点云/mesh 完成颜色选点清洁度评估。
4. 汇总各分区清洁度指标，输出面积、污染面积与全口加权清洁度。

## 项目结构

- `CleanUI.py`：自动化评估主界面，负责路径配置、调用后端流程并展示结果。
- `MeshSegmentationUI.py`：手动涂抹分割与逐区清洁度评估界面。
- `GeneratePathOffset66initAllnewdorobotsnew.py`：核心算法，包含配准、分割映射、批量清洁度计算与 `AdvancedCleanerWindow`。
- `CleanSingletooth2.py`：单牙/分区点云清理窗口。
- `segmentationsply/`：牙齿分割模板目录。
- `segementations/`：默认输出目录，包含 `mapped_meshes`、`cleaned_results` 等结果。

## 使用说明

1. 安装依赖：`PyQt5`、`pyvista`、`pyvistaqt`、`open3d`、`numpy`、`scipy`、`scikit-image` 等。
2. 运行 `CleanUI.py` 配置源 mesh、目标点云、模板目录与输出目录。
3. 运行 `MeshSegmentationUI.py` 进行手动标注分区与逐区选点评估。
4. 评估结果会保存到输出目录中的 `cleanliness_report.json`。

## 清洁度公式

```text
cleanliness = (remaining_gray_mean × remaining_area) / (picked_gray_mean × segment_area)
```

- `remaining_gray_mean`：剩余清洁区域的平均灰度
- `picked_gray_mean`：选中区域的平均灰度
- `remaining_area`：剩余清洁面的面积
- `segment_area`：分区总面积

## 注意事项

- 配准与清洁度窗口使用 Open3D/PyVista 可视化，建议在桌面环境运行。
- 大体积 `.ply` / `.pcd` 数据文件已被 `.gitignore` 忽略，避免仓库膨胀。
- 输出目录中的结果文件可按需保留或重新生成。
