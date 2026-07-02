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

### 环境准备

1. 安装 Python 依赖：`PyQt5`、`pyvista`、`pyvistaqt`、`open3d`、`numpy`、`scipy`、`scikit-image` 等。
2. 建议在桌面环境运行，确保 Open3D/PyVista 可视化窗口可正常显示。

### CleanUI 自动化评估

1. 运行 `python CleanUI.py`。
2. 在界面中配置：
   - 训练路径/源 mesh：`Source`
   - 扫描路径/目标点云：`Target`
   - 模板目录：牙齿分割模板文件夹
   - 输出目录：结果保存根目录
3. 点击「开始处理流程」后，系统会进入配准选点与后续计算流程。
4. 若弹出确认框，请根据实际配准/选点结果确认是否继续。
5. 结果会显示在界面表格中，并同步输出到 `mapped_meshes/cleaned_results/cleanliness_report.json`。

### MeshSegmentationUI 手动分割与评估

1. 运行 `python MeshSegmentationUI.py`。
2. 点击「加载原始 Mesh」导入需要分割的牙模。
3. 使用笔刷/橡皮擦进行涂抹分区，并输入类别名称保存分区。
4. 点击「分割并评估清洁度」后，系统会导出分区并逐个打开选点窗口。
5. 在选点窗口中右键选色过滤，完成后点击保存；按 `Esc` 可跳过该区域。
6. 界面会汇总各分区清洁度，并输出报告到所选输出目录。

### 结果查看

- 界面表格：分区面积、清洁度、污染面积、相对贡献与全口总计。
- 报告文件：`cleanliness_report.json`，包含各分区详情与全口加权清洁度。

## 清洁度计算原理

### 基本思想

系统通过颜色灰度区分“清洁区域”和“污染区域”，并基于面片面积做加权，避免单纯按点数评估带来的偏差。

### 核心公式

```text
cleanliness = (remaining_gray_mean × remaining_area) / (picked_gray_mean × segment_area)
```

- `remaining_gray_mean`：剩余清洁区域的平均灰度
- `picked_gray_mean`：选中区域的平均灰度
- `remaining_area`：剩余清洁面的面积
- `segment_area`：分区总面积

### 计算说明

- 灰度映射：颜色越黑权重越大，白=0，黑=1。
- 分子：剩余清洁区域的平均灰度 × 剩余清洁面积，反映“干净部分”的总体保留程度。
- 分母：选中区域的平均灰度 × 分区总面积，反映“该分区整体被污染的程度”。
- 全口加权清洁度：以各分区面积为权重，综合所有分区的清洁度表现。

### 输出指标

- 分区清洁度（%）：单个分区的清洁程度。
- 污染面积（mm²）：`segment_area - remaining_area`。
- 全口加权清洁度（%）：全部分区按面积加权后的总体清洁水平。

## 注意事项

- 配准与清洁度窗口使用 Open3D/PyVista 可视化，建议在桌面环境运行。
- 大体积 `.ply` / `.pcd` 数据文件已被 `.gitignore` 忽略，避免仓库膨胀。
- 输出目录中的结果文件可按需保留或重新生成。
