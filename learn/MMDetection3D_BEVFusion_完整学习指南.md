# MMDetection3D 框架与 BEVFusion 算法完整学习指南

> **适用人群**: 有 Python/PyTorch 基础，想系统学习 3D 目标检测框架和 BEVFusion 多模态融合算法。
> **核心思路**: 以 BEVFusion 为贯穿全文的实例——每学一个框架概念，立即看它在 BEVFusion 中的具体实现。
> **版本**: MMDetection3D v1.4.0

---

## 目录

**Stage 1: 基础认知（第1-2天）**
- [第1章: 3D 目标检测快速入门](#第1章-3d-目标检测快速入门)
- [第2章: 环境搭建与一分钟跑通](#第2章-环境搭建与一分钟跑通)

**Stage 2: 框架骨架（第3-6天）**
- [第3章: 框架总览——Registry 与配置驱动](#第3章-框架总览registry-与配置驱动)
- [第4章: 配置系统完全理解](#第4章-配置系统完全理解)
- [第5章: 数据流与数据结构](#第5章-数据流与数据结构)

**Stage 3: BEVFusion 深入+实践（第7-14天）**
- [第6章: BEVFusion 模型主类完整剖析](#第6章-bevfusion-模型主类完整剖析)
- [第7章: 视角变换——LSS/DepthLSS 核心](#第7章-视角变换lssdepthlss-核心)
- [第8章: 点云特征提取](#第8章-点云特征提取)
- [第9章: TransFusionHead——Transformer 检测头](#第9章-transfusionheadtransformer-检测头)
- [第10章: 融合模块与辅助组件](#第10章-融合模块与辅助组件)
- [第11章: CUDA 算子实现](#第11章-cuda-算子实现)
- [第12章: 完整代码追踪——端到端数据流](#第12章-完整代码追踪端到端数据流)
- [第13章: 动手实践指南](#第13章-动手实践指南)
- [第14章: 性能调试与优化](#第14章-性能调试与优化)

**附录**
- [附录A: 关键代码文件索引](#附录a-关键代码文件索引)
- [附录B: 术语表](#附录b-术语表)
- [附录C: BEVFusion 配置参数速查](#附录c-bevfusion-配置参数速查)
- [附录D: 常见命令速查](#附录d-常见命令速查)
- [附录E: BEVFusion vs 其他方法对比](#附录e-bevfusion-vs-其他方法对比)

---

## 第1章: 3D 目标检测快速入门

### 1.1 2D vs 3D 检测

```
2D 检测输出: [类别, bbox_2d(x1,y1,x2,y2), 置信度]                   ← 4 自由度
3D 检测输出: [类别, bbox_3d(中心x,y,z, 长宽高w,l,h, 旋转角θ), 置信度] ← 7~9 自由度
```

3D 检测的核心难点：
- **深度歧义**: 单张 2D 图像无法直接获知物体距离
- **点云稀疏性**: LiDAR 点云在远处非常稀疏，小物体可能只有几个点
- **多模态对齐**: 相机和 LiDAR 数据坐标系不同，需要精确标定

### 1.2 传感器与数据格式

**点云 (Point Cloud)**:
```python
# shape: (N, 4+) , 每行 [x, y, z, intensity, ...]
points = [[1.2, 0.5, 0.8, 0.9],
          [2.1, -0.3, 0.6, 0.7], ...]
```

**多视角图像**: nuScenes 使用 6 个相机（FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT），每张 (3, H, W)。

**相机内外参**:

```
内参矩阵 K (3x3):              外参矩阵 [R|T] (3x4):
┌ fx  0  cx ┐                  ┌ R00 R01 R02 Tx ┐
│  0  fy  cy │                  │ R10 R11 R12 Ty │
└  0   0   1 ┘                  └ R20 R21 R22 Tz ┘

坐标系转换: P_image = K * (R * P_world + T)
```

### 1.3 关键数据集

| 数据集 | 传感器 | 规模 | 类别数 | BEVFusion 使用 |
|--------|--------|------|--------|---------------|
| **nuScenes** | 32线LiDAR + 6相机 | 1000场景 | 10类 | **是（主要）** |
| KITTI | 64线LiDAR + 2相机 | 15K帧 | 3类 | 否 |
| Waymo | 64线LiDAR + 5相机 | 1150场景 | 4类 | 可选 |

BEVFusion LiDAR+Camera 在 nuScenes val 上达到 **71.4 NDS / 68.6 mAP**。

### 1.4 3D 检测方法分类

```
3D 目标检测方法
├── 基于点云
│   ├── Point-based: PointNet++, PointRCNN
│   ├── Voxel-based:  VoxelNet, SECOND, PointPillars, CenterPoint ← BEVFusion LiDAR 分支
│   └── 混合:         PV-RCNN
├── 基于图像
│   └── 单目/多目:    FCOS3D, PGD, DETR3D, BEVDet
└── 多模态融合
    ├── Point-level:   MVX-Net (点级融合)
    └── BEV-level:     BEVFusion ← 本文主角
```

BEVFusion 的定位：**Voxel-based LiDAR + 多目相机的 BEV 空间融合**。

### 1.5 nuScenes 评测指标速览

- **mAP** (mean Average Precision): 基于 BEV 中心距离 (0.5m, 1m, 2m, 4m) 的检测精度
- **NDS** (nuScenes Detection Score): mAP + 其他属性（平移/尺度/朝向/速度/属性误差）的加权综合分
- **mATE, mASE, mAOE, mAVE, mAAE**: 5 个 True Positive 误差指标

---

## 第2章: 环境搭建与一分钟跑通

### 2.1 依赖关系

```
PyTorch >= 1.8.0
  └── mmengine >= 0.8.0    ← 训练引擎（Runner, Registry, Hook, Config）
       └── mmcv >= 2.0.0   ← 计算机视觉基础库（CNN组件, CUDA Ops）
            └── mmdet >= 3.0.0  ← 2D 检测框架（BEVFusion 复用了 FocalLoss, FPN 等）
                 └── mmdet3d v1.4.0  ← 3D 检测框架（本文）
```

### 2.2 安装步骤

```bash
# 1. 创建环境
conda create -n bevfusion python=3.8 -y && conda activate bevfusion

# 2. 安装 PyTorch (CUDA 11.6 示例)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# 3. 安装 MMEngine, MMCV, MMDet
pip install mmengine mmcv==2.0.0 mmdet==3.0.0

# 4. 安装 MMDetection3D
cd mmdetection3d && pip install -e .

# 5. 编译 BEVFusion CUDA 算子
cd projects/BEVFusion && python setup.py develop
```

### 2.3 验证安装

```bash
cd /path/to/mmdetection3d
python tools/train.py projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
```

如果配置正确但无数据，会在 DataLoader 阶段报错——这说明框架已正常加载。

### 2.4 运行 Demo

```bash
# LiDAR-only demo
python projects/BEVFusion/demo/multi_modality_demo.py \
    projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    <checkpoint_path> \
    <point_cloud_file>

# LiDAR+Camera demo
python projects/BEVFusion/demo/multi_modality_demo.py \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    <checkpoint_path> \
    <point_cloud_file>
```

### 2.5 这个 Demo 背后发生了什么？（5 句话概括）

1. 配置文件被 `mmengine.Config.fromfile()` 读取，构建成嵌套字典
2. `Runner.from_cfg(cfg)` 通过 Registry 查找 `type='BEVFusion'` 对应的类并实例化
3. 点云被 Voxelization 切成体素 → SparseEncoder 提取特征 → BEV 特征图
4. （如果是 LiDAR+Camera）图像经 Swin-T → GeneralizedLSSFPN → DepthLSSTransform → 另一张 BEV 特征图
5. 两张 BEV 特征拼接 → ConvFuser → SECOND → SECONDFPN → TransFusionHead → 3D 检测框

---

## 第3章: 框架总览——Registry 与配置驱动

### 3.1 MMDetection3D 在 OpenMMLab 生态中的位置

MMDetection3D 不是独立的框架——它建立在 3 层抽象之上：

```
┌──────────────────────────────────────┐
│          mmdet3d v1.4.0              │  ← 3D 检测逻辑: detectors, voxel encoders,
│  (BEVFusion, CenterPoint, PointPillars)│     3D bbox, point cloud ops, evaluation
├──────────────────────────────────────┤
│          mmdet v3.0.0                │  ← 2D 检测逻辑: FPN, FocalLoss, DETR decoder
│  (复用 2D backbone, loss, head)       │
├──────────────────────────────────────┤
│          mmcv v2.0.0                 │  ← CV 基础库: ConvModule, CUDA ops, spconv
│  (SwinTransformer, ResNet, sparse conv)
├──────────────────────────────────────┤
│          mmengine v0.8.0             │  ← 训练基础设施: Runner, Registry, Hook,
│  (Runner, Config, Registry, Logger)   │     Checkpoint, Logging
└──────────────────────────────────────┘
```

**关键原则**: 如果你想修改训练流程（LR schedule、log、checkpoint 等），去理解 mmengine。如果你要改检测算法（backbone、head、loss），主要关注 mmdet3d + mmdet。

### 3.2 17 个 Registry——框架的"骨架"

Registry 是 MMDetection3D 最核心的设计模式。它维护一个**名称→类的映射**，让配置文件可以通过 `type` 字符串动态创建任何组件。

所有 17 个 Registry 定义在 [`mmdet3d/registry.py`](mmdet3d/registry.py)：

| Registry | 管理内容 | 自动发现位置 | BEVFusion 中的实例 |
|----------|---------|--------------|-------------------|
| `MODELS` | 所有 nn.Module 子类 | `mmdet3d.models` | `BEVFusion`, `TransFusionHead`, `ConvFuser`, `DepthLSSTransform`, `GeneralizedLSSFPN` |
| `TRANSFORMS` | 数据增强 | `mmdet3d.datasets.transforms` | `ImageAug3D`, `BEVFusionRandomFlip3D`, `BEVFusionGlobalRotScaleTrans` |
| `TASK_UTILS` | Anchor/BBox 操作 | `mmdet3d.models` | `TransFusionBBoxCoder`, `HungarianAssigner3D`, `BBoxBEVL1Cost` |
| `DATASETS` | 数据集类 | `mmdet3d.datasets` | `NuScenesDataset`, `CBGSDataset` |
| `METRICS` | 评测指标 | `mmdet3d.evaluation` | `NuScenesMetric` |
| `VISUALIZERS` | 可视化 | `mmdet3d.visualization` | `Det3DLocalVisualizer` |
| `OPTIMIZERS` | 优化器 | `mmdet3d.engine` | `AdamW` |
| `PARAM_SCHEDULERS` | LR 调度 | `mmdet3d.engine` | `CosineAnnealingLR` |
| `HOOKS` | 训练钩子 | `mmdet3d.engine.hooks` | `DisableObjectSampleHook` |
| `RUNNERS` | 运行器 | `mmdet3d.engine` | `EpochBasedRunner` |
| `LOOPS` | 训练循环 | `mmdet3d.engine` | `EpochBasedTrainLoop` |
| `DATASETS` | 数据集 | `mmdet3d.datasets` | `NuScenesDataset` |
| `DATA_SAMPLERS` | 采样器 | `mmdet3d.datasets` | `DefaultSampler` |
| `MODEL_WRAPPERS` | 分布式包装 | `mmdet3d.models` | `MMDistributedDataParallel` |
| `WEIGHT_INITIALIZERS` | 权重初始化 | `mmdet3d.models` | `Pretrained` (Swin-T 的 ImageNet 预训练) |
| `INFERENCERS` | 推理器 | `mmdet3d.api.inferencers` | `LidarDet3DInferencer` |
| `OPTIM_WRAPPERS` | 优化器包装 | `mmdet3d.engine` | `OptimWrapper` |
| `OPTIM_WRAPPER_CONSTRUCTORS` | 优化器构造器 | `mmdet3d.engine` | 很少自定义 |

**Registry 工作原理**（以 MODELS 为例）：

```python
# Step 1: 定义 Registry（在 registry.py 中）
MODELS = Registry('model', parent=MMENGINE_MODELS, locations=['mmdet3d.models'])

# Step 2: 注册类（在 bevfusion.py 中）
@MODELS.register_module()
class BEVFusion(Base3DDetector):
    ...

# Step 3: 通过配置构建（在 Runner 内部自动完成）
cfg = dict(type='BEVFusion', ...)           # 从配置文件读取
model = MODELS.build(cfg)                    # 等价于 BEVFusion(...)
```

### 3.3 配置驱动设计——为什么改配置就能换模型

配置文件中的所有 `type` 字段都被 Registry 解析为实际类。以下面的 BEVFusion 模型配置为例：

```python
model = dict(
    type='BEVFusion',                           # → MODELS.build() → BEVFusion()
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',            # → MODELS.build() → Det3DDataPreprocessor()
        voxelize_cfg=dict(...)),
    pts_voxel_encoder=dict(type='HardSimpleVFE'), # → MODELS.build() → HardSimpleVFE()
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder', ...),      # → MODELS.build() → BEVFusionSparseEncoder()
    bbox_head=dict(
        type='TransFusionHead',                   # → MODELS.build() → TransFusionHead()
        ...
    )
)
```

当你在配置中将 `type='TransFusionHead'` 换成 `type='CenterPointHead'`，整个检测头就会替换——不需要修改任何 Python 代码。

**跨库引用**: `type='mmdet.SwinTransformer'` 中的 `mmdet.` 前缀告诉 Registry 去 MMDet 的注册表中查找。

### 3.4 `_base_` 继承机制

每个实验配置文件通过 `_base_` 列表继承通用配置片段：

```python
# bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
_base_ = ['../../../configs/_base_/default_runtime.py']  # 继承 runtime 默认值

# bevfusion_lidar-cam_...py
_base_ = ['./bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py']  # 继承 LiDAR-only
```

继承规则：子配置的字段**覆盖**父配置的同名字段，list 类型的 pipeline 字段**完全替换**。

### 3.5 一次训练的生命周期

```
tools/train.py
  │
  ├─ Config.fromfile(args.config)          # 读取+继承解析配置
  ├─ register_all_modules()                # 导入 mmdet3d 所有模块, 触发注册
  │
  └─ Runner.from_cfg(cfg)                  # 构建 Runner
       │
       ├─ MODELS.build(cfg.model)           # 构建 BEVFusion 模型
       │    └─ BEVFusion.__init__()
       │         ├─ MODELS.build(bbox_head)      # → TransFusionHead()
       │         └─ MODELS.build(pts_backbone)   # → SECOND()
       │
       ├─ DATASETS.build(cfg.train_dataloader.dataset)  # 构建数据集
       │    └─ NuScenesDataset.__init__(pipeline=...)
       │         └─ Compose([LoadPointsFromFile, ..., Pack3DDetInputs])
       │
       ├─ OPTIMIZERS.build(...)             # 构建优化器
       └─ PARAM_SCHEDULERS.build(...)       # 构建 LR 调度器
       │
       └─ runner.train()                    # 开始训练
            └─ EpochBasedTrainLoop.run()
                 └─ for batch in dataloader:
                      data = model.data_preprocessor(batch)
                      losses = model(data, mode='loss')  # BEVFusion.loss()
                      optim_wrapper.update_params(losses['loss'])
```

---

## 第4章: 配置系统完全理解

### 4.1 配置文件结构解剖

一个完整的配置文件包含以下核心段：

| 段 | 作用 | BEVFusion 中的关键值 |
|----|------|---------------------|
| `model` | 神经网络架构定义 | `type='BEVFusion'`, Swin-T, SECOND, TransFusionHead |
| `train_pipeline` | 训练数据增强链 | LoadPoints → MultiSweeps → Aug → Pack |
| `train_dataloader` | DataLoader 配置 | batch_size=4, CBGSDataset wrapper |
| `val_dataloader` | 验证 DataLoader | batch_size=1, 无数据增强 |
| `optim_wrapper` | 优化器+梯度裁剪 | AdamW(lr=1e-4), clip_grad=35 |
| `param_scheduler` | LR + Momentum 调度 | CosineAnnealing warmup + decay |
| `train_cfg` | 训练循环参数 | max_epochs=20 (LiDAR) / 6 (Fusion) |
| `val_evaluator` / `test_evaluator` | 评测指标 | `NuScenesMetric` |
| `default_hooks` | 日志/checkpoint | LoggerHook(interval=50) |
| `custom_hooks` | 自定义钩子 | `DisableObjectSampleHook` (在 epoch 15 禁用 GT 采样) |

### 4.2 BEVFusion LiDAR 配置逐段解读

**model.data_preprocessor** [bevfusion_lidar config:48-54](projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py#L48-L54):
- `Det3DDataPreprocessor`: 负责将原始数据（点云列表、图像列表）处理为 batch tensor，执行归一化和体素化
- `voxelize_cfg`: 定义 Voxelization 配置——这部分会在 BEVFusion.__init__ 中被 `pop` 出来，单独构建为 `self.pts_voxel_layer`

**model.pts_voxel_encoder** [bevfusion_lidar config:55](projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py#L55):
- `HardSimpleVFE`: 最简单的体素特征编码器——对体素内的点取均值

**model.pts_middle_encoder** [bevfusion_lidar config:56-64](projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py#L56-L64):
- `BEVFusionSparseEncoder`: 3D 稀疏卷积编码器，4 个 stage
- `sparse_shape=[1440, 1440, 41]`: 体素网格维度 (H, W, D)
- point_cloud_range `[-54, -54, -5, 54, 54, 3]` 除以 voxel_size `[0.075, 0.075, 0.2]` = 网格维度 `[1440, 1440, 40]`（约等于 41）

**model.pts_backbone + pts_neck** [bevfusion_lidar config:66-81](projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py#L66-L81):
- `SECOND`: 两个 block (5层, 5层)，stride [1, 2]，输出通道 [128, 256]
- `SECONDFPN`: 将两个尺度的特征上采样 concat → [256, 256]

**model.bbox_head** [bevfusion_lidar config:82-152](projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py#L82-L152):
- `TransFusionHead`: 1 层 Transformer decoder，200 个 proposal
- Hungarian 匹配 (FocalLossCost + BEVL1Cost + IoUCost)
- code_size=10: (x, y, z, w, l, h, sin(θ), cos(θ), vx, vy)
- code_weights 中 velocity 权重仅 0.2（速度难学，降低惩罚）

**train_cfg** [bevfusion_lidar config:105-114](projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py#L105-L114):
- `out_size_factor=8`: SparseEncoder 8× 下采样，BEV 特征尺寸从 1440 → 180
- `gaussian_overlap=0.1, min_radius=2`: 高斯热力图生成参数

**param_scheduler** [bevfusion_lidar config:322-361](projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py#L322-L361):
- Epoch 0-8: Cosine warmup, LR 从 0 → 1e-3
- Epoch 8-20: Cosine decay, LR 从 1e-3 → 1e-6
- Momentum 同步变化: 0 → 0.85/0.95 → 1

### 4.3 BEVFusion LiDAR+Camera 配置的增量变化

基于 LiDAR-only 配置，**新增/修改**以下内容 [bevfusion_lidar-cam config](projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py):

| 变更项 | LiDAR-only | LiDAR+Camera |
|--------|-----------|--------------|
| `input_modality` | `use_camera=False` | `use_camera=True` |
| `img_backbone` | 无 | `mmdet.SwinTransformer` (Swin-T, embed_dim=96) |
| `img_neck` | 无 | `GeneralizedLSSFPN` (in=[192,384,768], out=256) |
| `view_transform` | 无 | `DepthLSSTransform` (in=256, out=80, downsample=2) |
| `fusion_layer` | 无 | `ConvFuser` (in=[80,256], out=256) |
| `data_preprocessor.mean/std` | 无 | [123.675, 116.28, 103.53] / [58.395, 57.12, 57.375] |
| `train_pipeline` | 纯 LiDAR 增强 | + `BEVLoadMultiViewImageFromFiles` + `ImageAug3D` + `GridMask(prob=0)` |
| `lr` | 1e-4 | 2e-4 |
| `max_epochs` | 20 | 6 |
| `param_scheduler` | 20 epoch schedule | 6 epoch schedule |
| 训练策略 | 从头训练 | 通常基于 LiDAR 预训练权重 fine-tune |

---

## 第5章: 数据流与数据结构

### 5.1 数据增强 Pipeline 链

**LiDAR-only 训练 Pipeline**:
```
LoadPointsFromFile          # 读取当前帧点云 (N, 5): [x, y, z, intensity, ring_index]
  → LoadPointsFromMultiSweeps  # 加载 9 帧历史点云, 补偿 ego-motion
  → LoadAnnotations3D          # 加载 GT 3D boxes + labels
  → ObjectSample               # GT 数据库采样增强 (复制其他场景物体)
  → GlobalRotScaleTrans        # 旋转/平移/缩放 (标注顺序 S-R-T)
  → BEVFusionRandomFlip3D      # 随机水平/垂直翻转
  → PointsRangeFilter          # 按范围过滤点云
  → ObjectRangeFilter          # 按范围过滤 GT box
  → ObjectNameFilter           # 按类别名过滤
  → PointShuffle               # 打乱点顺序
  → Pack3DDetInputs            # 打包为 model 输入格式
```

**LiDAR+Camera 额外步骤**:
```
BEVLoadMultiViewImageFromFiles  # 加载多视角图像 + 计算 cam2lidar/lidar2img
  → ImageAug3D                   # 图像 resize/crop/flip/rotate, 记录 img_aug_matrix
  → BEVFusionGlobalRotScaleTrans # 替换 GlobalRotScaleTrans, 顺序改为 R-T-S
```

### 5.2 Det3DDataSample——框架的通用数据载体

每个样本被封装为 `Det3DDataSample` 对象 [`mmdet3d/structures/det3d_data_sample.py`](mmdet3d/structures/det3d_data_sample.py)：

```python
Det3DDataSample
├── gt_instances_3d: InstanceData      # 训练时的 GT
│   ├── bboxes_3d: LiDARInstance3DBoxes  # (M, 7) [x, y, z, w, l, h, yaw]
│   ├── labels_3d: Tensor                # (M,) 类别索引
│   └── scores_3d: Tensor                # (M,) 全 1
│
├── pred_instances_3d: InstanceData     # 推理时的预测输出
│   ├── bboxes_3d: LiDARInstance3DBoxes  # (K, 7/9)
│   ├── labels_3d: Tensor                # (K,)
│   └── scores_3d: Tensor                # (K,)
│
└── metainfo: dict                      # 元信息（不参与梯度计算）
     ├── 'cam2img': (N_cam, 4, 4)       # 相机内参矩阵
     ├── 'lidar2cam': (N_cam, 4, 4)     # LiDAR→相机外参
     ├── 'cam2lidar': (N_cam, 4, 4)     # 相机→LiDAR (从 lidar2cam 求逆得到)
     ├── 'lidar2img': (N_cam, 4, 4)     # LiDAR→图像 (cam2img @ lidar2cam)
     ├── 'lidar_aug_matrix': (4, 4)     # LiDAR 增强矩阵 (累计乘积)
     ├── 'img_aug_matrix': (N_cam, 4, 4) # 每张图像的增强矩阵
     └── 'box_type_3d': type            # 'LiDAR' (3D box 坐标类型)
```

### 5.3 BEVFusion 特有数据加载

**BEVLoadMultiViewImageFromFiles** [`projects/BEVFusion/bevfusion/loading.py`](projects/BEVFusion/bevfusion/loading.py):

核心职责：
1. 加载 6 个相机的图像
2. 从 `results['images']` 中提取 `lidar2cam` 矩阵
3. **计算 `cam2lidar`**: `R_lidar2cam^T @ [I | -T]` = camera→LiDAR 的变换矩阵
4. **计算 `lidar2img`**: `K @ [R | T]` = LiDAR 坐标系→像素坐标系的变换矩阵
5. Multi-sweep 处理：加载历史帧图像并根据 ego-motion 补偿外参

关键代码路径 [loading.py:136-158](projects/BEVFusion/bevfusion/loading.py#L136-L158)：

```python
lidar2cam_array = np.array(cam_item['lidar2cam']).astype(np.float32)
lidar2cam_rot = lidar2cam_array[:3, :3]
lidar2cam_trans = lidar2cam_array[:3, 3:4]

# 求逆得到 cam2lidar
camera2lidar = np.eye(4)
camera2lidar[:3, :3] = lidar2cam_rot.T
camera2lidar[:3, 3:4] = -1 * np.matmul(lidar2cam_rot.T, lidar2cam_trans)

# lidar2img = K @ [R | T]  (但 K 是 4x4 增广矩阵)
lidar2img.append(cam2img_array @ lidar2cam_array)
```

**Multi-Sweep 聚合**: 对于 nuScenes，BEVFusion 使用 9 帧历史点云 (`sweeps_num=9`)。历史帧通过 ego-motion 矩阵 `cur2prev` 补偿到当前帧 LiDAR 坐标系。

### 5.4 坐标系的完整追踪

训练时，点云和 GT Box 经历的坐标系变换：

```
1. 加载: World 坐标系（nuScenes 的 global frame）
   ↓ ego2global 逆变换
2. Ego 坐标系（当前帧 LiDAR 坐标系）← 这是 BEVFusion 所有操作的参考系
   ↓ LoadPointsFromMultiSweeps: 历史帧 → cur2prev 补偿
3. 增强前 LiDAR 坐标系（所有 sweep 已对齐到当前帧）
   ↓ BEVFusionGlobalRotScaleTrans (R→T→S 顺序)
   ↓ BEVFusionRandomFlip3D
   ↓ (累计到 lidar_aug_matrix)
4. 增强后 LiDAR 坐标系 → 输入模型
```

**相机侧**：
```
1. 加载: Camera 坐标系（每张图像在自己的相机坐标系中）
   ↓ ImageAug3D: resize/crop/flip/rotate
   ↓ (每张图像记录 img_aug_matrix)
2. 增强后 Camera 坐标系
   ↓ DepthLSSTransform: 将图像特征投影到 BEV 空间时，
   ↓ 通过 img_aug_matrix 逆变换 + lidar_aug_matrix 正变换补偿增强
3. BEV 坐标系（与增强后 LiDAR 坐标系对齐）
```

---

## 第6章: BEVFusion 模型主类完整剖析

### 6.1 为什么直接继承 Base3DDetector？

`BEVFusion` 继承自 `Base3DDetector` [`projects/BEVFusion/bevfusion/bevfusion.py:20`](projects/BEVFusion/bevfusion/bevfusion.py#L20)，而不是 `SingleStage3DDetector`。

原因：`SingleStage3DDetector` 假设单一特征提取路径（backbone → neck → bbox_head），但 BEVFusion 有 **两条独立的特征提取路径**（Camera Branch + LiDAR Branch），需要在 `extract_feat` 中自行编排。

```python
# SingleStage3DDetector.extract_feat (伪代码):
def extract_feat(self, data):
    x = self.pts_voxel_encoder(data)
    x = self.pts_middle_encoder(x)
    x = self.pts_backbone(x)
    x = self.pts_neck(x)
    return x

# BEVFusion.extract_feat (实际):
def extract_feat(self, batch_inputs_dict, batch_input_metas):
    features = []
    if imgs is not None:
        img_feature = self.extract_img_feat(...)  # Camera Branch
        features.append(img_feature)
    pts_feature = self.extract_pts_feat(...)       # LiDAR Branch
    features.append(pts_feature)
    if self.fusion_layer:
        x = self.fusion_layer(features)             # Fusion
    x = self.pts_backbone(x)                        # Shared
    x = self.pts_neck(x)
    return x
```

### 6.2 `__init__` 模块构建顺序

[`bevfusion.py:22-63`](projects/BEVFusion/bevfusion/bevfusion.py#L22-L63):

```python
def __init__(self, ..., pts_voxel_encoder, pts_middle_encoder,
             img_backbone, img_neck, view_transform,
             fusion_layer, pts_backbone, pts_neck, bbox_head, ...):
    # 0. 特殊处理: 从 data_preprocessor 中弹出 voxelize_cfg
    voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
    super().__init__(data_preprocessor=data_preprocessor)

    # 1. 体素化层（非 nn.Module，是自定义 CUDA op）
    self.pts_voxel_layer = Voxelization(**voxelize_cfg)

    # 2. 体素特征编码器 (HardSimpleVFE)
    self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

    # 3. 图像分支（可选）
    self.img_backbone = MODELS.build(img_backbone) if img_backbone else None
    self.img_neck = MODELS.build(img_neck) if img_neck else None
    self.view_transform = MODELS.build(view_transform) if view_transform else None

    # 4. 点云中间编码器 (BEVFusionSparseEncoder)
    self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

    # 5. 融合层（可选）
    self.fusion_layer = MODELS.build(fusion_layer) if fusion_layer else None

    # 6. BEV Backbone + Neck (SECOND + SECONDFPN)
    self.pts_backbone = MODELS.build(pts_backbone)
    self.pts_neck = MODELS.build(pts_neck)

    # 7. 检测头 (TransFusionHead)
    self.bbox_head = MODELS.build(bbox_head)
```

### 6.3 `extract_feat`——两条路径的汇合点

[`bevfusion.py:240-284`](projects/BEVFusion/bevfusion/bevfusion.py#L240-L284) 是理解整个模型的核心函数。

**Tensor 形状追踪表**：

| 步骤 | 操作 | Camera Branch | LiDAR-only |
|------|------|---------------|------------|
| 输入 | | `imgs`: (B, N_cam=6, 3, 256, 704) | `points`: List[Tensor] |
| | | `points`: List[Tensor] | |
| Step 1 | img_backbone (Swin-T) | (B×6, 3, H, W) → feat_maps: (B×6, C, H/8, W/8), ... | — |
| Step 2 | img_neck (GeneralizedLSSFPN) | (B×6, 256, 32, 88) × 3 尺度 | — |
| Step 3 | view_transform (DepthLSSTransform) | → **camera BEV**: (B, 80, 180, 180) | — |
| Step 4 | voxelize | — | points → (N_voxels, 5, 10) → mean → (N_voxels, 5) |
| Step 5 | pts_middle_encoder (SparseEncoder) | — | (N_voxels, 5) → **lidar BEV**: (B, 256, 180, 180) |
| Step 6 | fusion (ConvFuser) | concat cam_BEV + lidar_BEV: (B, 336, 180, 180) → **fused**: (B, 256, 180, 180) | 跳过 |
| Step 7 | pts_backbone (SECOND) | (B, 256, 180, 180) → [(B, 128, 180, 180), (B, 256, 90, 90)] |
| Step 8 | pts_neck (SECONDFPN) | → **final**: [(B, 256, 180, 180), (B, 256, 180, 180)] → concat → (B, 512, 180, 180) |

**关键细节** [bevfusion.py:246-270](projects/BEVFusion/bevfusion/bevfusion.py#L246-L270):

```python
def extract_feat(self, batch_inputs_dict, batch_input_metas):
    imgs = batch_inputs_dict.get('imgs', None)
    # ... 从 metainfo 中收集所有标定矩阵
    for i, meta in enumerate(batch_input_metas):
        lidar2image.append(meta['lidar2img'])         # (N_cam, 4, 4)
        camera_intrinsics.append(meta['cam2img'])      # (N_cam, 4, 4)
        camera2lidar.append(meta['cam2lidar'])         # (N_cam, 4, 4)
        img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
        lidar_aug_matrix.append(meta.get('lidar_aug_matrix', np.eye(4)))

    # Camera Branch
    if imgs is not None:
        img_feature = self.extract_img_feat(
            imgs, deepcopy(points),
            lidar2image, camera_intrinsics, camera2lidar,
            img_aug_matrix, lidar_aug_matrix, batch_input_metas)
        features.append(img_feature)

    # LiDAR Branch
    pts_feature = self.extract_pts_feat(batch_inputs_dict)
    features.append(pts_feature)

    # Fusion (如果有)
    x = self.fusion_layer(features) if self.fusion_layer else features[0]
```

### 6.4 `extract_img_feat`——Camera Branch

[`bevfusion.py:130-164`](projects/BEVFusion/bevfusion/bevfusion.py#L130-L164):

```python
def extract_img_feat(self, x, points, lidar2image, camera_intrinsics,
                     camera2lidar, img_aug_matrix, lidar_aug_matrix, img_metas):
    B, N, C, H, W = x.size()            # (B, 6, 3, 256, 704)
    x = x.view(B * N, C, H, W)          # → (B*6, 3, 256, 704)

    x = self.img_backbone(x)             # Swin-T → [(B*6,192,64,176), (B*6,384,32,88), (B*6,768,16,44)]
    x = self.img_neck(x)                 # GeneralizedLSSFPN → 3× (B*6, 256, 32, 88)

    if not isinstance(x, torch.Tensor):
        x = x[0]                         # 取最后一个尺度
    BN, C, H, W = x.size()
    x = x.view(B, BN//B, C, H, W)       # → (B, 6, 256, 32, 88)

    # 视角变换 (需要使用 float32 精度)
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        x = self.view_transform(x, points, ...)
    return x                             # → (B, 80, 180, 180)
```

### 6.5 `extract_pts_feat`——LiDAR Branch

[`bevfusion.py:166-173`](projects/BEVFusion/bevfusion/bevfusion.py#L166-L173):

```python
def extract_pts_feat(self, batch_inputs_dict):
    points = batch_inputs_dict['points']            # List of (N_i, 5)
    with torch.autocast('cuda', enabled=False):     # 禁用 AMP，保持 float32
        points = [point.float() for point in points]
        feats, coords, sizes = self.voxelize(points) # → (N_voxels,5), (N_voxels,4), (N_voxels,)
        batch_size = coords[-1, 0] + 1
    x = self.pts_middle_encoder(feats, coords, batch_size)
    return x                                         # → (B, 256, 180, 180)
```

### 6.6 `voxelize`——体素化细节

[`bevfusion.py:176-201`](projects/BEVFusion/bevfusion/bevfusion.py#L176-L201):

```python
@torch.no_grad()
def voxelize(self, points):
    feats, coords, sizes = [], [], []
    for k, res in enumerate(points):
        ret = self.pts_voxel_layer(res)              # 自定义 CUDA Voxelization
        f, c, n = ret                                # feat, coord, num_points
        feats.append(f)
        coords.append(F.pad(c, (1, 0), value=k))     # 在坐标前加上 batch_idx
        sizes.append(n)

    feats = torch.cat(feats, dim=0)
    coords = torch.cat(coords, dim=0)

    # voxelize_reduce=True: 体素内特征取均值
    if self.voxelize_reduce:
        feats = feats.sum(dim=1) / sizes.type_as(feats).view(-1, 1)
    return feats, coords, sizes
```

体素化参数 [bevfusion_lidar config:49-54](projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py#L49-L54):
- `max_num_points=10`: 每个体素最多保留 10 个点（超出的随机丢弃）
- `max_voxels=[120000, 160000]`: 训练时最多 120k 体素，测试时 160k
- `voxel_size=[0.075, 0.075, 0.2]`: 非常小的体素尺寸，高保真度
- `point_cloud_range=[-54, -54, -5, 54, 54, 3]`: 覆盖范围 108m × 108m × 8m

### 6.7 `loss` 和 `predict` 方法

```python
def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
    batch_input_metas = [item.metainfo for item in batch_data_samples]
    feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
    losses = dict()
    if self.with_bbox_head:
        bbox_loss = self.bbox_head.loss(feats, batch_data_samples)
    losses.update(bbox_loss)
    return losses

def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
    batch_input_metas = [item.metainfo for item in batch_data_samples]
    feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
    outputs = self.bbox_head.predict(feats, batch_input_metas)
    res = self.add_pred_to_datasample(batch_data_samples, outputs)
    return res
```

---

## 第7章: 视角变换——LSS/DepthLSS 核心

### 7.1 LSS (Lift-Splat-Shoot) 原理

视角变换（View Transform）是 BEVFusion 最核心的技术创新。它解决的问题是：

> **如何将 2D 图像特征转换为 3D BEV 空间特征？**

LSS 的三步范式：

```
Lift:   对每个像素预测 D 个深度 bin 的概率分布
        → 每个像素 "升起" 成 D 个 3D 点 (u, v) → (u, v, d_i)

Splat:  通过相机内外参将这些 3D 点投影到 BEV 网格
        → 具有相同 (x, y) 网格索引的点被累加（pooling）

Shoot:  BEV 网格上的特征经过后续 backbone 处理
```

### 7.2 `BaseViewTransform` 基类

[`depth_lss.py:20-183`](projects/BEVFusion/bevfusion/depth_lss.py#L20-L183) 是所有视角变换器的基类。

**核心参数**：

| 参数 | 含义 | LiDAR+Camera 配置值 |
|------|------|---------------------|
| `xbound` | BEV X 轴范围+步长 | `[-54.0, 54.0, 0.3]` → 360 个 grid |
| `ybound` | BEV Y 轴范围+步长 | `[-54.0, 54.0, 0.3]` → 360 个 grid |
| `zbound` | BEV Z 轴范围+步长 | `[-10.0, 10.0, 20.0]` → 1 个 grid (collapse) |
| `dbound` | 深度范围+步长 | `[1.0, 60.0, 0.5]` → 119 个深度 bin |
| `image_size` | 原图分辨率 | `[256, 704]` |
| `feature_size` | 特征图分辨率 | `[32, 88]` (= 256/8, 704/8) |

**`create_frustum`** [depth_lss.py:52-69](projects/BEVFusion/bevfusion/depth_lss.py#L52-L69):

创建一个 (D, fH, fW, 3) 的张量，表示每个特征图像素在每个深度 bin 处的相机坐标系 3D 坐标 (u, v, d)：

```python
def create_frustum(self):
    ds = torch.arange(*self.dbound).view(-1, 1, 1).expand(-1, fH, fW)  # (D, fH, fW)
    xs = torch.linspace(0, iW-1, fW).view(1, 1, fW).expand(D, fH, fW)
    ys = torch.linspace(0, iH-1, fH).view(1, fH, 1).expand(D, fH, fW)
    frustum = torch.stack((xs, ys, ds), -1)   # (D, fH, fW, 3)
    return nn.Parameter(frustum, requires_grad=False)
```

**`get_geometry`** [depth_lss.py:71-111](projects/BEVFusion/bevfusion/depth_lss.py#L71-L111):

将 frustum 从像素坐标系变换到 BEV（LiDAR）坐标系，经历 5 步：

```
1. undo post-transformation (img_aug_matrix 逆变换: 从增强后图像 → 原始图像)
   points = frustum - post_trans → inv(post_rots) @ points

2. pixel → normalized camera coords
   (u, v, d) → (u*d/fx, v*d/fy, d)

3. camera → lidar
   combine = camera2lidar_rots @ inv(intrins)
   points = combine @ points + camera2lidar_trans

4. apply lidar augmentation (如果训练时增强了点云，需要补偿)
   points = extra_rots @ points + extra_trans

5. 得到 BEV 空间中的 (x, y, z) 坐标 → 准备 bev_pool
```

**`bev_pool`** [depth_lss.py:116-148](projects/BEVFusion/bevfusion/depth_lss.py#L116-L148):

```python
def bev_pool(self, geom_feats, x):
    # x: (B, N, D, H, W, C) — 加权图像特征
    B, N, D, H, W, C = x.shape
    Nprime = B * N * D * H * W

    # 将 BEV 坐标离散化为网格索引
    geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
    geom_feats = geom_feats.view(Nprime, 3)

    # 为每个点添加 batch 索引
    batch_ix = torch.cat([torch.full([Nprime//B, 1], ix, ...) for ix in range(B)])
    geom_feats = torch.cat((geom_feats, batch_ix), 1)   # (N', 4): [x_idx, y_idx, z_idx, b]

    # 过滤出 BEV 网格内的点
    kept = ((geom_feats[:, 0] >= 0) & ... & (geom_feats[:, 2] < self.nx[2]))
    x = x[kept]
    geom_feats = geom_feats[kept]

    # CUDA 加速的 BEV pooling
    x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

    # 沿 Z 轴折叠 → (B, C*D, H, W)
    final = torch.cat(x.unbind(dim=2), 1)
    return final
```

### 7.3 `LSSTransform`——纯学习的深度估计

[`depth_lss.py:187-253`](projects/BEVFusion/bevfusion/depth_lss.py#L187-L253):

```python
class LSSTransform(BaseViewTransform):
    def __init__(self, ...):
        # depthnet: 1×1 Conv, 输入 C 通道, 输出 (D + C) 通道
        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, 1)

    def get_cam_feats(self, x):            # x: (B, N, C, fH, fW)
        B, N, C, fH, fW = x.shape
        x = x.view(B * N, C, fH, fW)
        x = self.depthnet(x)
        depth = x[:, :self.D].softmax(dim=1)  # (B*N, D, fH, fW) — 深度分布
        x = depth.unsqueeze(1) * x[:, self.D:self.D+self.C].unsqueeze(2)
        # → (B*N, C, D, fH, fW) — 每个深度 bin 的特征乘上对应概率
        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)    # → (B, N, D, fH, fW, C)
        return x
```

核心操作：`depthnet` 输出 D+C 个通道，前 D 个经过 softmax 作为深度概率，后 C 个作为上下文特征。然后做 outer product：**每个像素 (u,v) 的每个深度 bin 的特征 = 该像素的上下文特征 × 该深度 bin 的概率**。

### 7.4 `DepthLSSTransform`——带 LiDAR 监督的深度估计

[`depth_lss.py:334-427`](projects/BEVFusion/bevfusion/depth_lss.py#L334-L427):

与 `LSSTransform` 的关键区别：

1. **点云投影**: `BaseDepthTransform.forward()` [depth_lss.py:256-331](projects/BEVFusion/bevfusion/depth_lss.py#L256-L331) 将 LiDAR 点投影到每张图像上，生成稀疏深度图：
   ```
   LiDAR 点 → inverse lidar_aug → lidar2image → inverse img_aug → (u, v) 像素坐标 + depth 值
   ```

2. **深度先验**: `dtransform` 是一个小型 CNN (`1→8→32→64`)，将稀疏深度图编码为特征，拼接到图像特征上：
   ```python
   # in get_cam_feats [depth_lss.py:406-421]
   d = self.dtransform(d)           # (B*N, 1, H_img, W_img) → (B*N, 64, fH, fW)
   x = torch.cat([d, x], dim=1)    # 拼接深度特征和图像特征
   x = self.depthnet(x)             # 更深的 depthnet (3 层 + 1 层 vs 单层)
   ```

**LSSTransform vs DepthLSSTransform 对比**:

| | LSSTransform | DepthLSSTransform |
|---|---|---|
| depthnet 结构 | 单层 1×1 Conv | 3 层 Conv + 1 层 Conv |
| 深度图来源 | 纯学习 (softmax) | 学习 + LiDAR 投影作为先验 |
| dtransform | 无 | 1→8→32→64 编码 LiDAR 深度图 |
| 使用场景 | 纯视觉 3D 检测 | 多模态融合（BEVFusion LiDAR+Camera） |

### 7.5 Downsample 处理

两种 transform 都支持 `downsample` 参数 [depth_lss.py:212-233](projects/BEVFusion/bevfusion/depth_lss.py#L212-L233):

```python
if downsample == 2:
    self.downsample = nn.Sequential(
        Conv2d(out, out, 3, padding=1), BN, ReLU,    # 第一层
        Conv2d(out, out, 3, stride=2, padding=1),     # 下采样
        BN, ReLU,
        Conv2d(out, out, 3, padding=1), BN, ReLU)     # 第三层
```

LiDAR+Camera 配置使用 `downsample=2`，所以 BEV 输出从 360×360 降为 180×180，与 LiDAR BEV 特征分辨率匹配。

---

## 第8章: 点云特征提取

### 8.1 Voxelization——Hard vs Dynamic

BEVFusion 使用 **Hard Voxelization**:

```python
# 配置 [bevfusion_lidar config:49-54]
voxelize_cfg=dict(
    max_num_points=10,          # 每体素最多 10 个点
    point_cloud_range=[-54, -54, -5, 54, 54, 3],
    voxel_size=[0.075, 0.075, 0.2],
    max_voxels=[120000, 160000], # [训练, 测试]
    voxelize_reduce=True)       # 体素内取均值
```

**Hard Voxelization 流程**:
1. 对每个点，计算其体素索引 `(x_idx, y_idx, z_idx)`
2. 将点按体素索引分组
3. 每个体素最多保留 `max_num_points` 个点（超出则随机丢弃）
4. 如果 `voxelize_reduce=True`，对体素内点的特征取均值

**HardSimpleVFE**: 之后没有可学习的编码器，直接使用 5 维原始特征 (x, y, z, intensity, ring_index)。

### 8.2 `BEVFusionSparseEncoder`——3D 稀疏卷积

[`sparse_encoder.py`](projects/BEVFusion/bevfusion/sparse_encoder.py) 继承自 mmdet3d 的 `SparseEncoder`，唯一区别是 **sparse_shape 的维度顺序**：

- 父类 `SparseEncoder`: `(D, H, W)` ← 遵循 MMDetection3D 惯例
- `BEVFusionSparseEncoder`: `(H, W, D)` ← 适配 BEVFusion 自定义 voxelization 的输出

```python
class BEVFusionSparseEncoder(SparseEncoder):
    # sparse_shape = [1440, 1440, 41] ← (H, W, D) 顺序
    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        # coors: (N, 4) → [batch_idx, z_idx, y_idx, x_idx]
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)           # SubMConv3d, 5→16

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        out = self.conv_out(encode_features[-1])        # SparseConv3d, stride=(1,1,2)
        spatial_features = out.dense()                  # → (N, C, H, W, D)
        N, C, H, W, D = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3)  # → (N, C, D, H, W)
        spatial_features = spatial_features.view(N, C * D, H, W)   # 折叠 Z → (B, 256, 180, 180)
        return spatial_features
```

**Encoder 各 Stage 通道变化** (`encoder_channels=((16,16,32), (32,32,64), (64,64,128), (128,128))`):

```
Stage 0 (conv_input):    5 → 16  (SubMConv, 不改变空间维度)
Stage 1:                 16 → 16 → 16 → 32 (1 次 downsample, 空间 ÷2)
Stage 2:                 32 → 32 → 32 → 64 (1 次 downsample, 空间 ÷2)
Stage 3:                 64 → 64 → 128     (1 次 downsample, 空间 ÷2)
Stage 4:                 128 → 128         (无 downsample, 仅调整 padding)
conv_out:                128 → 256         (kernel=(1,1,3), stride=(1,1,2), Z 轴 ÷2)
```

空间下采样: 1440 → 720 → 360 → 180 (8× 下采样)

**最终输出**: `(B, 256, 180, 180)` — BEV 特征图，每个 (x, y) 格点对应 0.3m × 0.3m 的地面区域。

### 8.3 SECOND Backbone + SECONDFPN

对 BEV 特征图（本质上是 2D 特征图）应用标准 2D 卷积：

```python
# SECOND
pts_backbone=dict(
    type='SECOND',
    in_channels=256,
    out_channels=[128, 256],      # 两个 block 的输出通道
    layer_nums=[5, 5],            # 每个 block 5 层 Conv
    layer_strides=[1, 2],         # 第二个 block stride=2, 空间缩小一半
)

# SECONDFPN
pts_neck=dict(
    type='SECONDFPN',
    in_channels=[128, 256],       # 来自 SECOND 两个 block
    out_channels=[256, 256],      # FPN 上采样后每个层级都输出 256
    upsample_strides=[1, 2],      # stride=2 的特征上采样 2× 以匹配 stride=1
    upsample_cfg=dict(type='deconv', bias=False),
)
```

最终 neck 输出 concat → `(B, 512, 180, 180)`。

---

## 第9章: TransFusionHead——Transformer 检测头

### 9.1 设计理念

[`transfusion_head.py`](projects/BEVFusion/bevfusion/transfusion_head.py) 实现了 TransFusion 检测头：

1. **Heatmap-based query 初始化**（非 DETR 的随机 query）：从 BEV 特征热力图中选出 top-K 峰值位置作为初始 object query
2. **BEV 特征作为 cross-attention 的 key/value**：query 直接 attend 到 BEV 空间的空间特征
3. **迭代优化**：每层 decoder 输出更新后的中心位置作为下一层的 query_pos

### 9.2 `forward_single` 逐步追踪

[`transfusion_head.py:205-320`](projects/BEVFusion/bevfusion/transfusion_head.py#L205-L320)：

```
输入: feats (B, 512, 180, 180)
  │
  ├─ Step 1: shared_conv
  │   conv2d(512→128, 3×3) → fusion_feat: (B, 128, 180, 180)
  │   flatten → fusion_feat_flatten: (B, 128, 32400)
  │
  ├─ Step 2: heatmap_head
  │   Conv(128→128) + Conv(128→10) → dense_heatmap: (B, 10, 180, 180)
  │
  ├─ Step 3: Query 初始化
  │   dense_heatmap.sigmoid() → NMS → top-200 peaks
  │   query_feat: (B, 128, 200)  ← 从 fusion_feat_flatten 中 gather
  │   query_pos:  (B, 200, 2)    ← BEV 网格坐标
  │   query_labels: (B, 200)     ← 每个 query 的类别
  │
  ├─ Step 4: Category Embedding
  │   one_hot(query_labels) → Conv1d(10→128) → + query_feat
  │
  ├─ Step 5: Transformer Decoder (× num_decoder_layers)
  │   for each layer:
  │     query_feat = decoder[i](query_feat, key=fusion_feat_flatten,
  │                              query_pos, key_pos=bev_pos)
  │     res = prediction_heads[i](query_feat)
  │     query_pos = res['center']  ← 更新位置用于下一层
  │
  └─ Step 6: 输出
      ret_dicts: [{center, height, dim, rot, vel, heatmap}, ...] × num_layers
```

**query 初始化细节** [transfusion_head.py:228-275](projects/BEVFusion/bevfusion/transfusion_head.py#L228-L275):

```python
# 热力图 + 局部最大值 NMS
heatmap = dense_heatmap.detach().sigmoid()
local_max = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=0)
# 特殊处理: 行人和交通锥使用 kernel_size=1 (小物体更容易被 max pooling 抑制)
heatmap = heatmap * (heatmap == local_max)

# 选出 top-200 proposal
top_proposals = heatmap.view(B, -1).argsort(descending=True)[..., :200]
top_proposals_class = top_proposals // 32400   # 类别索引
top_proposals_index = top_proposals % 32400     # 空间索引

# 从 BEV 特征中 gather 对应位置的 query 特征
query_feat = fusion_feat_flatten.gather(index=..., dim=-1)
```

### 9.3 TransformerDecoderLayer 细节

[`transformer.py`](projects/BEVFusion/bevfusion/transformer.py) 继承自 MMDet 的 `DetrTransformerDecoderLayer`：

```python
class TransformerDecoderLayer(DetrTransformerDecoderLayer):
    def forward(self, query, key, value, query_pos, key_pos, ...):
        # Position Encoding (Learned, 通过 Conv1d)
        query_pos = self.self_posembed(query_pos)   # (B, Pq, 2) → (B, Pq, 128)
        key_pos = self.cross_posembed(key_pos)      # (B, 32400, 2) → (B, 32400, 128)

        # Self-Attention (query 之间)
        # 注意: value = query + query_pos (位置信息注入到 value 而非 key)
        query = self.self_attn(
            query=query, key=query, value=query + query_pos,
            query_pos=query_pos, key_pos=query_pos)

        # Cross-Attention (query attend BEV 特征)
        # 注意: value = key + key_pos (BEV 特征的每个位置加上位置编码)
        query = self.cross_attn(
            query=query, key=key, value=key + key_pos,
            query_pos=query_pos, key_pos=key_pos)

        # FFN
        query = self.ffn(query)
        return query
```

**关键差异** vs 标准 DETR Decoder：
- **`value = query + query_pos`** 而非 `value = query`
- **`value = key + key_pos`** 而非 `value = key`
- 位置信息被注入到 value 中，让 attention 可以直接利用位置感知的特征

### 9.4 BBoxCoder——编解码过程

[`utils.py:16-148`](projects/BEVFusion/bevfusion/utils.py#L16-L148) 定义 `TransFusionBBoxCoder`：

**Encode**（GT → 网络输出空间）:
```python
def encode(self, dst_boxes):          # dst_boxes: (M, 7) [x, y, z, w, l, h, yaw]
    targets[:, 0] = (x - pc_range[0]) / (out_size_factor * voxel_size[0])  # 归一化 BEV x
    targets[:, 1] = (y - pc_range[1]) / (out_size_factor * voxel_size[1])  # 归一化 BEV y
    targets[:, 2] = z + h * 0.5       # bottom center → gravity center
    targets[:, 3:6] = log(w, l, h)    # log 编码
    targets[:, 6:8] = sin(θ), cos(θ)  # 角度编码
    targets[:, 8:10] = vx, vy         # 速度 (如果 code_size=10)
```

**Decode**（网络输出 → 真实世界坐标）:
```python
def decode(self, heatmap, rot, dim, center, height, vel):
    # 网络输出 → 真实世界
    center_x = center_x * out_size_factor * voxel_size[0] + pc_range[0]
    center_y = center_y * out_size_factor * voxel_size[1] + pc_range[1]
    dim = dim.exp()
    height = height - dim[2] * 0.5    # gravity center → bottom center
    rot = atan2(rots, rotc)           # sin/cos → angle
```

### 9.5 标签分配与损失函数

**HungarianAssigner3D** [`utils.py:248-311`](projects/BEVFusion/bevfusion/utils.py#L248-L311):

匈牙利算法在预测框和 GT 框之间进行最优匹配，代价函数 = 分类代价 + 回归代价 + IoU 代价：

```python
cost = cls_cost + reg_cost + iou_cost
# cls_cost:   FocalLossCost(weight=0.15)
# reg_cost:   BBoxBEVL1Cost(weight=0.25) — 归一化 BEV 中心距离 L1
# iou_cost:   IoU3DCost(weight=0.25)      — 负 3D IoU

matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
```

**损失函数** [`transfusion_head.py:765-872`](projects/BEVFusion/bevfusion/transfusion_head.py#L765-L872):

| 损失 | 类型 | 作用对象 |
|------|------|---------|
| `loss_heatmap` | `GaussianFocalLoss` | 密集热力图 (180×180)，对 GT 中心位置生成高斯峰值 |
| `loss_cls` | `FocalLoss(sigmoid)` | 每个 proposal 的分类分数 |
| `loss_bbox` | `L1Loss` | 每个 proposal 的回归目标 (x,y,z,w,l,h,sinθ,cosθ,vx,vy) |

**code_weights**: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]`
- 前 8 维（位置+尺寸+旋转）权重 1.0
- 后 2 维（速度）权重 0.2，因为速度估计不确定性大

### 9.6 NMS 后处理

[`transfusion_head.py:343-496`](projects/BEVFusion/bevfusion/transfusion_head.py#L343-L496):

BEVFusion 使用 **Circle NMS** 而非标准 3D NMS：

- 车辆类 (0-7): `radius=-1` → 不做 NMS（原因：center-based 预测天然抑制重复）
- 行人类 (8): `radius=0.175m` → BEV 中心距离 < 0.175m 的框合并
- 交通锥类 (9): `radius=0.175m` → 同上

---

## 第10章: 融合模块与辅助组件

### 10.1 ConvFuser——最简洁的融合方式

[`transfusion_head.py:28-42`](projects/BEVFusion/bevfusion/transfusion_head.py#L28-L42):

```python
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))
```

对 LiDAR+Camera 配置：
- 输入: `[cam_BEV (B, 80, 180, 180), lidar_BEV (B, 256, 180, 180)]`
- Concat → `(B, 336, 180, 180)`
- Conv+BN+ReLU → `(B, 256, 180, 180)`

**为什么这么简单就有效？** 因为两个分支的特征已经在 **同一个 BEV 空间**对齐——不需要做 feature warping 或 soft association。这是 BEVFusion 的核心优势。

### 10.2 GeneralizedLSSFPN——图像特征金字塔

[`bevfusion_necks.py`](projects/BEVFusion/bevfusion/bevfusion_necks.py) 是一个简化的 FPN：

```python
class GeneralizedLSSFPN(BaseModule):
    def forward(self, inputs):             # [swin_out1, swin_out2, swin_out3]
        laterals = [inputs[i] for i in range(len(inputs))]

        # Top-down 路径: 从高语义层级往下融合
        for i in range(used_backbone_levels - 1, -1, -1):
            x = F.interpolate(laterals[i+1], size=laterals[i].shape[2:])
            laterals[i] = torch.cat([laterals[i], x], dim=1)  # 跨层 concat
            laterals[i] = self.lateral_convs[i](laterals[i])  # 1×1 Conv 压缩
            laterals[i] = self.fpn_convs[i](laterals[i])      # 3×3 Conv 平滑

        return tuple(outs)                 # 3 个尺度，每个 (B*N, 256, 32, 88)
```

输入来自 Swin-T 的 3 个 stage:
- Stage 1: `(B*N, 192, 64, 176)` → Swin-T 1/4 stride
- Stage 2: `(B*N, 384, 32, 88)` → Swin-T 1/8 stride
- Stage 3: `(B*N, 768, 16, 44)` → Swin-T 1/16 stride

经过 FPN，所有尺度统一到 `(32, 88)` 空间分辨率和 256 通道。

### 10.3 ImageAug3D——图像数据增强与矩阵追踪

[`transforms_3d.py:13-108`](projects/BEVFusion/bevfusion/transforms_3d.py#L13-L108) 执行图像增强，并**同步记录增强矩阵**：

```python
class ImageAug3D(BaseTransform):
    def transform(self, data):
        for img in imgs:
            # 1. resize → crop → flip → rotate
            # 2. 累积变换矩阵到 img_aug_matrix
            transform = torch.eye(4)
            transform[:2, :2] = rotation   # 2×2 旋转+缩放
            transform[:2, 3] = translation # 2×1 平移
            transforms.append(transform)

        data['img_aug_matrix'] = transforms  # (N_cam, 4, 4)
```

这个矩阵在 `get_geometry` 中被用于：
- `post_rots` / `post_trans`: 在将 frustum 投影到 3D 之前，先反做图像增强
- 确保增强后的像素坐标能正确映射回 3D 空间

### 10.4 BEVFusion 专有增强

**BEVFusionGlobalRotScaleTrans** [`transforms_3d.py:146-186`](projects/BEVFusion/bevfusion/transforms_3d.py#L146-L186):
- 与父类 `GlobalRotScaleTrans` 的区别：操作顺序改为 **R→T→S**（旋转→平移→缩放）
- 更新的 `lidar_aug_matrix` 会在 view transform 中被用于补偿 LiDAR 增强

**BEVFusionRandomFlip3D** [`transforms_3d.py:111-143`](projects/BEVFusion/bevfusion/transforms_3d.py#L111-L143):
- 翻转点云和 GT box
- 将翻转记录到 `lidar_aug_matrix`（而非直接修改坐标）

---

## 第11章: CUDA 算子实现

### 11.1 BEV Pooling CUDA 算子

位置: [`projects/BEVFusion/bevfusion/ops/bev_pool/bev_pool.py`](projects/BEVFusion/bevfusion/ops/bev_pool/bev_pool.py)

**问题**: 将 ~10^6 个 frustum 点 scatter 到 BEV 网格中，通过 PyTorch 原生操作非常慢（~200ms）。

**CUDA 优化**:
1. 将所有 frustum 点按 `(batch, x, y, z)` 网格索引排序
2. 使用 `cumsum` 在排序后的序列上计算前缀和
3. 相邻具有相同网格索引的点段，取末尾 cumsum 减开头 cumsum = 该网格的 sum

核心实现在 `bev_pool_cuda.cu` 中，实现了 `bev_pool_kernel` (forward) 和 `bev_pool_grad_kernel` (backward)。

加速比：~40× vs 原生 PyTorch scatter。

### 11.2 Voxelization CUDA 算子

位置: [`projects/BEVFusion/bevfusion/ops/voxel/voxelize.py`](projects/BEVFusion/bevfusion/ops/voxel/voxelize.py)

实现了两个 kernel：
- **`voxelization.cpp` + `voxelization_cuda.cu`**: Hard voxelization — 将点分配到体素并采样
- **`scatter_points.cpp` + `scatter_points_cuda.cu`**: Dynamic scatter — 将体素特征写回 dense 张量

编译脚本: [`projects/BEVFusion/setup.py`](projects/BEVFusion/setup.py)

```bash
cd projects/BEVFusion && python setup.py develop
```

编译后会生成两个 CUDA 扩展：
- `bev_pool_ext` → `from .ops import bev_pool`
- `voxel_layer` → `from .ops import Voxelization`

---

## 第12章: 完整代码追踪——端到端数据流

### 12.1 训练一个 Batch 的完整追踪

```
train_dataloader 产出一个 batch
  │
  ├─ data = {'inputs': {'points': List[(N_i, 5)], 'imgs': Tensor (B, 6, 3, 256, 704)},
  │          'data_samples': List[Det3DDataSample]}
  │
  └─ model.train_step(data)   ——————————— (来自 mmengine)
       │
       └─ model.forward(data['inputs'], data['data_samples'], mode='loss')
            │
            └─ BEVFusion.loss() [bevfusion.py:286]
                 │
                 ├─ batch_input_metas = [sample.metainfo for sample in data_samples]
                 │
                 ├─ feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
                 │    │
                 │    ├─ [CAMERA] extract_img_feat():
                 │    │     imgs (B,6,3,256,704) → view(B*6,...) → Swin-T
                 │    │     → [(B*6,192,64,176), (B*6,384,32,88), (B*6,768,16,44)]
                 │    │     → GeneralizedLSSFPN → [(B*6,256,32,88), (B*6,256,32,88), (B*6,256,32,88)]
                 │    │     → DepthLSSTransform.forward():
                 │    │         frustum: (D=119, fH=32, fW=88, 3)
                 │    │         get_geometry() → BEV coords
                 │    │         BaseDepthTransform.forward() → project LiDAR → sparse depth
                 │    │         get_cam_feats() → depthnet → outer product
                 │    │         bev_pool() → collapse Z
                 │    │         downsample(2) → (B, 80, 180, 180)
                 │    │
                 │    ├─ [LIDAR] extract_pts_feat():
                 │    │     voxelize(points) → (N_vox, 5), (N_vox, 4), (N_vox,)
                 │    │     BEVFusionSparseEncoder → (B, 256, 180, 180)
                 │    │
                 │    ├─ [FUSION] ConvFuser:
                 │    │     cat([cam (B,80,180,180), lidar (B,256,180,180)]) → (B,336,180,180)
                 │    │     Conv2d + BN + ReLU → (B, 256, 180, 180)
                 │    │
                 │    ├─ SECOND:
                 │    │     block0: (B,256,180,180) → (B,128,180,180)
                 │    │     block1: (B,128,180,180) → (B,256,90,90)
                 │    │
                 │    └─ SECONDFPN:
                 │          upsample (B,256,90,90) → (B,256,180,180)
                 │          concat → (B, 512, 180, 180)
                 │
                 └─ bbox_head.loss(feats, batch_data_samples):
                      │
                      ├─ forward(feats) → forward_single():
                      │    shared_conv: (B,512,180,180) → (B,128,180,180)
                      │    heatmap_head: (B,128,180,180) → (B,10,180,180)
                      │    top-200 query init
                      │    category embedding
                      │    transformer decoder ×1
                      │    prediction heads: center, height, dim, rot, vel, heatmap
                      │
                      ├─ get_targets():
                      │    bbox_coder.decode(preds) → decoded boxes in world coords
                      │    HungarianAssigner3D.assign() → 匹配结果
                      │    bbox_coder.encode(GT) → regression targets
                      │    create gaussian heatmap targets
                      │
                      └─ loss_by_feat():
                           loss_heatmap: GaussianFocalLoss(dense_heatmap, gt_heatmap)
                           loss_cls:     FocalLoss(proposal_cls, matched_labels)
                           loss_bbox:    L1Loss(proposal_reg, matched_targets) * code_weights
```

### 12.2 关键 Tensor 形状追踪表

| 阶段 | 操作 | 输入形状 | 输出形状 |
|------|------|---------|---------|
| 数据加载 | LiDAR | List[(N_i, 5)] | 打包为 batch |
| 数据加载 | Camera | List[(H, W, 3)] × 6 | (B, 6, 3, 256, 704) |
| Voxelization | HVFE | (N, 5) 点 | (N_vox, 5) 体素特征 |
| SparseEncoder | conv_input | (N_vox, 5) | (N_vox, 16) |
| SparseEncoder | Stage 1-4 | 稀疏 | (N_vox', 128) |
| SparseEncoder | conv_out | (N_vox', 128) | (N_vox', 256) |
| SparseEncoder | dense + fold Z | 稀疏 | **(B, 256, 180, 180)** `<-- LiDAR BEV` |
| Swin-T | Stage 1 | (B×6, 3, 256, 704) | (B×6, 96, 64, 176) |
| Swin-T | Stage 2 | → | (B×6, 192, 32, 88) |
| Swin-T | Stage 3 | → | (B×6, 384, 16, 44) |
| Swin-T | Stage 4 | → | (B×6, 768, 8, 22) |
| LSSFPN | FPN | 3 scales | (B×6, 256, 32, 88) × 3 |
| DepthLSS | frustum | — | (119, 32, 88, 3) |
| DepthLSS | dtransform | (B, 6, 1, 256, 704) | (B×6, 64, 32, 88) |
| DepthLSS | depthnet | (B×6, 320, 32, 88) | (B×6, 375, 32, 88) |
| DepthLSS | bev_pool | — | (B, 80, 360, 360) |
| DepthLSS | downsample | (B, 80, 360, 360) | **(B, 80, 180, 180)** `<-- Camera BEV` |
| ConvFuser | concat | cam:(80ch,180²) + lidar:(256ch,180²) | (B, 336, 180, 180) |
| ConvFuser | Conv+BN+ReLU | (B, 336, 180, 180) | **(B, 256, 180, 180)** `<-- Fused` |
| SECOND | block0 ×5 | (B, 256, 180, 180) | (B, 128, 180, 180) |
| SECOND | block1 ×5 | (B, 128, 180, 180) | (B, 256, 90, 90) |
| SECONDFPN | upsample + concat | [128ch@180², 256ch@90²→180²] | **(B, 512, 180, 180)** `<-- Final` |
| TransFusionHead | shared_conv | (B, 512, 180, 180) | (B, 128, 180, 180) |
| TransFusionHead | heatmap | (B, 128, 180, 180) | (B, 10, 180, 180) |
| TransFusionHead | queries | — | (B, 128, 200) feat + (B, 200, 2) pos |
| TransFusionHead | decoder | — | (B, 128, 200) |
| TransFusionHead | pred heads | (B, 128, 200) | center(2), height(1), dim(3), rot(2), vel(2) |
| Output | NMS decode | — | List[Det3DDataSample] with pred_instances_3d |

### 12.3 推理一张图的完整流程

```
BEVFusion.predict()
  │
  ├─ extract_feat() ← 同上
  │
  ├─ bbox_head.predict():
  │    forward_single() → preds_dict
  │    predict_by_feat():
  │      heatmap.sigmoid() × query_heatmap_score × one_hot(query_labels)
  │      bbox_coder.decode(score, rot, dim, center, height, vel)
  │      Circle NMS (per task group)
  │      filter by score & post_center_range
  │
  └─ add_pred_to_datasample():
       创建 InstanceData(bboxes_3d, scores_3d, labels_3d)
       填充到 data_sample.pred_instances_3d
```

---

## 第13章: 动手实践指南

### 13.1 任务 1: 修改 BEVFusion 配置

**目标**: 理解配置参数对模型行为的影响。

```bash
# 1. 复制一份配置
cp configs/bevfusion_lidar_...py configs/bevfusion_lidar_myexp.py

# 2. 修改 voxel_size 观察内存/精度 trade-off
# voxel_size = [0.1, 0.1, 0.2]  # 更粗的体素 → 更少体素 → 更少内存
# sparse_shape 相应调整: [1080, 1080, 41]

# 3. 添加 decoder layer
# num_decoder_layers=2  # 从 1 改为 2

# 4. 切换视角变换
# view_transform=dict(type='LSSTransform', ...)  # 从 DepthLSSTransform 改回纯 LSS

# 5. 运行验证
python tools/train.py configs/bevfusion_lidar_myexp.py
```

### 13.2 任务 2: 断点调试数据流

**关键断点位置**:

```python
# 1. 在 bevfusion.py 中
#   - extract_feat() line 246: 观察 camera/lidar 分支输入形状
#   - extract_img_feat() line 141: x.view(B*N,...) 前后的形状变化
#   - voxelize() line 201: return 前检查 feats/coords/sizes
#   - loss() line 289: 检查 losses dict

# 2. 在 depth_lss.py 中
#   - BaseViewTransform.forward() line 170: 检查 geom 的形状和值范围
#   - get_cam_feats() line 248: 检查 depth.softmax 的分布
#   - bev_pool() line 143: 检查 bev_pool 前后的形状

# 3. 在 transfusion_head.py 中
#   - forward_single() line 229: 检查 heatmap.max/min
#   - get_targets_single() line 608: 检查 assign_result
#   - loss_by_feat() line 783: 检查各 loss 的数值
```

**推荐的 hook 方式**:
```python
# 在 BEVFusion.__init__ 中添加
self.pts_middle_encoder.register_forward_hook(
    lambda m, inp, out: print(f"SparseEncoder output: {out.shape}"))
```

### 13.3 任务 3: 提取中间特征可视化

```python
import matplotlib.pyplot as plt
import torch

# 1. 可视化 BEV 特征（融合前后）
def visualize_bev(feat, title, save_path):
    """feat: (B, C, H, W)"""
    bev = feat[0].mean(dim=0).detach().cpu()  # 通道平均
    plt.figure(figsize=(10, 10))
    plt.imshow(bev, cmap='viridis')
    plt.title(title)
    plt.savefig(save_path)

# 2. 可视化深度分布
def visualize_depth(depth, save_path):
    """depth: (B*N, D, H, W) after softmax"""
    d = depth[0].mean(dim=(1,2)).detach().cpu()  # 空间平均
    plt.figure()
    plt.plot(d.numpy())
    plt.xlabel('Depth bin')
    plt.ylabel('Probability')
    plt.savefig(save_path)

# 3. 可视化热力图
def visualize_heatmap(heatmap, class_names, save_path):
    """heatmap: (B, num_classes, H, W)"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(heatmap[0, i].detach().cpu(), cmap='hot')
        ax.set_title(class_names[i])
    plt.savefig(save_path)
```

### 13.4 任务 4: 微调 BEVFusion（思路）

1. **准备自定义数据集**: 转换为 nuScenes 格式的 pkl 文件
2. **创建数据集配置**: 修改 `class_names`, `metainfo`, `data_root`
3. **修改点云范围**: 调整 `point_cloud_range`, `sparse_shape`, `grid_size`
4. **加载预训练权重**: `load_from = 'bevfusion_lidar.pth'`
5. **减小学习率**: `lr = 1e-5`（微调场景）

### 13.5 任务 5: 开发自定义模块

**示例：添加 Cross-Attention Fusion 替换 ConvFuser**:

```python
@MODELS.register_module()
class CrossAttnFuser(nn.Module):
    def __init__(self, cam_channels, lidar_channels, out_channels):
        super().__init__()
        self.cam_proj = nn.Conv2d(cam_channels, out_channels, 1)
        self.lidar_proj = nn.Conv2d(lidar_channels, out_channels, 1)
        self.attn = nn.MultiheadAttention(out_channels, 8, batch_first=False)

    def forward(self, inputs):
        cam_feat, lidar_feat = inputs  # (B,Ccam,H,W), (B,Clidar,H,W)
        B, C, H, W = lidar_feat.shape
        Q = self.lidar_proj(lidar_feat).view(B, C, -1).permute(2, 0, 1)
        KV = self.cam_proj(cam_feat).view(B, C, -1).permute(2, 0, 1)
        fused, _ = self.attn(Q, KV, KV)
        return fused.permute(1, 2, 0).view(B, -1, H, W)
```

然后在配置中替换:
```python
fusion_layer=dict(type='CrossAttnFuser', cam_channels=80, lidar_channels=256, out_channels=256)
```

---

## 第14章: 性能调试与优化

### 14.1 常见训练问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| Loss 不下降 | 学习率过小/过大 | 检查 LR、尝试 warmup |
| NaN Loss | 梯度爆炸、混合精度不稳定 | 减小 clip_grad、关闭 AMP |
| CUDA OOM | batch size 过大或 voxels 过多 | 减小 max_voxels (120000→80000)、减小 batch_size、启用 gradient checkpointing |
| BEV pooling 编译失败 | CUDA 版本不匹配 | 检查 nvcc 版本、尝试 `TORCH_CUDA_ARCH_LIST="7.5;8.0" python setup.py develop` |
| 多 GPU 训练死锁 | 数据 pipeline 不对称 | 检查 `persistent_workers=True`、确保 `drop_last=True` |

### 14.2 内存分析

```python
# 在训练脚本中添加
torch.cuda.reset_peak_memory_stats()

# ... 跑一个 batch ...

print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(torch.cuda.memory_summary())
```

BEVFusion LiDAR+Camera 的主要内存占用：
- SparseEncoder: ~2GB
- DepthLSSTransform (BEV pooling 中间结果): ~3GB
- TransFusionHead (cross-attention): ~2GB

**内存优化技巧**:
- AMP (Automatic Mixed Precision): `--amp` 标志，降低 ~40% 内存
- Gradient Checkpointing: 在 backbone 上启用，用时间换空间
- Reduce `max_voxels`: 120000 → 80000

### 14.3 速度分析

```python
# Profiling
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    losses = model(batch_inputs, batch_data_samples, mode='loss')
    losses['loss'].backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

BEVFusion 的主要速度瓶颈：
1. BEV Pooling CUDA (已充分优化，~20ms)
2. SparseConv 3D (受限于 spconv 库性能)
3. Swin-T forward (~50ms)
4. Transformer cross-attention (~30ms)

---

## 附录A: 关键代码文件索引

| 文件 | 作用 | 想理解 X 就看这里 |
|------|------|-------------------|
| [`mmdet3d/registry.py`](mmdet3d/registry.py) | 17 个 Registry 定义 | 框架如何通过 `type` 字符串装配组件 |
| [`mmdet3d/models/detectors/base.py`](mmdet3d/models/detectors/base.py) | `Base3DDetector` 基类 | 所有检测器的三个 forward 模式 |
| [`projects/BEVFusion/bevfusion/bevfusion.py`](projects/BEVFusion/bevfusion/bevfusion.py) | `BEVFusion` 主类 | 两条特征路径如何编排、融合、送入 backbone |
| [`projects/BEVFusion/bevfusion/depth_lss.py`](projects/BEVFusion/bevfusion/depth_lss.py) | `DepthLSSTransform` / `LSSTransform` | 图像→BEV 的视角变换，LSS 深度估计 |
| [`projects/BEVFusion/bevfusion/sparse_encoder.py`](projects/BEVFusion/bevfusion/sparse_encoder.py) | `BEVFusionSparseEncoder` | 3D 稀疏卷积如何将体素变为 BEV 特征图 |
| [`projects/BEVFusion/bevfusion/transfusion_head.py`](projects/BEVFusion/bevfusion/transfusion_head.py) | `TransFusionHead` + `ConvFuser` | 检测头完整逻辑：heatmap→query→transformer→prediction |
| [`projects/BEVFusion/bevfusion/transformer.py`](projects/BEVFusion/bevfusion/transformer.py) | `TransformerDecoderLayer` | 为什么 value=query+query_pos（与 DETR 不同） |
| [`projects/BEVFusion/bevfusion/bevfusion_necks.py`](projects/BEVFusion/bevfusion/bevfusion_necks.py) | `GeneralizedLSSFPN` | 图像多尺度特征融合 |
| [`projects/BEVFusion/bevfusion/transforms_3d.py`](projects/BEVFusion/bevfusion/transforms_3d.py) | 数据增强 | ImageAug3D 如何记录增强矩阵 |
| [`projects/BEVFusion/bevfusion/loading.py`](projects/BEVFusion/bevfusion/loading.py) | `BEVLoadMultiViewImageFromFiles` | 多视角图像加载 + 标定矩阵计算 |
| [`projects/BEVFusion/bevfusion/utils.py`](projects/BEVFusion/bevfusion/utils.py) | `TransFusionBBoxCoder`, `HungarianAssigner3D` | 编码/解码 + 匈牙利匹配 |
| [`projects/BEVFusion/configs/bevfusion_lidar_...py`](projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py) | LiDAR-only 配置 | 完整配置参数参考 |
| [`projects/BEVFusion/configs/bevfusion_lidar-cam_...py`](projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py) | LiDAR+Camera 配置 | 图像分支的配置增量 |
| [`mmdet3d/structures/det3d_data_sample.py`](mmdet3d/structures/det3d_data_sample.py) | `Det3DDataSample` | 数据在模型间如何传递 |
| [`mmdet3d/structures/points/`](mmdet3d/structures/points/) | `LiDARPoints`, `CameraPoints` | 点云结构体的 flip/transform 方法 |

---

## 附录B: 术语表

| 术语 | 英文 | 含义 |
|------|------|------|
| BEV | Bird's Eye View | 鸟瞰图，从正上方俯视场景 |
| LSS | Lift-Splat-Shoot | 视角变换范式：拉升(深度估计)→投影(BEV pooling)→投射(后续处理) |
| VFE | Voxel Feature Encoder | 体素特征编码器，将体素内点转为特征向量 |
| SparseEncoder | — | 3D 稀疏卷积编码器，只对非空体素计算 |
| FPN | Feature Pyramid Network | 特征金字塔，融合多尺度特征 |
| NDS | nuScenes Detection Score | nuScenes 综合评测指标 |
| mAP | mean Average Precision | 平均精度 |
| mAOE | mean Average Orientation Error | 平均朝向误差 |
| Frustum | — | 视锥体：相机投影的 3D 锥形区域 |
| DepthNet | — | 深度估计网络，预测每个像素在多个深度 bin 的概率 |
| SubMConv | Submanifold Sparse Convolution | 子流形稀疏卷积，不改变空间稀疏结构 |
| Ego-motion | — | 自动驾驶车自身运动，multi-sweep 聚合时需要补偿 |
| GT-Aug | Ground Truth Augmentation | 将其他场景的 GT 物体复制到当前场景进行增强 |
| CBGS | Class-Balanced Grouping Sampling | 类别平衡分组采样器 |

---

## 附录C: BEVFusion 配置参数速查

### BEVFusionSparseEncoder

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `in_channels` | 5 | 输入特征维度 (x,y,z,intensity,ring) |
| `sparse_shape` | `[1440, 1440, 41]` | 体素网格 (H, W, D) |
| `encoder_channels` | `((16,16,32), (32,32,64), (64,64,128), (128,128))` | 各 stage 输出通道 |
| `base_channels` | 16 | conv_input 输出通道 |
| `output_channels` | 128 | conv_out 输出通道 |

### DepthLSSTransform

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `in_channels` | 256 | 输入特征通道 |
| `out_channels` | 80 | 输出 BEV 通道 |
| `image_size` | `[256, 704]` | 原图尺寸 |
| `feature_size` | `[32, 88]` | 特征图尺寸 (image_size / 8) |
| `xbound` | `[-54, 54, 0.3]` | BEV X 轴 [min, max, stride] |
| `ybound` | `[-54, 54, 0.3]` | BEV Y 轴 |
| `zbound` | `[-10, 10, 20.0]` | BEV Z 轴 (20 表示跨越整个范围做 1 个 bin，即直接折叠) |
| `dbound` | `[1.0, 60.0, 0.5]` | 深度 bin [min, max, stride] → 119 个 bin |
| `downsample` | 2 | BEV 特征下采样倍率 (360→180) |

### TransFusionHead

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `num_proposals` | 200 | 每张图选择多少个 proposal |
| `num_decoder_layers` | 1 | Transformer decoder 层数 |
| `hidden_channel` | 128 | query feature 维度 |
| `in_channels` | 512 | 来自 neck 的输入通道 |
| `num_classes` | 10 | nuScenes 类别数 |
| `code_size` | 10 | BBox 编码: (x,y,z,w,l,h,sin,cos,vx,vy) |
| `code_weights` | `[1,1,1,1,1,1,1,1,0.2,0.2]` | 回归损失权重 |
| `out_size_factor` | 8 | 特征图相对于原始体素网格的下采样率 |

### Voxelization

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `max_num_points` | 10 | 每体素最多保留的点数 |
| `voxel_size` | `[0.075, 0.075, 0.2]` | 体素尺寸 (m) |
| `point_cloud_range` | `[-54, -54, -5, 54, 54, 3]` | 检测范围 (m) |
| `max_voxels` | `[120000, 160000]` | [训练, 测试] 最大体素数 |
| `voxelize_reduce` | True | 体素内点取均值 |

---

## 附录D: 常见命令速查

### 训练

```bash
# 单 GPU 训练
python tools/train.py <config_path>

# 多 GPU 训练
bash tools/dist_train.sh <config_path> <num_gpus>

# 从 checkpoint 恢复训练
python tools/train.py <config_path> --resume

# 加载预训练权重（不加载 optimizer/scheduler）
python tools/train.py <config_path> --cfg-options load_from=<checkpoint_path>
```

### 测试

```bash
# 单 GPU 测试
python tools/test.py <config_path> <checkpoint_path>

# 多 GPU 测试
bash tools/dist_test.sh <config_path> <checkpoint_path> <num_gpus>

# 可视化测试结果
python tools/test.py <config_path> <checkpoint_path> --show --show-dir <output_dir>
```

### 调试与分析

```bash
# 查看完整配置（展开所有 _base_ 继承后）
python tools/misc/print_config.py <config_path>

# 可视化一个 batch 的数据
python tools/analysis_tools/browse_dataset.py <config_path> --output-dir <output_dir>

# Benchmark 推理速度
python tools/analysis_tools/benchmark.py <config_path> <checkpoint_path>
```

### BEVFusion 专用

```bash
# 编译 CUDA 算子
cd projects/BEVFusion && python setup.py develop

# 运行多模态 Demo
python projects/BEVFusion/demo/multi_modality_demo.py \
    <config_path> <checkpoint_path> <point_cloud_file>
```

---

## 附录E: BEVFusion vs 其他方法对比

### nuScenes val 集性能对比

| 方法 | 模态 | NDS | mAP | 年份 |
|------|------|-----|-----|------|
| PointPillars | LiDAR | 53.7 | 40.1 | 2019 |
| CenterPoint | LiDAR | 67.3 | 60.3 | 2021 |
| **BEVFusion (LiDAR-only)** | LiDAR | **69.6** | **64.9** | 2022 |
| MVX-Net | LiDAR+Cam | 67.2 | 60.8 | 2020 |
| **BEVFusion (LiDAR+Camera)** | LiDAR+Cam | **71.4** | **68.6** | 2022 |

### 方法特点对比

| 方法 | 融合位置 | 视角变换 | 检测头 |
|------|---------|---------|--------|
| PointPillars | 无（纯 LiDAR） | — | Anchor3DHead |
| CenterPoint | 无（纯 LiDAR） | — | CenterHead (heatmap) |
| MVX-Net | Point-level | 投影到点云 | Anchor3DHead |
| BEVFusion | **BEV-level** | **LSS + BEV Pooling** | **TransFusionHead (Transformer)** |

### 选择建议

- **只有 LiDAR**: 用 CenterPoint 或 BEVFusion (LiDAR-only)
- **LiDAR + 多目相机**: 用 BEVFusion (LiDAR+Camera)
- **仅有多目相机**: 考虑 BEVDet / PETR 系列
- **推理速度优先**: PointPillars（最快）
- **精度优先**: BEVFusion 或未来更强的多模态方法

---

> **本文基于 MMDetection3D v1.4.0 编写，代码文件路径以该版本为准。**
> 学习建议：本文每个章节都有明确的代码位置引用（文件路径:行号），建议一边阅读一边在 IDE 中打开对应源文件对照查看。
