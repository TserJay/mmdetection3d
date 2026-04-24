# MMDetection3D 框架详细学习指南

## 1. 框架概述

### 1.1 什么是 MMDetection3D
MMDetection3D 是 OpenMMLab 项目的一部分，是一个基于 PyTorch 的 3D 目标检测开源工具箱，支持从点云、图像或多模态数据进行 3D 目标检测、分割等任务。

**官方仓库**: https://github.com/open-mmlab/mmdetection3d

### 1.2 支持的任务类型
| 任务类型 | 说明 | 示例模型 |
|---------|------|---------|
| 3D 物体检测 | 从点云/图像中检测 3D 边界框 | PointPillars, SECOND, PV-RCNN |
| 单目 3D 检测 | 从单目图像预测 3D 框 | FCOS3D, SMOKE, PGD |
| 多模态检测 | 融合 LiDAR 和相机信息 | BEVFusion, MVX-Net |
| 3D 分割 | 点云或 BEV 语义分割 | MinkUNet, SPVCNN |

### 1.3 依赖关系
```
mmdet3d 1.4.0
├── mmcv >= 2.0.0rc4, < 2.2.0
├── mmdet >= 3.0.0rc5, < 3.4.0
├── mmengine >= 0.8.0, < 1.0.0
└── torch >= 1.8.0
```

---

## 2. 目录结构

```
mmdetection3d/
├── mmdet3d/                    # 核心代码库
│   ├── apis/                   # 推理 API
│   │   ├── inference.py        # 推理函数
│   │   └── inferencers/        # 推理器类
│   ├── datasets/               # 数据集处理
│   │   ├── transforms/         # 数据增强
│   │   ├── kitti_dataset.py
│   │   ├── nuscenes_dataset.py
│   │   └── ...
│   ├── models/                 # 模型组件
│   │   ├── backbones/          # 骨干网络
│   │   ├── detectors/           # 检测器
│   │   ├── dense_heads/        # 检测头
│   │   ├── necks/              # FPN 等
│   │   ├── voxel_encoders/     # 体素编码器
│   │   ├── middle_encoders/    # 中间编码器
│   │   └── ...
│   ├── engine/                 # 训练引擎
│   │   └── hooks/              # 自定义 Hooks
│   ├── evaluation/             # 评估指标
│   │   └── metrics/            # 各类指标
│   ├── structures/             # 数据结构
│   │   ├── det3d_data_sample.py
│   │   └── points/             # 点云数据结构
│   └── visualization/          # 可视化
├── configs/                    # 配置文件
│   ├── _base_/                # 基础配置
│   ├── pointpillars/           # 各模型配置
│   ├── second/
│   ├── bevfusion/
│   └── ...
├── tools/                     # 训练测试工具
│   ├── train.py                # 训练入口
│   ├── test.py                 # 测试入口
│   └── ...
├── projects/                   # 项目示例 (如 BEVFusion)
└── tests/                      # 单元测试
```

---

## 3. 核心模块详解

### 3.1 模型模块 (`mmdet3d/models/`)

#### 3.1.1 检测器基类 (`detectors/base.py`)

**基类**: `Base3DDetector` (继承自 `mmdet.models.BaseDetector`)

**核心方法**:
| 方法 | 功能 |
|------|------|
| `forward()` | 统一前向入口，支持 loss/predict/tensor 三种模式 |
| `extract_feat()` | 提取特征（需子类实现） |
| `loss()` | 计算损失 |
| `predict()` | 预测并后处理 |
| `add_pred_to_datasample()` | 将预测结果转为 Det3DDataSample 格式 |

**forward 模式**:
```python
# 训练模式 - 返回损失
losses = model(inputs, data_samples, mode='loss')

# 推理模式 - 返回预测结果
predictions = model(inputs, data_samples, mode='predict')

# 张量模式 - 返回中间特征
features = model(inputs, data_samples, mode='tensor')
```

#### 3.1.2 检测器类型 (`detectors/`)

| 文件 | 检测器类型 | 代表模型 |
|------|----------|---------|
| `single_stage.py` | 单阶段检测器 | PointPillars, SECOND |
| `two_stage.py` | 两阶段检测器 | PV-RCNN, Part-A2 |
| `voxelnet.py` | VoxelNet 系列 | VoxelNet, Dynamic VoxelNet |
| `base.py` | 基类 | - |

#### 3.1.3 骨干网络 (`backbones/`)

| 骨干网络 | 文件 | 特点 |
|---------|------|------|
| SECOND | `second.py` | 稀疏卷积加速 |
| PointPillars | (在 voxel_encoders 中) | Pillar 化 |
| PointNet2 | `pointnet2_sa_msg.py` | Set Abstraction |
| DLA | `dla.py` | 深度学习架构 |
| SwinTransformer | (来自 mmdet) | 视觉 Transformer |

#### 3.1.4 体素编码器 (`voxel_encoders/`)

**核心类**: `VoxelEncoder`

| 编码器 | 文件 | 说明 |
|--------|------|------|
| `VoxelNet` | `voxel_encoder.py` | 原始 VoxelNet 编码器 |
| `PillarEncoder` | `pillar_encoder.py` | PointPillars 的 Pillar 编码 |
| `HardSimpleVFE` | `voxel_encoder.py` | 简单体素化 |
| `DynamicVoxelEncoder` | `voxel_encoder.py` | 动态体素化 |

#### 3.1.5 中间编码器 (`middle_encoders/`)

负责将体素特征编码为中间表示：

| 编码器 | 文件 | 说明 |
|--------|------|------|
| `PointPillarsScatter` | - | 将 Pillar 特征散射回 BEV |
| `SparseEncoder` | - | 稀疏卷积编码 |
| `HeightCompression` | - | 高度压缩 |

#### 3.1.6 检测头 (`dense_heads/`)

| 检测头 | 文件 | 特点 |
|--------|------|------|
| `CenterPointHead` | `centerpoint_head.py` | 基于中心点的检测 |
| `Anchor3DHead` | `anchor3d_head.py` | 基于锚框 |
| `FreeAnchor3DHead` | `free_anchor3d_head.py` | FreeAnchor |
| `TransFusionHead` | (projects) | Transformer 融合 |
| `FCOS3DHead` | `fcos_mono3d_head.py` | 单目 FCOS3D |
| `PGDHead` | `pgd_head.py` | 单目 PGD |

#### 3.1.7 Neck 网络 (`necks/`)

| Neck | 文件 | 说明 |
|------|------|------|
| `SECONDFPN` | - | SECOND 的 FPN |
| `GeneralizedLSSFPN` | (projects) | LSS 风格的 FPN |

---

### 3.2 数据集模块 (`mmdet3d/datasets/`)

#### 3.2.1 支持的数据集

| 数据集 | 文件 | 说明 |
|--------|------|------|
| nuScenes | `nuscenes_dataset.py` | 多模态自动驾驶数据集 |
| KITTI | `kitti_dataset.py` | 自动驾驶数据集 |
| Waymo | `waymo_dataset.py` | 大规模自动驾驶数据集 |
| ScanNet | `scannet_dataset.py` | 室内 3D 数据集 |
| SUNRGB-D | `sunrgbd_dataset.py` | 室内 RGB-D 数据集 |
| Lyft | `lyft_dataset.py` | 自动驾驶数据集 |
| S3DIS | `s3dis_dataset.py` | 室内分割数据集 |

#### 3.2.2 数据集基类 (`det3d_dataset.py`)

**核心类**: `Det3DDataset` 继承自 `mmengine.dataset.BaseDataset`

**关键方法**:
- `load_data_list()`: 加载数据列表
- `get_data_info()`: 获取单个数据信息
- `get_cat_ids()`: 获取类别 ID

#### 3.2.3 数据增强 (`datasets/transforms/`)

| 增强方法 | 文件 | 功能 |
|---------|------|------|
| `LoadPointsFromFile` | `loading.py` | 加载点云 |
| `LoadAnnotations3D` | `loading.py` | 加载 3D 标注 |
| `ObjectSample` | `transforms_3d.py` | 采样目标增强 |
| `GlobalRotScaleTrans` | `transforms_3d.py` | 全局旋转缩放 |
| `RandomFlip3D` | `transforms_3d.py` | 随机翻转 |
| `PointsRangeFilter` | `transforms_3d.py` | 过滤范围外点 |
| `ObjectRangeFilter` | `transforms_3d.py` | 过滤范围外目标 |
| `PhotoMetricDistortion3D` | `transforms_3d.py` | 光度畸变 |
| `Pack3DDetInputs` | `formating.py` | 打包为模型输入 |

---

### 3.3 数据结构 (`mmdet3d/structures/`)

#### 3.3.1 Det3DDataSample

统一的数据样本格式，存储标注和预测结果：

```python
# 属性
gt_instance_3d    # 3D 实例标注
gt_panoptic_seg_3d # 3D 全景分割
gt_semantic_seg_3d # 3D 语义分割
pred_instances_3d  # 预测的 3D 实例
pred_instance_3d  # 单帧预测
```

#### 3.3.2 3D 边界框结构

支持多种 3D 边界框格式：
- `LiDAR` - 中心点 + 尺寸 + 角度 (cx, cy, cz, w, h, l, rot)
- `Camera` - 相机坐标系的框
- `Depth` - 深度模式

---

### 3.4 推理 API (`mmdet3d/apis/`)

#### 3.4.1 核心函数

```python
# 初始化模型
from mmdet3d.apis import init_model
model = init_model(config_file, checkpoint_file, device='cuda:0')

# 单帧推理
from mmdet3d.apis import inference_detector
result = inference_detector(model, point_cloud_or_data_info)

# 多模态推理
from mmdet3d.apis import inference_multi_modality_detector
result = inference_multi_modality_detector(model, data_info)

# 单目推理
from mmdet3d.apis import inference_mono_3d_detector
result = inference_mono_3d_detector(model, image, calib)
```

#### 3.4.2 推理器类 (`inferencers/`)

| 推理器 | 功能 |
|--------|------|
| `LidarDet3DInferencer` | LiDAR 3D 检测推理 |
| `MonoDet3DInferencer` | 单目 3D 检测推理 |
| `MultiModalityDet3DInferencer` | 多模态 3D 检测推理 |
| `LidarSeg3DInferencer` | LiDAR 3D 分割推理 |

---

### 3.5 训练引擎 (`mmdet3d/engine/`)

#### 3.5.1 自定义 Hooks

| Hook | 文件 | 功能 |
|------|------|------|
| `DisableObjectSampleHook` | `hooks/` | 训练后期禁用目标采样 |
| `CheckpointHook` | (mmengine) | 定期保存模型 |

#### 3.5.2 优化器

支持标准 PyTorch 优化器 + 自定义：
- Adam, AdamW, SGD 等
- 通过 `OPTIMIZERS` registry 注册

---

### 3.6 评估指标 (`mmdet3d/evaluation/`)

| 指标 | 文件/类 | 说明 |
|------|---------|------|
| NuScenes 检测指标 | `metrics/nuscenes_metrics.py` | NDS, mAP 等 |
| KITTI 指标 | `metrics/kitti_metrics.py` | KITTI 风格 AP |
| 3D IoU | `bbox_3d/` | 3D 框的 IoU 计算 |

---

### 3.7 Registry 系统

MMDetection3D 使用 MMEngine 的 Registry 系统管理所有模块：

```python
from mmdet3d.registry import MODELS, DATASETS, TRANSFORMS

# 注册模型
@MODELS.register_module()
class MyModel(Base3DDetector):
    ...

# 构建模型
model = MODELS.build(dict(type='MyModel', ...))

# 注册数据集
@DATASETS.register_module()
class MyDataset(Det3DDataset):
    ...
```

**主要 Registry**:
| Registry | 管理内容 |
|----------|---------|
| `MODELS` | 模型组件 |
| `DATASETS` | 数据集 |
| `TRANSFORMS` | 数据变换 |
| `TASK_UTILS` | 任务工具（锚框生成器、编码器等） |
| `HOOKS` | 训练钩子 |
| `METRICS` | 评估指标 |
| `OPTIMIZERS` | 优化器 |

---

## 4. 配置文件系统

### 4.1 配置结构

典型配置文件包含以下部分：

```python
# 模型配置
model = dict(
    type='PointPillars',
    data_preprocessor=...,
    pts_voxel_encoder=...,
    pts_backbone=...,
    pts_neck=...,
    bbox_head=...,
)

# 数据集配置
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
train_pipeline = [...]
test_pipeline = [...]
train_dataloader = {...}
val_dataloader = {...}
test_dataloader = {...}

# 评估配置
val_evaluator = {...}
test_evaluator = {...}

# 训练配置
optim_wrapper = {...}
param_scheduler = {...}
train_cfg = {...}
val_cfg = {...}
test_cfg = {...}
```

### 4.2 配置继承

```python
# 基础配置
_base_ = ['../_base_/default_runtime.py']

# 覆盖配置
model = dict(...)
```

---

## 5. 训练和测试

### 5.1 训练命令

```bash
# 单卡训练
python tools/train.py configs/pointpillars/pointpillars_xxx.py

# 分布式训练 (8卡)
bash tools/dist_train.sh configs/pointpillars/pointpillars_xxx.py 8

# 带验证的训练
python tools/train.py configs/pointpillars/pointpillars_xxx.py --cfg-options train_cfg.val_interval=1
```

### 5.2 测试命令

```bash
# 单卡测试
python tools/test.py configs/pointpillars/pointpillars_xxx.py checkpoint.pth

# 分布式测试
bash tools/dist_test.sh configs/pointpillars/pointpillars_xxx.py checkpoint.pth 8
```

### 5.3 训练入口 (`tools/train.py`)

```python
# 核心流程
runner = Runner.from_cfg(cfg)
runner.train()
```

---

## 6. 核心流程图解

### 6.1 单阶段检测器流程 (如 PointPillars)

```
Input Point Cloud
      │
      ▼
┌─────────────────┐
│  VoxelEncoder   │  ← Pillarization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Backbone       │  ← SECOND (稀疏卷积)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Neck (FPN)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Dense Head     │  ← 预测中心、尺寸、角度
└────────┬────────┘
         │
         ▼
   Loss / NMS / Predictions
```

### 6.2 两阶段检测器流程 (如 PV-RCNN)

```
Input Point Cloud
      │
      ▼
┌─────────────────┐
│  VoxelEncoder   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Backbone       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Neck (FPN)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RPN Head       │  ← 第一阶段：生成提案
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ROI Head      │  ← 第二阶段：精炼框
└────────┬────────┘
         │
         ▼
   Loss / NMS / Predictions
```

### 6.3 多模态融合流程 (如 BEVFusion)

```
                    ┌──────────────┐
     Images ───────►│ Image Backbone│
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   View      │
                    │  Transform   │──► BEV Camera Features
                    │  (LSS)      │
                    └──────────────┘

     Points ───────►┌──────────────┐
                    │    Voxel     │
                    │   Encoder    │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Backbone   │──► BEV LiDAR Features
                    └──────────────┘

                           │
                    ┌──────▼───────┐
                    │ Fusion Layer │  ← Concat + Conv
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Detection   │
                    │    Head      │
                    └──────────────┘
```

---

## 7. 学习路线建议

### 阶段 1: 框架入门 (第 1-3 天)

**目标**: 了解框架架构，能运行基本训练/测试

1. **阅读资料**:
   - [README_zh-CN.md](README_zh-CN.md) - 框架简介
   - [BEVFusion学习计划.md](BEVFusion学习计划.md) - BEVFusion 详细学习

2. **环境搭建**:
   ```bash
   pip install -U openmim
   mim install mmcv-full
   pip install mmdet mmengine
   pip install mmdet3d
   ```

3. **运行 demo**:
   ```bash
   python demo/multi_modality_demo.py \
       demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin \
       demo/data/nuscenes/ \
       demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl \
       projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
       checkpoint.pth --score-thr 0.2 --show
   ```

### 阶段 2: 核心模块深入 (第 4-10 天)

**目标**: 掌握模型、数据、训练的核心组件

1. **模型组件** (必读):
   - `mmdet3d/models/detectors/base.py` - 检测器基类
   - `mmdet3d/models/detectors/single_stage.py` - 单阶段检测器
   - `mmdet3d/models/backbones/second.py` - SECOND 骨干
   - `mmdet3d/models/voxel_encoders/voxel_encoder.py` - 体素编码器

2. **数据处理** (必读):
   - `mmdet3d/datasets/det3d_dataset.py` - 数据集基类
   - `mmdet3d/datasets/transforms/loading.py` - 数据加载
   - `mmdet3d/datasets/transforms/transforms_3d.py` - 3D 增强

3. **理解配置文件**:
   - 阅读 `configs/_base_/` 下的基础配置
   - 对比不同模型的配置文件

### 阶段 3: 特定模型学习 (第 11-18 天)

**目标**: 深入理解具体模型实现

**推荐学习顺序**:
1. **PointPillars** - 最简单的单模态模型
   - 配置: `configs/pointpillars/`
   - 论文: PointPillars: Fast Encoders for Object Detection from Point Clouds

2. **SECOND** - 稀疏卷积加速
   - 配置: `configs/second/`
   - 论文: SECOND: Sparsely Embedded Convolutional Detection

3. **BEVFusion** - 多模态融合
   - 配置: `projects/BEVFusion/configs/`
   - 论文: BEVFusion: Multi-Task Multi-Sensor Fusion with Unified BEV Representation

4. **CenterPoint** - 基于中心的检测
   - 配置: `configs/centerpoint/`
   - 论文: CenterPoint: Center-based 3D Object Detection and Tracking

### 阶段 4: 高级主题 (第 19-25 天)

**目标**: 掌握训练技巧、部署、扩展开发

1. **训练技巧**:
   - 学习率调度 (`param_scheduler`)
   - 混合精度训练 (AMP)
   - 数据增强策略

2. **模型部署**:
   - `tools/deployment/` - 部署工具
   - ONNX 导出
   - TensorRT 加速

3. **自定义开发**:
   - 如何添加新数据集
   - 如何添加新模型
   - 如何添加新损失函数

---

## 8. 关键文件索引

### 8.1 必须掌握的文件

| 文件路径 | 内容 | 重要性 |
|---------|------|--------|
| `mmdet3d/models/detectors/base.py` | 3D 检测器基类 | ★★★★★ |
| `mmdet3d/registry.py` | Registry 系统 | ★★★★★ |
| `mmdet3d/structures/det3d_data_sample.py` | 数据样本结构 | ★★★★☆ |
| `mmdet3d/apis/inference.py` | 推理 API | ★★★★☆ |
| `mmdet3d/models/detectors/single_stage.py` | 单阶段检测器模板 | ★★★★☆ |

### 8.2 按任务参考的文件

**3D 检测任务**:
- `mmdet3d/models/dense_heads/` - 各检测头实现
- `mmdet3d/models/task_modules/` - 锚框、编码器等

**点云分割任务**:
- `mmdet3d/models/segmentors/` - 分割器
- `mmdet3d/models/decode_heads/` - 分割头

**数据处理**:
- `mmdet3d/datasets/transforms/` - 所有数据增强
- `mmdet3d/datasets/convert_utils.py` - 数据集转换

---

## 9. 常用命令速查

```bash
# 安装
pip install mmdet3d

# 训练
python tools/train.py <config> [--cfg-options ...]

# 测试
python tools/test.py <config> <checkpoint> [--out <result.pkl>]

# 模型导出
python tools/deployment/pytorch2onnx.py <config> <checkpoint> <output.onnx>

# 数据集准备
python tools/create_data.py <dataset_name> --data-root <path>
```

---

## 10. 参考资源

- [OpenMMLab 官网](https://openmmlab.com/)
- [MMDetection3D 文档](https://mmdetection3d.readthedocs.io/)
- [MMEngine 文档](https://mmengine.readthedocs.io/)
- [MMCV 文档](https://mmcv.readthedocs.io/)
- [nuScenes 数据集](https://www.nuscenes.org/)
- [KITTI 数据集](http://www.cvlibs.net/datasets/kitti/)

---

## 11. 验证清单

完成 MMDetection3D 学习后，你应该能够：

- [ ] 理解框架的整体架构和模块划分
- [ ] 掌握检测器、骨干网络、检测头的结构
- [ ] 理解数据 pipeline 的各个阶段
- [ ] 能够修改配置文件调整模型参数
- [ ] 能够运行训练、测试、推理流程
- [ ] 理解 Registry 系统的作用
- [ ] 能够添加新数据集或新模型组件
- [ ] 能够在 nuScenes/KITTI 等数据集上训练模型
