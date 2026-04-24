# BEVFusion 详细学习计划

## 1. 概述

### 1.1 什么是 BEVFusion
BEVFusion 是由 MIT-Han-Lab 提出的多任务多传感器融合框架，核心创新是在**统一的 Bird's-Eye View (BEV)** 表示空间中进行多模态特征融合。

**论文**: [BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation](https://arxiv.org/abs/2205.13542)

### 1.2 核心贡献
- 在 BEV 空间统一多模态特征（LiDAR 和 Camera），保留几何和语义信息
- 优化 BEV pooling，将 view transformation 延迟降低 40 倍以上
- 任务无关架构，支持 3D 检测和 BEV 地图分割等任务

---

## 2. 代码结构总览

```
projects/BEVFusion/
├── bevfusion/
│   ├── __init__.py              # 模块导出
│   ├── bevfusion.py             # 主模型 BEVFusion 类
│   ├── depth_lss.py              # View Transform (LSS-based)
│   ├── bevfusion_necks.py        # FPN neck (GeneralizedLSSFPN)
│   ├── transfusion_head.py       # 检测头 (TransFusionHead + ConvFuser)
│   ├── sparse_encoder.py         # 稀疏编码器
│   ├── transformer.py            # Transformer 解码器层
│   ├── loading.py                # 数据加载增强
│   ├── transforms_3d.py          # 3D 数据增强
│   └── utils.py                  # 工具类 (BBoxCoder, Assigner 等)
├── configs/
│   ├── bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py    # LiDAR-only 配置
│   └── bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py  # LiDAR+Camera 融合配置
└── demo/
    └── multi_modality_demo.py    # 演示脚本
```

---

## 3. 核心模块详解

### 3.1 主模型: `BEVFusion` (bevfusion.py)

**类继承**: `Base3DDetector`

**初始化参数**:
| 参数 | 说明 |
|------|------|
| `pts_voxel_encoder` | 点云体素编码器 (HardSimpleVFE) |
| `pts_middle_encoder` | 中间编码器 (BEVFusionSparseEncoder) |
| `fusion_layer` | 融合层 (ConvFuser) - 仅多模态版本 |
| `img_backbone` | 图像骨干网络 (SwinTransformer) |
| `pts_backbone` | 点云骨干网络 (SECOND) |
| `view_transform` | 视角变换 (DepthLSSTransform) |
| `img_neck` | 图像 neck (GeneralizedLSSFPN) |
| `pts_neck` | 点云 neck (SECONDFPN) |
| `bbox_head` | 检测头 (TransFusionHead) |

**前向流程** (`extract_feat`):
1. **图像分支**: `imgs` → `img_backbone` → `img_neck` → `view_transform` → BEV 特征
2. **点云分支**: `points` → `voxelize` → `pts_middle_encoder` → BEV 特征
3. **融合**: `fusion_layer` 合并两个分支 (仅多模态)
4. **检测**: `pts_backbone` → `pts_neck` → `bbox_head`

### 3.2 视角变换: `DepthLSSTransform` (depth_lss.py)

**继承链**: `BaseViewTransform` → `BaseDepthTransform` → `DepthLSSTransform`

**核心功能**: 将多视角相机图像转换到 BEV 空间

**关键步骤**:
1. **创建 frustum** (`create_frustum`): 沿深度方向创建采样点
2. **几何变换** (`get_geometry`):
   - 逆向相机数据增强 (post_rots, post_trans)
   - 相机坐标 → LiDAR 坐标变换 (camera2lidar)
   - 考虑内参 (intrinsics) 和外参
3. **深度估计** (`get_cam_feats`): 使用 depthnet 预测深度分布
4. **BEV Pooling** (`bev_pool`): 高效聚合 BEV 特征

**关键参数** (来自配置文件):
```python
xbound=[-54.0, 54.0, 0.3]   # BEV 空间 x 范围和分辨率
ybound=[-54.0, 54.0, 0.3]   # BEV 空间 y 范围和分辨率
zbound=[-10.0, 10.0, 20.0]  # BEV 空间 z 范围
dbound=[1.0, 60.0, 0.5]     # 深度范围和分辨率
```

### 3.3 融合层: `ConvFuser` (transfusion_head.py)

**功能**: 将 LiDAR BEV 特征和 Camera BEV 特征在通道维度拼接后卷积融合

```python
# 融合 [80, 256] → [256, 256]
nn.Conv2d(80 + 256, 256, 3, padding=1)
```

### 3.4 稀疏编码器: `BEVFusionSparseEncoder` (sparse_encoder.py)

**继承**: `SparseEncoder`

**特点**: 3D 卷积的 shape order 为 (H, W, D)，而非标准的 (D, H, W)

**网络结构**:
- `conv_input`: SubMConv3d
- 4 个 encoder layers (稀疏卷积)
- `conv_out`: SparseConv3d (1x1x3 kernel)
- 输出: `(N, C*D, H, W)` 展平的空间特征

### 3.5 检测头: `TransFusionHead` (transfusion_head.py)

**功能**: 基于 Transformer 的 3D 检测头

**结构**:
1. **共享卷积** (`shared_conv`): 处理融合特征
2. **HeatmapHead**: 生成类别热力图
3. **TransformerDecoderLayer**: 堆叠的解码器层
4. **回归头**: center, height, dim, rot, vel

**损失函数**:
- `loss_cls`: Focal Loss (分类)
- `loss_bbox`: L1 Loss (回归)
- `loss_heatmap`: GaussianFocalLoss (热力图)

### 3.6 工具类 (utils.py)

| 类名 | 功能 |
|------|------|
| `TransFusionBBoxCoder` | 边界框编码/解码 |
| `BBoxBEVL1Cost` | BEV 空间 L1 损失成本 |
| `IoU3DCost` | 3D IoU 损失成本 |
| `HeuristicAssigner3D` | 基于距离的启发式匹配 |
| `HungarianAssigner3D` | 基于匈牙利算法的匹配 |

---

## 4. 数据处理流程

### 4.1 训练 pipeline (多模态)

```python
# 数据加载
1. BEVLoadMultiViewImageFromFiles    # 加载多视角图像 + 内外参
2. LoadPointsFromFile                 # 加载点云
3. LoadPointsFromMultiSweeps          # 加载多帧点云 (9帧)
4. LoadAnnotations3D                   # 加载 3D 标注

# 增强
5. ImageAug3D                         # 图像增强 (resize, crop, flip, rotate)
6. BEVFusionGlobalRotScaleTrans      # 全局旋转、缩放、平移
7. BEVFusionRandomFlip3D              # 3D 随机翻转
8. PointsRangeFilter                  # 过滤范围外点云
9. ObjectRangeFilter                  # 过滤范围外目标
10. ObjectNameFilter                  # 按类别过滤

# 打包
11. Pack3DDetInputs                   # 打包为模型输入
```

### 4.2 关键数据元信息

```python
cam2img       # 相机内参矩阵
lidar2cam     # LiDAR 到相机外参
lidar2img     # LiDAR 到图像投影矩阵
cam2lidar     # 相机到 LiDAR 外参
img_aug_matrix    # 图像增强矩阵
lidar_aug_matrix  # LiDAR 增强矩阵
```

---

## 5. 配置文件解析

### 5.1 LiDAR-only 配置
```python
model = dict(
    type='BEVFusion',
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(type='BEVFusionSparseEncoder', ...),
    pts_backbone=dict(type='SECOND', ...),
    pts_neck=dict(type='SECONDFPN', ...),
    bbox_head=dict(type='TransFusionHead', ...)
)
# 注意: 没有 img_backbone, view_transform, fusion_layer
```

### 5.2 LiDAR + Camera 融合配置
```python
model = dict(
    type='BEVFusion',
    img_backbone=dict(type='mmdet.SwinTransformer', ...),   # 新增
    img_neck=dict(type='GeneralizedLSSFPN', ...),           # 新增
    view_transform=dict(type='DepthLSSTransform', ...),       # 新增
    fusion_layer=dict(type='ConvFuser', ...),                # 新增
    # ... 点云分支同上
)
```

---

## 6. 学习路线建议

### 阶段 1: 理解框架整体架构 (第 1-2 天)
1. 阅读论文 Abstract 和 Introduction
2. 查看项目 README 了解使用方法
3. 画出整体数据流图

### 阶段 2: 深入核心模块 (第 3-7 天)

**必读文件**:
1. `bevfusion.py` - 主模型流程
2. `depth_lss.py` - View Transform (核心创新)
3. `transfusion_head.py` - 检测头
4. `sparse_encoder.py` - 稀疏卷积

**理解要点**:
- BEV 空间是如何定义的 (xbound, ybound, zbound)
- 视角变换如何将 2D 图像特征 Lift 到 3D BEV 空间
- 多模态特征如何融合

### 阶段 3: 数据处理流程 (第 8-10 天)

**必读文件**:
1. `loading.py` - 数据加载
2. `transforms_3d.py` - 3D 增强
3. 配置文件中的 train_pipeline

**理解要点**:
- 相机内外参如何用于坐标变换
- 数据增强如何保证图像和点云一致性

### 阶段 4: 运行和调试 (第 11-14 天)

**实践任务**:
1. 编译 CUDA ops: `python projects/BEVFusion/setup.py develop`
2. 运行 demo: `python projects/BEVFusion/demo/multi_modality_demo.py ...`
3. 尝试训练: `bash tools/dist_train.py ...`

### 阶段 5: 扩展学习 (第 15+ 天)

**进阶主题**:
- 迁移学习 / 预训练模型使用
- 在其他数据集 (KITTI, Waymo) 上复现
- 模型部署 (ONNX, TensorRT)
- 扩展到分割任务

---

## 7. 关键参数配置参考

### 7.1 BEV 空间参数
```python
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]  # 检测范围
voxel_size = [0.075, 0.075, 0.2]                             # 体素大小
grid_size = [1440, 1440, 41]                                 # BEV 网格大小
```

### 7.2 训练参数
```python
batch_size = 4
optimizer = AdamW(lr=0.0002, weight_decay=0.01)
max_epochs = 20 (LiDAR-only) / 6 (LiDAR+Camera)
scheduler = CosineAnnealingLR
```

---

## 8. 参考资源

- **原始论文**: https://arxiv.org/abs/2205.13542
- **官方代码**: https://github.com/mit-han-lab/bevfusion
- **MMDetection3D 实现**: `projects/BEVFusion/`
- **数据集**: nuScenes (https://www.nuscenes.org/)

---

## 9. 验证清单

完成学习后，你应该能够:
- [ ] 画出 BEVFusion 的完整网络架构图
- [ ] 解释 BEV Pooling 的工作原理
- [ ] 说明 LiDAR 和 Camera 特征如何在 BEV 空间融合
- [ ] 理解数据 pipeline 中图像增强和坐标变换的对应关系
- [ ] 能够修改配置文件调整检测范围和模型参数
- [ ] 能够在本地运行训练和测试
