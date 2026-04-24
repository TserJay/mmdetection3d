# MMDetection3D 完整学习指南 (新手友好版)

## 目录
1. [3D 目标检测基础概念](#1-3d-目标检测基础概念)
2. [MMDetection3D 框架整体架构](#2-mmdetection3d-框架整体架构)
3. [核心模块详解](#3-核心模块详解)
4. [BEVFusion 模型深入解析](#4-bevfusion-模型深入解析)
5. [数据处理流程](#5-数据处理流程)
6. [配置文件详解](#6-配置文件详解)
7. [代码流程分析](#7-代码流程分析)
8. [学习路线与实践](#8-学习路线与实践)

---

## 1. 3D 目标检测基础概念

### 1.1 什么是 3D 目标检测？

3D 目标检测是计算机视觉任务，不仅要识别图像/点云中的物体类别，还要预测物体在 3D 空间中的位置和朝向。

**2D 检测 vs 3D 检测**:
```
2D 检测输出: [类别, bbox_2d(x1,y1,x2,y2), 置信度]
3D 检测输出: [类别, bbox_3d(中心点xyz, 长宽高, 旋转角), 置信度]
```

### 1.2 3D 检测常用数据格式

#### 1.2.1 点云 (Point Cloud)
点云是由大量 3D 点组成的数据，每个点包含 (x, y, z, intensity) 等信息。

```python
# 点云示例：一个包含 N 个点的点云
# 每行是一个点: [x, y, z, intensity]
# x, y, z 是空间坐标，intensity 是反射强度
points = [
    [1.2, 0.5, 0.8, 0.9],  # 点1
    [2.1, -0.3, 0.6, 0.7],  # 点2
    ...
    [0.5, 1.0, 0.2, 0.5],  # 点N
]
```

#### 1.2.2 图像 (Image)
相机拍摄的 RGB 图像，分辨率通常是 H x W x 3。

#### 1.2.3 相机内外参

**内参矩阵 (Camera Intrinsics)**:
```
┌ fx  0  cx ┐
│  0  fy  cy │
└  0   0   1 ┘
```
- fx, fy: 焦距
- cx, cy: 主点偏移

**外参矩阵 (Camera Extrinsics)**:
```
┌ R00 R01 R02 Tx ┐
│ R10 R11 R12 Ty │
│ R20 R21 R22 Tz │
└  0   0   0   1 ┘
```
- R: 旋转矩阵 (3x3)
- T: 平移向量 (3x1)

**坐标系转换**:
```
3D空间中的点 P_world --外参--> P_cam --内参--> P_image
P_cam = R * P_world + T
P_image = K * P_cam (K是内参)
```

### 1.3 常见数据集

| 数据集 | 传感器 | 特点 |
|--------|--------|------|
| **KITTI** | 64线激光雷达 + 相机 | 自动驾驶经典数据集 |
| **nuScenes** | 32线激光雷达 + 6个相机 | 多模态，360度覆盖 |
| **Waymo** | 64线激光雷达 + 5个相机 | 大规模，高精度 |
| **ScanNet** | RGB-D相机 | 室内场景 |
| **SUNRGB-D** | RGB-D相机 | 室内场景 |

### 1.4 常见 3D 检测方法分类

```
3D 目标检测方法
├── 基于点云的检测
│   ├── Point-based (PointNet系列)
│   │   ├── PointNet
│   │   ├── PointNet++
│   │   └── PointRCNN
│   ├── Voxel-based (体素化方法)
│   │   ├── VoxelNet
│   │   ├── SECOND
│   │   └── PointPillars
│   └── 混合方法
│       └── PV-RCNN
│
├── 基于图像的检测 (单目)
│   ├── FCOS3D
│   ├── SMOKE
│   └── PGD
│
└── 多模态融合检测
    ├── MVX-Net (点云+图像)
    ├── BEVFusion (BEV空间融合)
    └── PointPainting (点云+语义分割)
```

### 1.5 BEV (Bird's Eye View) 是什么？

BEV 是从正上方俯视场景的视角，类似于地图视图。

```
        前
         │
    ─────┼─────► 右
         │
        后

俯视图 (BEV):
    ┌─────────────────┐
    │       车        │  ← 从正上方看，这辆车是一个矩形框
    │   ┌───────┐    │
    │   │       │    │
    │   └───────┘    │
    │       ↓ 朝向(角度)
    └─────────────────┘

BEV 优点:
1. 消除遮挡问题
2. 方便多传感器融合
3. 适合做跟踪和motion planning
```

---

## 2. MMDetection3D 框架整体架构

### 2.1 框架定位

MMDetection3D 是 OpenMMLab 生态中专门处理 3D 感知任务的开源库，架构图：

```
OpenMMLab 生态
┌─────────────────────────────────────────────┐
│                 MMDetection3D               │
│              (3D 感知工具箱)                  │
├─────────────────────────────────────────────┤
│                                             │
│   依赖关系:                                  │
│   ┌──────────┐    ┌──────────┐             │
│   │   MMCV   │ ◄─ │  MMEngine │            │
│   │ (计算机视觉) │    │  (训练引擎) │            │
│   └──────────┘    └──────────┘             │
│        ▲                                  │
│        │                                   │
│   ┌──────────┐                             │
│   │  MMDET   │ ◄─ 2D 检测 (MMDetection)   │
│   └──────────┘                             │
│                                             │
└─────────────────────────────────────────────┘
```

### 2.2 核心设计思想

MMDetection3D 的核心设计基于 **Registry 模式** 和 **配置驱动**：

#### 2.2.1 Registry 模式
所有模块（模型、数据集、优化器等）都通过 Registry 管理，通过字符串名称动态创建。

```python
# 示例：注册一个新模型
from mmdet3d.registry import MODELS

@MODELS.register_module()  # 注册到 MODELS
class My3DDetector(Base3DDetector):
    def __init__(self, ...):
        ...

# 之后可以通过字符串创建实例
config = dict(type='My3DDetector', other_params=...)
model = MODELS.build(config)
```

#### 2.2.2 配置驱动
模型结构、训练策略等都通过配置文件定义，不需要修改代码。

```python
# config.py
model = dict(
    type='PointPillars',  # 模型类型
    voxel_size=[0.16, 0.16, 4],  # 体素大小
    point_cloud_range=[-54, -54, -5, 54, 54, 3],  # 范围
)
```

### 2.3 模块划分

```
MMDetection3D 代码结构
├── mmdet3d/
│   ├── apis/          # 推理接口 (init_model, inference_detector)
│   ├── datasets/       # 数据集 (NuScenes, KITTI, Waymo...)
│   ├── models/         # 模型组件 (核心！)
│   │   ├── backbones/      # 骨干网络
│   │   ├── detectors/       # 检测器
│   │   ├── dense_heads/     # 检测头
│   │   ├── necks/           # FPN等
│   │   ├── voxel_encoders/  # 体素编码器
│   │   └── ...
│   ├── structures/     # 数据结构 (Box3D, PointCloud...)
│   ├── evaluation/     # 评估指标
│   ├── engine/         # 训练相关 (hooks等)
│   └── visualization/  # 可视化
├── configs/           # 配置文件
├── tools/             # 训练测试工具
└── projects/          # 项目示例 (BEVFusion等)
```

---

## 3. 核心模块详解

### 3.1 检测器基类 (Base3DDetector)

**文件位置**: `mmdet3d/models/detectors/base.py`

这是所有 3D 检测器的基类，理解它就能理解整个框架。

```python
class Base3DDetector(BaseDetector):
    """
    3D 检测器的基类

    核心设计：
    1. 统一的 forward 接口，支持三种模式
    2. 数据预处理统一在 data_preprocessor 中
    """

    def forward(self, inputs, data_samples, mode='tensor'):
        """
        统一的 forward 接口

        Args:
            inputs: 输入数据 (点云或图像)
            data_samples: 数据样本 (包含标注信息)
            mode: 'loss' | 'predict' | 'tensor'

        Returns:
            根据 mode 不同返回不同内容
        """

        if mode == 'loss':
            # 训练模式：返回损失
            return self.loss(inputs, data_samples)

        elif mode == 'predict':
            # 推理模式：返回预测结果
            return self.predict(inputs, data_samples)

        elif mode == 'tensor':
            # 张量模式：返回中间特征
            return self._forward(inputs, data_samples)
```

### 3.2 单阶段检测器模板 (SingleStage3DDetector)

**文件位置**: `mmdet3d/models/detectors/single_stage.py`

```python
class SingleStage3DDetector(Base3DDetector):
    """
    单阶段 3D 检测器

    流程: 输入 → 特征提取 → 检测头 → 输出
         Input → extract_feat → bbox_head → Result

    组成:
    - backbone: 特征提取网络
    - neck: 特征金字塔网络 (可选)
    - bbox_head: 检测头，输出最终预测
    """

    def __init__(self,
                 backbone,      # 骨干网络配置
                 neck=None,     # neck配置
                 bbox_head=None, # 检测头配置
                 ...):
        super().__init__()

        # 使用 MODELS.build() 从配置创建模块
        self.backbone = MODELS.build(backbone)
        if neck:
            self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(bbox_head)

    def extract_feat(self, batch_inputs_dict):
        """
        提取特征的核心函数

        Args:
            batch_inputs_dict: 包含输入数据的字典
                - 'points': 点云列表
                - 'imgs': 图像 (可选)

        Returns:
            x: 提取的特征
        """
        points = batch_inputs_dict['points']
        # Stack points: [N, C] -> [B, N, C]
        x = torch.stack(points)

        # backbone: 提取特征
        x = self.backbone(x)

        # neck: 特征增强
        if self.with_neck:
            x = self.neck(x)

        return x

    def loss(self, batch_inputs, batch_data_samples):
        """计算损失 (训练时调用)"""
        x = self.extract_feat(batch_inputs)
        # bbox_head 计算损失
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses

    def predict(self, batch_inputs, batch_data_samples):
        """预测 (推理时调用)"""
        x = self.extract_feat(batch_inputs)
        # bbox_head 预测
        results = self.bbox_head.predict(x, batch_data_samples)
        # 转换为 Det3DDataSample 格式
        predictions = self.add_pred_to_datasample(batch_data_samples, results)
        return predictions
```

### 3.3 骨干网络 (Backbone)

骨干网络负责从原始输入中提取特征。

#### 3.3.1 SECOND 骨干 (稀疏卷积)

**文件**: `mmdet3d/models/backbones/second.py`

SECOND (Sparsely Embedded Convolutional Detection) 是常用的点云 3D 检测 backbone。

```python
class SECOND(BaseBackbone):
    """
    SECOND 骨干网络

    核心特点：使用稀疏卷积 (Sparse Convolution) 加速

    稀疏卷积原理：
    - 普通卷积对所有位置计算
    - 稀疏卷积只对有数据的位置计算
    - 对于点云这种稀疏数据非常高效
    """

    def forward(self, x):
        """
        输入: sparse_features (稀疏张量)
        输出: 多尺度特征列表

        网络结构:
        downsampling layers (稀疏卷积)
        ├── conv0: 16 channels
        ├── conv1: 32 channels
        ├── conv2: 64 channels
        └── conv3: 128 channels
        """
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        return outs  # 返回多层特征用于 FPN
```

#### 3.3.2 SwinTransformer (图像分支)

BEVFusion 中使用 SwinTransformer 作为图像的 backbone。

```python
# 来自 mmdet 的 SwinTransformer
img_backbone = dict(
    type='mmdet.SwinTransformer',
    embed_dims=96,
    depths=[2, 2, 6, 2],  # 4个stage的层数
    num_heads=[3, 6, 12, 24],  # 每个stage的head数
    window_size=7,  # 窗口大小
)
```

### 3.4 体素编码器 (VoxelEncoder)

体素编码器将点云转换为体素表示。

```python
class HardSimpleVFE(nn.Module):
    """
    简单体素编码器

    体素化过程:
    1. 将点云划分为 3D 网格 (体素)
    2. 每个体素内的点取平均/最大值

    例如: point_cloud_range=[-54, -54, -5, 54, 54, 3], voxel_size=[0.075, 0.075, 0.2]
    表示:
    - x方向: (-54, 54) 范围，每个体素 0.075 宽 -> 1440 个体素
    - y方向: (-54, 54) 范围，每个体素 0.075 高 -> 1440 个体素
    - z方向: (-5, 3) 范围，每个体素 0.2 高 -> 41 个体素
    """

    def forward(self, points):
        # points: [N, C] 点坐标+特征
        # 输出: voxel_features, voxel_coords, voxel_num_points
        ...
```

### 3.5 检测头 (DenseHead)

检测头负责从特征图预测目标的位置和类别。

#### 3.5.1 Anchor3DHead (基于锚框)

```python
class Anchor3DHead(BaseDenseHead):
    """
    基于锚框的 3D 检测头

    原理:
    1. 在每个位置放置多个不同尺寸/朝向的锚框
    2. 预测锚框与真实框的偏移量
    3. 分类判断是否有物体
    """
```

#### 3.5.2 CenterPointHead (基于中心点)

```python
class CenterPointHead(BaseDenseHead):
    """
    基于中心点的检测头 (CenterPoint 的实现)

    核心思想：不预测锚框，而是直接预测物体中心点

    输出 Heatmap:
    - heatmap: 每个类别的热力图，峰值位置是物体中心
    - offset: 中心点精确偏移
    - height: 物体高度
    - dim: 物体尺寸
    - rot: 旋转角度

    优点:
    - 不需要设计锚框数量和尺寸
    - 对密集物体效果好
    """
```

### 3.6 数据预处理 (DataPreprocessor)

```python
class Det3DDataPreprocessor:
    """
    3D 检测数据预处理器

    功能:
    1. 归一化图像
    2. 填充/对齐数据
    3. 转换为张量
    """

    def forward(self, data):
        # 图像: (mean, std) 归一化
        # 点云: 保持或简单变换
        ...
```

---

## 4. BEVFusion 模型深入解析

### 4.1 BEVFusion 整体架构

BEVFusion 的核心思想：**在统一的 BEV 空间融合 LiDAR 和 Camera 特征**

```
BEVFusion 流程图:

┌─────────────────────────────────────────────────────────────┐
│                         输入                                │
│   ┌─────────┐              ┌─────────┐                    │
│   │  图像   │              │  点云   │                    │
│   │ 6张相机 │              │ 点云    │                    │
│   └────┬────┘              └────┬────┘                    │
└────────┼───────────────────────┼───────────────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐      ┌─────────────────┐
│  Image Backbone │      │   Voxel Encoder │
│  (SwinTransformer)      │   + Middle Encoder │
│  + FPN           │      └────────┬────────┘
└────────┬─────────┘               │
         │                         ▼
         │              ┌─────────────────┐
         │              │  Backbone (SECOND)│
         │              └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐      ┌─────────────────┐
│  View Transform │      │   BEV 特征图    │
│   (DepthLSS)    │      │   (同尺寸)      │
│        │        │      │                 │
│        ▼        │      └────────┬────────┘
│  BEV 特征图     │               │
└────────┬─────────┘               │
         │                         │
         └───────────┬─────────────┘
                     │
                     ▼
           ┌─────────────────┐
           │  Fusion Layer   │  ← 融合: Concat + Conv
           │   (ConvFuser)   │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │    Neck (FPN)   │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │   检测头        │  ← TransFusionHead
           │  (预测bbox)     │
           └─────────────────┘
```

### 4.2 核心组件详解

#### 4.2.1 主模型 (BEVFusion)

**文件**: `projects/BEVFusion/bevfusion/bevfusion.py`

```python
@MODELS.register_module()
class BEVFusion(Base3DDetector):
    """
    BEVFusion 多模态 3D 检测器

    初始化参数:
    - pts_voxel_encoder: 点云体素编码器
    - pts_middle_encoder: 点云中间编码器
    - img_backbone: 图像骨干网络
    - img_neck: 图像 neck
    - view_transform: 视角变换 (2D图像 → BEV)
    - fusion_layer: 融合层
    - pts_backbone: 点云骨干网络
    - pts_neck: 点云 neck
    - bbox_head: 检测头
    """

    def extract_feat(self, batch_inputs_dict, batch_input_metas):
        """
        特征提取核心函数

        输入: batch_inputs_dict = {
            'imgs': 图像张量 [B, N, C, H, W]
            'points': 点云列表 [[N_i, C], ...]
        }

        返回: 融合后的 BEV 特征
        """

        # ======== 图像分支 ========
        if 'imgs' in batch_inputs_dict:
            imgs = batch_inputs_dict['imgs']

            # 提取相机参数
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for meta in batch_input_metas:
                lidar2image.append(meta['lidar2img'])      # LiDAR→图像投影
                camera_intrinsics.append(meta['cam2img'])   # 相机内参
                camera2lidar.append(meta['cam2lidar'])      # 相机→LiDAR外参
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(meta.get('lidar_aug_matrix', np.eye(4)))

            # 图像特征提取
            # 1. backbone: SwinTransformer 提取多尺度特征
            # 2. neck: FPN 融合多尺度特征
            # 3. view_transform: 将 2D 特征转换到 BEV 空间
            img_feature = self.extract_img_feat(
                imgs, points, lidar2image, camera_intrinsics,
                camera2lidar, img_aug_matrix, lidar_aug_matrix, batch_input_metas
            )

        # ======== 点云分支 ========
        # 1. voxelize: 点云体素化
        # 2. pts_middle_encoder: 稀疏卷积编码
        pts_feature = self.extract_pts_feat(batch_inputs_dict)

        # ======== 融合 ========
        if self.fusion_layer is not None:
            # 将图像 BEV 特征和点云 BEV 特征融合
            x = self.fusion_layer([img_feature, pts_feature])
        else:
            x = pts_feature

        # ======== 检测 ========
        # backbone: SECOND 进一步处理
        x = self.pts_backbone(x)
        # neck: FPN 特征融合
        x = self.pts_neck(x)

        return x
```

#### 4.2.2 视角变换 (View Transform - DepthLSSTransform)

**文件**: `projects/BEVFusion/bevfusion/depth_lss.py`

这是 BEVFusion 的核心创新点之一。

```python
class DepthLSSTransform(BaseViewTransform):
    """
    将多视角相机图像转换到 BEV 空间

    核心原理 (LSS: Lift, Splat, Shoot):

    1. Lift (提升):
       - 对于图像上每个像素，假设它在不同深度 d 上的概率
       - 生成一个 "frustum" (视锥体) 采样点

    2. Splat (泼洒):
       - 将这些采样点根据相机内外参投影到 BEV 空间
       - 使用 BEV Pooling 聚合特征

    3. Shoot (拍摄):
       - 生成 BEV 特征图
    """

    def forward(self, img, points, lidar2image, camera_intrinsics, ...):
        """
        Args:
            img: [B, N, C, H, W] 图像特征
            points: 点云 (用于深度监督)
            lidar2image: LiDAR到图像的投影矩阵

        Returns:
            x: [B, C, H, W] BEV 特征图
        """

        # 1. 获取相机参数
        intrins = camera_intrinsics[..., :3, :3]  # 内参
        post_rots = img_aug_matrix[..., :3, :3]   # 图像增强旋转
        camera2lidar_rots = camera2lidar[..., :3, :3]  # 外参旋转
        camera2lidar_trans = camera2lidar[..., :3, 3]  # 外参平移

        # 2. 创建 frustum (视锥体采样点)
        # 对于图像上每个 (u, v) 像素，在深度方向创建多个采样点
        frustum = self.create_frustum()
        # frustum shape: [D, H, W, 3] (D=深度层数, H,W=特征图尺寸)

        # 3. 几何变换: 图像坐标 → 相机坐标 → LiDAR坐标 → BEV坐标
        geom = self.get_geometry(
            camera2lidar_rots, camera2lidar_trans, intrins,
            post_rots, post_trans,
            extra_rots=lidar_aug_matrix[..., :3, :3],
            extra_trans=lidar_aug_matrix[..., :3, 3],
        )
        # geom shape: [B, N, D, H, W, 3] 每个采样点对应的 BEV 坐标

        # 4. 提取图像特征 + 预测深度分布
        x = self.get_cam_feats(img)
        # x shape: [B, N, D, C, H, W] 每像素的深度分布 × 特征

        # 5. BEV Pooling: 将 frustum 特征聚合到 BEV 网格
        x = self.bev_pool(geom, x)
        # x shape: [B, C, H_bev, W_bev]

        return x
```

#### 4.2.3 BEV Pooling 详解

```python
def bev_pool(self, geom_feats, x):
    """
    将图像特征 pooling 到 BEV 空间

    Args:
        geom_feats: [B*N*D*H*W, 3] 每点对应的 BEV 坐标
        x: [B*N*D*H*W, C] 每点的图像特征

    过程:
    1. 将 BEV 坐标转换为体素索引
    2. 过滤掉超出范围的点
    3. 使用 GPU 加速的 bev_pool 算子聚合
    """

    B, N, D, H, W, C = x.shape
    Nprime = B * N * D * H * W

    # 展平
    x = x.reshape(Nprime, C)
    geom_feats = geom_feats.view(Nprime, 3)

    # 坐标归一化为体素索引
    # bx: 体素起点, dx: 体素大小
    geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()

    # 过滤超出范围的点
    kept = (
        (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) &
        (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) &
        (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
    )

    # 调用 CUDA 算子进行高效 pooling
    x = bev_pool(x[kept], geom_feats[kept], B, self.nx[2], self.nx[0], self.nx[1])

    # 压缩 Z 维度
    final = torch.cat(x.unbind(dim=2), 1)

    return final
```

#### 4.2.4 融合层 (ConvFuser)

```python
class ConvFuser(nn.Sequential):
    """
    融合 LiDAR BEV 特征和 Camera BEV 特征

    融合方式: 通道拼接 + 卷积
    """

    def forward(self, inputs):
        """
        Args:
            inputs: [img_feature, pts_feature]
                - img_feature: [B, C_img, H, W]
                - pts_feature: [B, C_pts, H, W]

        Returns:
            fused: [B, C_out, H, W]
        """
        # 在通道维度拼接
        x = torch.cat(inputs, dim=1)  # [B, C_img + C_pts, H, W]

        # 卷积融合
        return super().forward(x)
```

#### 4.2.5 检测头 (TransFusionHead)

```python
class TransFusionHead(nn.Module):
    """
    Transformer 风格的检测头

    结构:
    1. 共享卷积层处理特征
    2. HeatmapHead 生成热力图
    3. TransformerDecoderLayer 精炼预测
    4. 回归头预测 (center, size, rot, vel)
    """

    def forward(self, x, batch_data_samples):
        # 处理特征
        x = self.shared_conv(x)

        # 生成 heatmap 和初始预测
        heatmap = self.heatmap_head(x)

        # Transformer 解码器
        for layer in self.decoder_layers:
            x = layer(x, heatmap)

        # 回归
        preds = self.prediction(feats)

        return preds
```

---

## 5. 数据处理流程

### 5.1 完整训练 Pipeline

以 nuScenes 数据集为例，训练流程如下：

```python
train_pipeline = [
    # ======== 1. 数据加载 ========
    dict(type='BEVLoadMultiViewImageFromFiles', ...),  # 加载多视角图像
    dict(type='LoadPointsFromFile', ...),              # 加载点云
    dict(type='LoadPointsFromMultiSweeps', ...),        # 加载历史帧
    dict(type='LoadAnnotations3D', ...),                # 加载 3D 标注

    # ======== 2. 数据增强 ========
    dict(type='ImageAug3D', ...),                       # 图像增强
    dict(type='BEVFusionGlobalRotScaleTrans', ...),     # 全局旋转缩放
    dict(type='BEVFusionRandomFlip3D', ...),            # 随机翻转

    # ======== 3. 过滤 ========
    dict(type='PointsRangeFilter', ...),                 # 过滤范围外点云
    dict(type='ObjectRangeFilter', ...),                 # 过滤范围外目标
    dict(type='ObjectNameFilter', ...),                  # 按类别过滤

    # ======== 4. 打包 ========
    dict(type='Pack3DDetInputs', ...),                  # 打包为模型输入
]
```

### 5.2 数据加载详解

#### 5.2.1 加载多视角图像

```python
class BEVLoadMultiViewImageFromFiles:
    """
    加载多视角相机图像

    nuScenes 有 6 个相机:
    - CAM_FRONT: 前视
    - CAM_FRONT_LEFT: 左前
    - CAM_FRONT_RIGHT: 右前
    - CAM_BACK: 后视
    - CAM_BACK_LEFT: 左后
    - CAM_BACK_RIGHT: 右后
    """

    def transform(self, results):
        # 1. 加载 6 张图像
        imgs = [load_image(path) for path in results['img_filename']]

        # 2. 计算并存储相机参数
        # cam2img: 相机内参
        # lidar2cam: LiDAR到相机外参
        # lidar2img = cam2img @ lidar2cam
        # cam2lidar = inverse(lidar2cam)

        # 3. 返回处理后的数据
        return results
```

#### 5.2.2 加载点云

```python
class LoadPointsFromFile:
    """
    从文件加载点云

    nuScenes 点云格式:
    [x, y, z, intensity, ring_index]
    - x, y, z: 坐标
    - intensity: 反射强度
    - ring_index: 激光线束索引
    """

    def transform(self, results):
        points = load_pcd(results['lidar_path'])

        # 坐标系通常是 LiDAR 坐标系
        results['points'] = points

        return results
```

### 5.3 数据增强详解

#### 5.3.1 图像增强 (ImageAug3D)

```python
class ImageAug3D:
    """
    3D 场景的图像增强

    需要保证图像增强和点云增强一致！

    增强参数:
    - final_dim: 输出图像尺寸
    - resize_lim: 缩放范围
    - rot_lim: 旋转角度范围
    - rand_flip: 是否随机翻转
    """

    def transform(self, data):
        # 1. 对每张图像独立采样增强参数
        resize, crop, flip, rotate = self.sample_augmentation()

        # 2. 应用图像变换
        for img in data['imgs']:
            img = resize(img, resize)
            img = crop(img, crop)
            if flip:
                img = horizontal_flip(img)
            img = rotate(img, rotate)

        # 3. 记录变换矩阵 (用于后续坐标转换)
        transform_matrix = self.compute_transform_matrix(...)
        data['img_aug_matrix'] = transform_matrix

        return data
```

#### 5.3.2 3D 增强

```python
class BEVFusionGlobalRotScaleTrans:
    """
    全局 3D 旋转、缩放、平移

    对点云和 3D 框同时应用相同的变换
    """

    def transform(self, input_dict):
        # 1. 旋转
        rotation = sample_rotation()
        points = rotate_points(points, rotation)
        bboxes = rotate_bboxes(bboxes, rotation)

        # 2. 缩放
        scale = sample_scale()
        points = scale_points(points, scale)
        bboxes = scale_bboxes(bboxes, scale)

        # 3. 平移
        translation = sample_translation()
        points = translate_points(points, translation)
        bboxes = translate_bboxes(bboxes, translation)

        # 4. 记录变换矩阵
        lidar_aug_matrix = compute_transform_matrix(rotation, scale, translation)
        input_dict['lidar_aug_matrix'] = lidar_aug_matrix

        return input_dict
```

### 5.4 坐标系统变换

```python
# nuScenes 坐标系统
# ================

# LiDAR 坐标系: 原点在 LiDAR 传感器中心
# - x: 前
# - y: 左
# - z: 上

# 相机坐标系: 原点在相机光心
# - x: 右
# - y: 下
# - z: 前

# 变换关系:
# P_lidar = lidar_T_cam @ P_cam
# P_cam = cam_T_lidar @ P_lidar
# cam_T_lidar = inverse(lidar_T_cam)

# 图像投影:
# P_image = cam_intrinsics @ cam_T_lidar @ P_lidar
#         = lidar2img @ P_lidar
```

---

## 6. 配置文件详解

### 6.1 配置文件结构

```python
# bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py

# ======== 1. 基础配置 ========
_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# ======== 2. 数据集配置 ========
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)

# ======== 3. 模型配置 ========
model = dict(
    type='BEVFusion',

    # 数据预处理器
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],  # ImageNet mean
        std=[58.395, 57.12, 57.375],       # ImageNet std
    ),

    # ======== 图像分支 ========
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,              # 嵌入维度
        depths=[2, 2, 6, 2],       # 每阶段层数
        num_heads=[3, 6, 12, 24],   # 每阶段注意力头数
        ...
    ),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],  # Swin 多尺度输出通道
        out_channels=256,             # 输出通道
        ...
    ),
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,              # 输入通道
        out_channels=80,              # 输出 BEV 通道
        # BEV 空间定义
        xbound=[-54.0, 54.0, 0.3],   # x: [-54, 54], 步长 0.3
        ybound=[-54.0, 54.0, 0.3],   # y: [-54, 54], 步长 0.3
        zbound=[-10.0, 10.0, 20.0],  # z: [-10, 10]
        dbound=[1.0, 60.0, 0.5],     # 深度: [1, 60], 步长 0.5
        downsample=2,
    ),

    # ======== 点云分支 ========
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=5,
        sparse_shape=[1440, 1440, 41],  # 体素网格形状
        ...
    ),
    pts_backbone=dict(type='SECOND', ...),
    pts_neck=dict(type='SECONDFPN', ...),

    # ======== 融合 & 检测 ========
    fusion_layer=dict(type='ConvFuser', in_channels=[80, 256], out_channels=256),
    bbox_head=dict(type='TransFusionHead', ...),
)

# ======== 4. 训练 Pipeline ========
train_pipeline = [...]

# ======== 5. 数据加载器 ========
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    ...
)

# ======== 6. 优化器 ========
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# ======== 7. 学习率调度 ========
param_scheduler = [
    dict(type='LinearLR', start_factor=0.333, ...),
    dict(type='CosineAnnealingLR', T_max=6, ...),
]

# ======== 8. 训练设置 ========
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)
```

### 6.2 关键参数解释

| 参数 | 含义 | 示例值 |
|------|------|--------|
| `point_cloud_range` | 3D 检测范围 (xmin, ymin, zmin, xmax, ymax, zmax) | [-54, -54, -5, 54, 54, 3] |
| `voxel_size` | 体素大小 (x, y, z) | [0.075, 0.075, 0.2] |
| `grid_size` | 体素网格数量 (x, y, z) | [1440, 1440, 41] |
| `xbound/ybound` | BEV 空间范围 [min, max, step] | [-54, 54, 0.3] |
| `dbound` | 深度范围 [min, max, step] | [1, 60, 0.5] |

---

## 7. 代码流程分析

### 7.1 训练流程

```
训练命令
    │
    ▼
tools/train.py
    │
    ├── Runner.from_cfg(cfg)  # 创建 Runner
    │
    ▼
runner.train()
    │
    ├── for epoch in range(max_epochs):
    │       │
    │       ▼
    │   TrainLoop
    │       │
    │       ├── for batch in dataloader:
    │       │       │
    │       │       ▼
    │       │   model.forward(inputs, data_samples, mode='loss')
    │       │       │
    │       │       ├── data_preprocessor.forward()  # 数据预处理
    │       │       │
    │       │       ├── model.extract_feat()          # 特征提取
    │       │       │       ├── img: backbone → neck → view_transform
    │       │       │       └── pts: voxelize → middle_encoder → backbone
    │       │       │
    │       │       ├── fusion_layer()                # 特征融合
    │       │       │
    │       │       ├── pts_backbone()
    │       │       ├── pts_neck()
    │       │       │
    │       │       └── bbox_head.loss()              # 计算损失
    │       │
    │       ▼
    │   ValLoop (定期)
    │
    ▼
checkpoint 保存
```

### 7.2 推理流程

```
推理命令
    │
    ▼
tools/test.py
    │
    ├── init_model(cfg, checkpoint)  # 初始化模型
    │
    ▼
model.forward(inputs, data_samples, mode='predict')
    │
    ├── data_preprocessor.forward()
    │
    ├── model.extract_feat()
    │
    ├── bbox_head.predict()
    │       │
    │       ├── heatmap → NMS → bbox
    │       │
    │       └── decode boxes (坐标解码)
    │
    ▼
add_pred_to_datasample()  # 转换为 Det3DDataSample
    │
    ▼
保存结果
```

---

## 8. 学习路线与实践

### 8.1 阶段一：环境搭建 (第1-2天)

```bash
# 1. 创建 conda 环境
conda create -n mmdet3d python=3.8
conda activate mmdet3d

# 2. 安装 PyTorch
pip install torch torchvision

# 3. 安装 mmcv
pip install openmim
mim install mmcv-full

# 4. 安装 mmdet 和 mmengine
pip install mmdet mmengine

# 5. 安装 mmdet3d
pip install mmdet3d

# 6. (可选) 编译扩展
cd projects/BEVFusion
python setup.py develop
```

### 8.2 阶段二：运行 Demo (第3天)

```bash
# 下载测试数据和模型
# 准备 nuScenes mini 数据

# 运行 demo
python demo/multi_modality_demo.py \
    data/nuscenes/xxx.pcd.bin \
    data/nuscenes/ \
    data/nuscenes/xxx.pkl \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_xxx.py \
    checkpoint.pth \
    --score-thr 0.2 \
    --show
```

### 8.3 阶段三：阅读核心代码 (第4-10天)

**必读文件清单**:

| 优先级 | 文件 | 行数 | 理由 |
|--------|------|------|------|
| ★★★★★ | `mmdet3d/models/detectors/base.py` | 150 | 基类，框架核心 |
| ★★★★★ | `mmdet3d/models/detectors/single_stage.py` | 170 | 单阶段模板 |
| ★★★★★ | `projects/BEVFusion/bevfusion/bevfusion.py` | 300 | BEVFusion 主模型 |
| ★★★★★ | `projects/BEVFusion/bevfusion/depth_lss.py` | 430 | View Transform 核心 |
| ★★★★☆ | `mmdet3d/datasets/det3d_dataset.py` | 300 | 数据集基类 |
| ★★★★☆ | `mmdet3d/datasets/transforms/loading.py` | 400 | 数据加载 |
| ★★★★☆ | `mmdet3d/structures/det3d_data_sample.py` | 200 | 数据结构 |
| ★★★☆☆ | `projects/BEVFusion/bevfusion/transfusion_head.py` | 900 | 检测头 |

### 8.4 阶段四：深入理解 (第11-15天)

1. **理解 BEV 空间表示**
   - 手动计算 BEV 坐标变换
   - 可视化 BEV 特征图

2. **理解稀疏卷积**
   - 阅读 SECOND 论文
   - 对比普通卷积和稀疏卷积

3. **理解多模态融合**
   - 对比不同融合策略 (早期融合/晚期融合/BEV融合)

### 8.5 阶段五：实践与扩展 (第16-20天)

1. **修改模型配置**
   - 改变检测范围
   - 调整特征通道数

2. **添加新组件**
   - 添加新的数据增强
   - 添加新的损失函数

3. **训练自己的数据**
   - 数据集格式转换
   - 配置文件修改

---

## 附录 A: 常用命令

```bash
# 训练
python tools/train.py configs/pointpillars/pointpillars_xxx.py

# 测试
python tools/test.py configs/pointpillars/pointpillars_xxx.py checkpoint.pth

# 分布式训练
bash tools/dist_train.sh configs/pointpillars/pointpillars_xxx.py 8

# 分布式测试
bash tools/dist_test.sh configs/pointpillars/pointpillars_xxx.py checkpoint.pth 8

# 模型转换
python tools/model_converters/xxx.py

# 数据集准备
python tools/create_data.py nuscenes --data-root data/nuscenes
```

---

## 附录 B: 调试技巧

```python
# 1. 打印中间特征形状
def extract_feat(self, batch_inputs_dict):
    x = self.backbone(x)
    print(f"Backbone output: {x.shape}")  # 调试
    x = self.neck(x)
    print(f"Neck output: {x.shape}")  # 调试
    return x

# 2. 可视化数据
from mmdet3d.visualization import Det3DLocalVisualizer
vis = Det3DLocalVisualizer()
vis.visualize_points(points, ...)

# 3. 使用更少的数据快速迭代
# 修改 dataloader 配置
train_dataloader = dict(batch_size=1, ...)  # 单样本测试
```

---

## 附录 C: 常见问题

**Q1: CUDA out of memory**
```python
# 方案1: 减小 batch size
train_dataloader = dict(batch_size=2, ...)

# 方案2: 启用混合精度
optim_wrapper = dict(..., amp=True)

# 方案3: 使用更小的模型配置
```

**Q2: 数据加载慢**
```python
# 方案1: 增加 num_workers
train_dataloader = dict(num_workers=8, ...)

# 方案2: 预加载数据到内存
# 在 dataset 中设置 cache
```

**Q3: 训练不收敛**
```python
# 检查学习率
optim_wrapper = dict(lr=0.002, ...)  # 检查学习率设置

# 检查损失函数
loss_weight=1.0  # 检查损失权重

# 检查数据标注
# 确保标注格式正确
```

---

*文档版本: v1.0*
*更新日期: 2024*
*参考: MMDetection3D 官方文档, BEVFusion 论文*
