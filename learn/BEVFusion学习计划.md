# BEVFusion 详细学习指南 (新手友好版)

## 目录
1. [BEVFusion 基础概念](#1-bevfusion-基础概念)
2. [BEVFusion 整体架构](#2-bevfusion-整体架构)
3. [核心模块代码详解](#3-核心模块代码详解)
4. [数据处理流程](#4-数据处理流程)
5. [配置文件解析](#5-配置文件解析)
6. [代码流程分析](#6-代码流程分析)
7. [实践指南](#7-实践指南)

---

## 1. BEVFusion 基础概念

### 1.1 什么是 BEVFusion？

BEVFusion 是由 MIT-Han-Lab 提出的一种多传感器融合 3D 检测方法，论文于 2022 年发表。

**论文**: [BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation](https://arxiv.org/abs/2205.13542)

**核心思想**: 将 LiDAR 点云和相机图像的特征都转换到 **Bird's Eye View (BEV)** 空间，然后在 BEV 空间进行融合。

### 1.2 什么是 BEV (Bird's Eye View)？

BEV 是从正上方俯视场景的视角，就像在看地图一样。

```
传统相机视角 (左)          BEV 视角 (右)

      车                      ┌─────────────────┐
     ╱  ╲                    │                 │
    ╱    ╲                   │      ┌───┐      │
   ▕──────▏                   │      │车 │ →    │
    ╲    ╱                    │      └───┘      │
     ╲  ╱                     │                 │
      ╲╱                      └─────────────────┘

在相机视角，车被遮挡        在BEV视角，可以清楚看到所有物体
在 BEV 视角，每个物体        的位置和大小
的位置、大小一目了然
```

**BEV 的优点**:
1. **无遮挡**: 从上方看，所有物体都不会互相遮挡
2. **方便融合**: 不同传感器的数据可以转换到同一空间
3. **适合下游任务**: 自动驾驶规划控制需要 BEV 视角

### 1.3 什么是多传感器融合？

自动驾驶汽车通常配备多种传感器：

```
自动驾驶汽车传感器配置
        ┌──────────────────────────────┐
        │         顶部 LiDAR           │ ← 360度旋转激光雷达
        │    (发射激光束,测量距离)      │
        └──────────────────────────────┘

前视相机      侧视相机      后视相机      环视相机
(CAM_FRONT)  (CAM_LEFT)  (CAM_BACK)  (CAM_RIGHT)
   │             │            │            │
   └─────────────┴────────────┴────────────┘
        (6个相机环绕,提供360度视野)
```

**各传感器优缺点**:

| 传感器 | 优点 | 缺点 |
|--------|------|------|
| LiDAR (激光雷达) | 精确的 3D 位置信息，不受光照影响 | 稀疏，语义信息少 |
| Camera (相机) | 丰富的语义信息，颜色纹理 | 没有深度信息，受光照影响 |

**融合的目的**: 结合两者的优点，得到既精确又有丰富语义的检测结果。

### 1.4 BEVFusion 的核心创新

**传统方法的融合方式** (Point-level Fusion):
```
图像 ──► CNN ──► 2D特征 ──► 投影到点云 ──► 点云特征
                                                    │
点云 ──► 体素化 ──► 特征提取 ──────────────────────► 融合特征 ──► 检测
                                                              ↑
                    问题: 投影会丢失语义信息! ◄────────────────┘
```

**BEVFusion 的融合方式** (BEV-space Fusion):
```
图像 ──► CNN ──► 2D特征 ──► View Transform ──► BEV特征 ──┐
                                                        │──► Concat ──► 检测
点云 ──► 体素化 ──► 特征提取 ──► BEV特征 ──────────────┘
                                │
                                └──► 保留几何+语义信息!
```

**关键创新**:
1. **统一 BEV 表示**: LiDAR 和 Camera 都转换到 BEV 空间
2. **优化 BEV Pooling**: 将 View Transform 加速 40 倍以上
3. **任务无关**: 可以用于检测、分割等多种任务

---

## 2. BEVFusion 整体架构

### 2.1 网络结构总览

```
BEVFusion 完整流程图:

┌────────────────────────────────────────────────────────────────────┐
│                         输入                                         │
│   ┌─────────────────┐              ┌─────────────────┐            │
│   │   6张相机图像    │              │   点云数据      │            │
│   │  (CAM_FRONT,    │              │  (x,y,z,intensity)
│   │   CAM_LEFT...)  │              │                 │            │
│   └────────┬────────┘              └────────┬────────┘            │
└────────────┼───────────────────────────────┼─────────────────────────┘
             │                                │
             ▼                                ▼
┌─────────────────────────┐      ┌─────────────────────────┐
│    图像分支 (Image)     │      │    点云分支 (LiDAR)     │
│                         │      │                         │
│  ┌───────────────────┐ │      │  ┌───────────────────┐ │
│  │ img_backbone      │ │      │  │  pts_voxel_encoder│ │
│  │ (SwinTransformer) │ │      │  │   体素化+编码     │ │
│  └─────────┬─────────┘ │      │  └─────────┬─────────┘ │
│            │           │      │            │           │
│            ▼           │      │            ▼           │
│  ┌───────────────────┐ │      │  ┌───────────────────┐ │
│  │    img_neck       │ │      │  │ pts_middle_encoder│ │
│  │  (GeneralizedLSS) │ │      │  │  (SparseEncoder) │ │
│  └─────────┬─────────┘ │      │  └─────────┬─────────┘ │
│            │           │      │            │           │
│            ▼           │      │            ▼           │
│  ┌───────────────────┐ │      │  ┌───────────────────┐ │
│  │  view_transform   │ │      │  │  pts_backbone    │ │
│  │  (DepthLSSTransform│ │     │  │    (SECOND)     │ │
│  │   2D → BEV)       │ │      │  └─────────┬─────────┘ │
│  └─────────┬─────────┘ │      │            │           │
└────────────┼───────────┘      └────────────┼───────────┘
             │                                │
             │         ┌─────────────────────┘
             │         │
             ▼         ▼
┌─────────────────────────────────────┐
│         Fusion Layer                 │
│      (ConvFuser: Concat + Conv)     │
│                                       │
│  [img_bev_feature] + [pts_bev_feature]
│         │                    │
│         └──► Concat ──► Conv ──► 融合特征
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│           pts_neck (SECONDFPN)      │
│              特征融合                 │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│          bbox_head                   │
│       (TransFusionHead)              │
│        检测头,输出框                  │
└─────────────────────────────────────┘
```

### 2.2 模块详解表

| 模块 | 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|------|
| `img_backbone` | SwinTransformer | 提取图像特征 | [B,6,3,256,704] | [B,6,C,32,88] 多尺度 |
| `img_neck` | GeneralizedLSSFPN | 特征融合 | 多尺度特征 | [B,6,256,32,88] |
| `view_transform` | DepthLSSTransform | 2D→BEV | [B,6,256,32,88] | [B,80,180,180] |
| `pts_voxel_encoder` | HardSimpleVFE | 点云体素编码 | points | voxel_features |
| `pts_middle_encoder` | BEVFusionSparseEncoder | 稀疏卷积 | voxel_features | [B,256,180,180] |
| `fusion_layer` | ConvFuser | 特征融合 | [B,80,180,180] + [B,256,180,180] | [B,256,180,180] |
| `pts_backbone` | SECOND | 特征提取 | [B,256,180,180] | 多尺度 |
| `pts_neck` | SECONDFPN | 特征融合 | 多尺度 | 融合特征 |
| `bbox_head` | TransFusionHead | 检测输出 | 特征 | bboxes, scores |

---

## 3. 核心模块代码详解

### 3.1 主模型 (BEVFusion)

**文件**: `projects/BEVFusion/bevfusion/bevfusion.py`

#### 3.1.1 类初始化

```python
@MODELS.register_module()
class BEVFusion(Base3DDetector):
    """
    BEVFusion 多模态 3D 检测器

    继承自 Base3DDetector,使用 MMDetection3D 的注册机制
    """

    def __init__(
        self,
        # 数据预处理器,负责归一化等
        data_preprocessor: OptConfigType = None,

        # ========== 点云分支 ==========
        pts_voxel_encoder: Optional[dict] = None,   # 体素编码器
        pts_middle_encoder: Optional[dict] = None,   # 中间编码器

        # ========== 图像分支 ==========
        img_backbone: Optional[dict] = None,        # 图像骨干网络
        img_neck: Optional[dict] = None,            # 图像 neck

        # ========== 融合相关 ==========
        view_transform: Optional[dict] = None,      # 视角变换 (图像→BEV)
        fusion_layer: Optional[dict] = None,         # 融合层

        # ========== 点云处理 ==========
        pts_backbone: Optional[dict] = None,        # 点云骨干网络
        pts_neck: Optional[dict] = None,           # 点云 neck

        # ========== 检测头 ==========
        bbox_head: Optional[dict] = None,           # 检测头
        ...
    ) -> None:
        # 先处理 voxelize 相关的预处理器配置
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(data_preprocessor=data_preprocessor, ...)

        # 点云体素化层
        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)

        # 使用 MODELS.build() 从配置创建各模块
        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)
        self.img_backbone = MODELS.build(img_backbone) if img_backbone else None
        self.img_neck = MODELS.build(img_neck) if img_neck else None
        self.view_transform = MODELS.build(view_transform) if view_transform else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
        self.fusion_layer = MODELS.build(fusion_layer) if fusion_layer else None
        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)
        self.bbox_head = MODELS.build(bbox_head)
```

#### 3.1.2 特征提取 (extract_feat)

这是 BEVFusion 的核心函数，负责处理双分支输入并融合。

```python
def extract_feat(self, batch_inputs_dict, batch_input_metas, **kwargs):
    """
    特征提取核心函数

    Args:
        batch_inputs_dict: 批量输入数据
            - 'imgs': 图像张量 [B, N, C, H, W]
                B: batch size
                N: 相机数量 (nuScenes 有 6 个相机)
                C: 通道数 (3 for RGB)
                H: 图像高度 (256)
                W: 图像宽度 (704)
            - 'points': 点云列表 [[point_num, C], ...]
                每帧点云包含 [x, y, z, intensity]

        batch_input_metas: 每帧的元信息
            - 'lidar2img': LiDAR 到图像的投影矩阵
            - 'cam2img': 相机内参矩阵
            - 'cam2lidar': 相机到 LiDAR 外参矩阵
            - 'img_aug_matrix': 图像增强矩阵
            - 'lidar_aug_matrix': LiDAR 增强矩阵

    Returns:
        x: 融合后的 BEV 特征 [B, C, H, W]
    """

    imgs = batch_inputs_dict.get('imgs', None)
    points = batch_inputs_dict.get('points', None)
    features = []

    # ========== 图像分支 ==========
    if imgs is not None:
        imgs = imgs.contiguous()  # 确保内存连续

        # 提取相机参数
        # 这些矩阵用于后续坐标变换
        lidar2image, camera_intrinsics, camera2lidar = [], [], []
        img_aug_matrix, lidar_aug_matrix = [], []

        for i, meta in enumerate(batch_input_metas):
            # LiDAR 坐标到图像像素的投影矩阵
            # 用于将 LiDAR 点投影到图像上
            lidar2image.append(meta['lidar2img'])

            # 相机内参矩阵 (fx, fy, cx, cy)
            camera_intrinsics.append(meta['cam2img'])

            # 相机到 LiDAR 的变换矩阵 (外参)
            camera2lidar.append(meta['cam2lidar'])

            # 图像增强矩阵 (resize, crop, flip, rotate)
            img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
            lidar_aug_matrix.append(meta.get('lidar_aug_matrix', np.eye(4)))

        # 转换为张量
        lidar2image = imgs.new_tensor(np.asarray(lidar2image))
        camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
        camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
        img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
        lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))

        # 提取图像特征: backbone → neck → view_transform
        img_feature = self.extract_img_feat(
            imgs,
            deepcopy(points),  # 深度拷贝,避免修改原始数据
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            batch_input_metas
        )
        features.append(img_feature)

    # ========== 点云分支 ==========
    pts_feature = self.extract_pts_feat(batch_inputs_dict)
    features.append(pts_feature)

    # ========== 特征融合 ==========
    if self.fusion_layer is not None:
        # 将图像 BEV 特征和点云 BEV 特征在通道维度拼接后卷积
        x = self.fusion_layer(features)
    else:
        # 只有点云分支
        x = features[0]

    # ========== 检测特征提取 ==========
    x = self.pts_backbone(x)    # SECOND 特征提取
    x = self.pts_neck(x)        # FPN 特征融合

    return x
```

#### 3.1.3 图像特征提取 (extract_img_feat)

```python
def extract_img_feat(self, x, points, lidar2image, camera_intrinsics,
                    camera2lidar, img_aug_matrix, lidar_aug_matrix, img_metas):
    """
    提取图像特征并转换到 BEV 空间

    Args:
        x: 图像 [B, N, C, H, W]
        points: 点云 (用于深度监督)
        lidar2image: LiDAR→图像投影矩阵 [B, N, 4, 4]
        camera_intrinsics: 相机内参 [B, N, 4, 4]
        camera2lidar: 相机→LiDAR外参 [B, N, 4, 4]
        img_aug_matrix: 图像增强矩阵 [B, N, 4, 4]
        lidar_aug_matrix: LiDAR增强矩阵 [B, N, 4, 4]
        img_metas: 元信息列表

    Returns:
        BEV 空间图像特征 [B, C, H_bev, W_bev]
    """

    B, N, C, H, W = x.size()
    # [B, N, C, H, W] -> [B*N, C, H, W]
    x = x.view(B * N, C, H, W).contiguous()

    # 1. backbone: SwinTransformer 提取多尺度图像特征
    # 输出: [B*N, 768, 32, 88] (多尺度特征)
    x = self.img_backbone(x)

    # 2. neck: GeneralizedLSSFPN 融合多尺度特征
    # 输出: [B*N, 256, 32, 88]
    x = self.img_neck(x)

    # 处理多尺度输出
    if not isinstance(x, torch.Tensor):
        x = x[0]

    # [B*N, C, H, W] -> [B, N, C, H, W]
    BN, C, H, W = x.size()
    x = x.view(B, int(BN / B), C, H, W)

    # 3. view_transform: 将 2D 特征转换到 BEV 空间
    # 这是 BEVFusion 的核心创新点!
    # 使用混合精度加速
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        x = self.view_transform(
            x,                    # 图像特征
            points,               # 点云
            lidar2image,          # 投影矩阵
            camera_intrinsics,    # 内参
            camera2lidar,         # 外参
            img_aug_matrix,       # 图像增强
            lidar_aug_matrix,     # LiDAR增强
            img_metas,            # 元信息
        )

    return x
```

#### 3.1.4 点云特征提取 (extract_pts_feat)

```python
def extract_pts_feat(self, batch_inputs_dict):
    """提取点云特征"""
    points = batch_inputs_dict['points']

    # 使用 float32 精度处理点云
    with torch.autocast('cuda', enabled=False):
        points = [point.float() for point in points]

        # 体素化: 将点云划分为体素网格
        # 返回:
        # - feats: [N_voxels, C] 每个体素的特征
        # - coords: [N_voxels, 4] (batch_idx, x, y, z)
        # - sizes: [N_voxels] 每个体素包含的点数
        feats, coords, sizes = self.voxelize(points)

        # 计算 batch_size
        batch_size = coords[-1, 0] + 1

        # 中间编码器: 稀疏卷积处理
        # 输入: voxel features, coordinates, batch_size
        # 输出: BEV 特征图 [B, C, H, W]
        x = self.pts_middle_encoder(feats, coords, batch_size)

    return x
```

#### 3.1.5 体素化 (voxelize)

```python
@torch.no_grad()
def voxelize(self, points):
    """
    将点云体素化

    Args:
        points: 点云列表 [[N_i, C], ...]

    Returns:
        feats: [N_voxels, C] 体素特征
        coords: [N_voxels, 4] 体素坐标 (batch_idx, x, y, z)
        sizes: [N_voxels] 每体素点数 (用于平均)
    """

    feats, coords, sizes = [], [], []

    for k, res in enumerate(points):
        # 对每帧点云分别体素化
        ret = self.pts_voxel_layer(res)

        if len(ret) == 3:
            # 硬体素化: 有限制点数
            f, c, n = ret
        else:
            # 软体素化: 无限制
            f, c = ret
            n = None

        feats.append(f)
        # 在 x 维度添加 batch_index
        coords.append(F.pad(c, (1, 0), mode='constant', value=k))

        if n is not None:
            sizes.append(n)

    # 拼接所有帧
    feats = torch.cat(feats, dim=0)
    coords = torch.cat(coords, dim=0)

    if len(sizes) > 0:
        sizes = torch.cat(sizes, dim=0)

        # 如果启用 voxelize_reduce,对同一体素内的点取平均
        if self.voxelize_reduce:
            # sizes.type_as(feats) 确保类型一致
            feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
            feats = feats.contiguous()

    return feats, coords, sizes
```

---

### 3.2 视角变换 (DepthLSSTransform)

**文件**: `projects/BEVFusion/bevfusion/depth_lss.py`

这是 BEVFusion 最核心的创新模块，负责将 2D 图像特征转换到 BEV 空间。

#### 3.2.1 类的继承关系

```python
class BaseViewTransform(nn.Module):
    """视角变换的基类"""
    # 定义 frustum, 计算几何变换, BEV pooling 等基础功能

class BaseDepthTransform(BaseViewTransform):
    """基于深度估计的变换基类"""
    # 使用点云监督深度估计

class DepthLSSTransform(BaseDepthTransform):
    """LSS 风格的深度感知视角变换"""
    # BEVFusion 实际使用的类
```

#### 3.2.2 初始化参数详解

```python
class DepthLSSTransform(BaseDepthTransform):
    def __init__(self,
                 in_channels=256,         # 输入通道数 (来自 img_neck)
                 out_channels=80,          # 输出 BEV 通道数
                 image_size=[256, 704],   # 输入图像尺寸 [H, W]
                 feature_size=[32, 88],   # 特征图尺寸 [H, W] (图像/8)
                 xbound=[-54.0, 54.0, 0.3],   # BEV x 范围和分辨率
                 ybound=[-54.0, 54.0, 0.3],   # BEV y 范围和分辨率
                 zbound=[-10.0, 10.0, 20.0],  # BEV z 范围
                 dbound=[1.0, 60.0, 0.5],      # 深度范围和分辨率
                 downsample=2,             # 下采样因子
                 ...):
```

**关键参数解释**:

| 参数 | 值 | 含义 |
|------|-----|------|
| `xbound` | [-54, 54, 0.3] | x 方向范围 [-54m, 54m], 步长 0.3m |
| `ybound` | [-54, 54, 0.3] | y 方向范围 [-54m, 54m], 步长 0.3m |
| `zbound` | [-10, 10, 20] | z 方向范围 [-10m, 10m], 20 个 slice |
| `dbound` | [1, 60, 0.5] | 深度范围 [1m, 60m], 步长 0.5m |
| `image_size` | [256, 704] | 输入相机图像尺寸 |
| `feature_size` | [32, 88] | 特征图尺寸 (图像尺寸/8) |

**BEV 网格大小计算**:
```
x 方向: (54 - (-54)) / 0.3 = 360 个格子
y 方向: (54 - (-54)) / 0.3 = 360 个格子
但实际配置中 grid_size 是 [180, 180], 因为后续有 downsample=2
```

#### 3.2.3 创建 Frustum (视锥体)

```python
def create_frustum(self):
    """
    创建用于采样的视锥体网格

    对于图像上的每个像素,沿深度方向创建多个采样点

    Returns:
        frustum: [D, H, W, 3] 采样点坐标 (u, v, d)
    """
    iH, iW = self.image_size   # 256, 704
    fH, fW = self.feature_size  # 32, 88 (特征图尺寸)

    # 沿深度方向创建采样点
    # ds shape: [D, 1, 1], D = (60-1)/0.5 = 118 层
    ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
    D, _, _ = ds.shape  # D = 118

    # 图像 x 坐标 (列)
    # xs shape: [1, 1, W] -> [1, 1, 88]
    xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)

    # 图像 y 坐标 (行)
    # ys shape: [1, H, 1] -> [1, 32, 1]
    ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

    # 堆叠为 (u, v, d) 坐标
    # frustum shape: [D, H, W, 3]
    frustum = torch.stack((xs, ys, ds), -1)

    return nn.Parameter(frustum, requires_grad=False)
```

**Frustum 图解**:
```
图像平面 (u,v)                    深度方向 (d)
    ┌─────────────────┐
    │  ●  ●  ●  ●  ● │  ───► 沿深度方向扩展
    │  ●  ●  ●  ●  ● │       每像素有 D 个采样点
    │  ●  ●  ●  ●  ● │
    └─────────────────┘

    每个 (u,v) 位置沿深度 d 创建多个采样点
    形成一个 "视锥体" (Frustum)
```

#### 3.2.4 几何变换 (get_geometry)

```python
def get_geometry(self,
                 camera2lidar_rots,    # 相机→LiDAR 旋转矩阵 [B, N, 3, 3]
                 camera2lidar_trans,   # 相机→LiDAR 平移向量 [B, N, 3]
                 intrins,              # 相机内参 [B, N, 3, 3]
                 post_rots,            # 图像增强旋转 [B, N, 3, 3]
                 post_trans,           # 图像增强平移 [B, N, 3]
                 ...):
    """
    计算每个采样点到 BEV 空间的映射关系

    坐标变换链:
    图像像素 → 相机坐标 → LiDAR坐标 → BEV坐标
    """

    B, N, _ = camera2lidar_trans.shape

    # ======== 1. 逆向图像增强 ========
    # 先应用增强再逆向,恢复到增强前的坐标
    points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
    # 逆向旋转变换
    points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
    # points shape: [B, N, D, H, W, 3]

    # ======== 2. 相机坐标变换 (考虑畸变) ========
    # 像素坐标乘以深度得到相机坐标系下的 3D 坐标
    # x = u * d, y = v * d, z = d
    points = torch.cat(
        (
            points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],  # u*d, v*d
            points[:, :, :, :, :, 2:3],                               # d
        ),
        5,  # cat dim
    )
    # points shape: [B, N, D, H, W, 3] (x_cam, y_cam, z_cam)

    # ======== 3. 相机 → LiDAR 坐标变换 ========
    # P_lidar = R @ P_cam + t
    combine = camera2lidar_rots.matmul(torch.inverse(intrins))
    points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
    points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

    # ======== 4. 考虑 LiDAR 增强 (全局旋转缩放) ========
    if 'extra_rots' in kwargs:
        extra_rots = kwargs['extra_rots']
        points = extra_rots.view(B, 1, 1, 1, 1, 3, 3).repeat(1, N, 1, 1, 1, 1, 1).matmul(points.unsqueeze(-1)).squeeze(-1)

    if 'extra_trans' in kwargs:
        extra_trans = kwargs['extra_trans']
        points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

    return points  # [B, N, D, H, W, 3] LiDAR 坐标系下的采样点坐标
```

**坐标变换图解**:
```
图像像素 (u,v)
       │
       │ 乘以深度 d
       ▼
相机坐标 (x_cam, y_cam, z_cam=d)
       │
       │ camera2lidar 外参变换
       ▼
LiDAR坐标 (x_lidar, y_lidar, z_lidar)
       │
       │ 全局增强逆变换 (如果有)
       ▼
最终 BEV 坐标 (用于 BEV pooling)
```

#### 3.2.5 深度估计和特征提取 (get_cam_feats)

```python
def get_cam_feats(self, x):
    """
    从图像特征预测深度分布并提取语义特征

    Args:
        x: 图像特征 [B, N, C, fH, fW]

    Returns:
        x: 深度加权的特征 [B, N, D, C, fH, fW]
    """
    B, N, C, fH, fW = x.shape

    # x.view: [B, N, C, fH, fW] -> [B*N, C, fH, fW]
    x = x.view(B * N, C, fH, fW)

    # depthnet: 预测深度分布和语义特征
    # 输出通道: D(深度层数) + C(特征通道)
    x = self.depthnet(x)  # [B*N, D+C, fH, fW]

    # 深度分布 (softmax 得到概率)
    depth = x[:, :self.D].softmax(dim=1)  # [B*N, D, fH, fW]

    # 语义特征
    # x[:, self.D:] shape: [B*N, C, fH, fW]
    # depth.unsqueeze(1) shape: [B*N, 1, D, fH, fW]
    # 相乘得到: [B*N, C, D, fH, fW]
    x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

    # 调整维度顺序: [B*N, C, D, fH, fW] -> [B, N, D, fH, fW, C]
    x = x.view(B, N, self.C, self.D, fH, fW)
    x = x.permute(0, 1, 3, 4, 5, 2)  # [B, N, D, fH, fW, C]

    return x
```

**图解**:
```
输入特征
[B*N, C, fH, fW]
     │
     ▼
┌────────────┐
│ depthnet  │ Conv2d(in=C, out=D+C)
└─────┬─────┘
      │
      ├──────► softmax ──► depth [B*N, D, fH, fW] (每点深度概率)
      │
      └──────► 取后半部分 ──► feat [B*N, C, fH, fW]

depth[:,:,None] * feat[:,None,:]
     │                    │
     ▼                    ▼
[B*N, D, 1, fH, fW]  [B*N, 1, C, fH, fW]
     │
     │ element-wise multiply
     ▼
[B*N, C, D, fH, fW] ──► reshape ─► [B, N, D, fH, fW, C]
```

#### 3.2.6 BEV Pooling

```python
def bev_pool(self, geom_feats, x):
    """
    将图像特征聚合到 BEV 空间

    这是 BEVFusion 优化的核心,使用高效的 CUDA 算子

    Args:
        geom_feats: 采样点对应的 BEV 坐标 [B*N*D*H*W, 3]
        x: 采样点的图像特征 [B*N*D*H*W, C]

    Returns:
        BEV 特征图 [B, C, H_bev, W_bev]
    """
    B, N, D, H, W, C = x.shape
    Nprime = B * N * D * H * W  # 总采样点数

    # ======== 1. 展平 ========
    x = x.reshape(Nprime, C)      # [Nprime, C]
    geom_feats = geom_feats.view(Nprime, 3)  # [Nprime, 3]

    # ======== 2. 坐标归一化为体素索引 ========
    # (coord - start) / stride = index
    geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()

    # ======== 3. 添加 batch 索引 ========
    batch_ix = torch.cat([
        torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
        for ix in range(B)
    ])
    geom_feats = torch.cat((geom_feats, batch_ix), 1)  # [Nprime, 4]

    # ======== 4. 过滤范围外的点 ========
    kept = (
        (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) &  # x
        (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) &  # y
        (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])    # z
    )
    x = x[kept]
    geom_feats = geom_feats[kept]

    # ======== 5. BEV Pooling (CUDA 加速) ========
    # 高效地将落在同一 BEV 体素的特征聚合
    x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

    # ======== 6. 压缩 Z 维度 ========
    # 将 Z 维度的特征拼接回 C 维度
    final = torch.cat(x.unbind(dim=2), 1)

    return final  # [B, C, H_bev, W_bev]
```

**BEV Pooling 图解**:
```
输入: 图像特征 + BEV 坐标
┌─────────────────────────────────────────┐
│ 采样点1: (x=10.5, y=20.3, z=1.2) → 特征1  │
│ 采样点2: (x=10.7, y=20.1, z=0.8) → 特征2  │
│ 采样点3: (x=35.2, y=45.8, z=2.1) → 特征3  │
│ ...                                      │
│ 采样点N: (x=-12.3, y=8.5, z=0.5) → 特征N  │
└─────────────────────────────────────────┘
                    │
                    ▼ 转换为体素索引
┌─────────────────────────────────────────┐
│ 体素(10, 20, 1): [特征1, 特征2]          │ ← 聚合
│ 体素(35, 45, 2): [特征3]                │
│ ...                                      │
└─────────────────────────────────────────┘
                    │
                    ▼ Sum pooling
┌─────────────────────────────────────────┐
│      BEV Feature Map                    │
│   [B, C, H_bev=180, W_bev=180]         │
└─────────────────────────────────────────┘
```

---

### 3.3 融合层 (ConvFuser)

**文件**: `projects/BEVFusion/bevfusion/transfusion_head.py`

```python
@MODELS.register_module()
class ConvFuser(nn.Sequential):
    """
    融合 LiDAR BEV 特征和 Camera BEV 特征

    融合方式: 通道维度拼接 + 卷积融合
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels  # [80, 256]
        self.out_channels = out_channels  # 256

        super().__init__(
            # 拼接: [80, 256] -> 336 通道
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: [camera_bev_feature, lidar_bev_feature]
                - camera_bev_feature: [B, 80, H, W]
                - lidar_bev_feature: [B, 256, H, W]

        Returns:
            fused: [B, 256, H, W]
        """
        # 通道维度拼接
        return super().forward(torch.cat(inputs, dim=1))
```

---

### 3.4 检测头 (TransFusionHead)

**文件**: `projects/BEVFusion/bevfusion/transfusion_head.py`

TransFusionHead 负责从融合特征预测 3D 边界框。

```python
@MODELS.register_module()
class TransFusionHead(nn.Module):
    """
    基于 Transformer 的 3D 检测头

    结构:
    1. 共享卷积层
    2. HeatmapHead 生成热力图
    3. Transformer 解码器层
    4. 回归头
    """

    def __init__(self,
                 num_proposals=128,      # 提案数量
                 in_channels=384,        # 输入通道 (512 = 256 + 256)
                 hidden_channel=128,      # Transformer 隐层通道
                 num_classes=10,         # 类别数
                 num_decoder_layers=3,   # Transformer 解码器层数
                 ...):
        super().__init__()

        # 共享卷积
        self.shared_conv = build_conv_layer(...)

        # Heatmap 预测头
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channel, 3, padding=1),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(),
            nn.Conv2d(hidden_channel, num_classes, 1),  # 每个类别一个 heatmap
        )

        # 回归头: center, height, dim, rot, vel
        self.task_head = nn.ModuleDict({
            'center': nn.Conv2d(...),    # 中心点偏移
            'height': nn.Conv2d(...),    # 高度
            'dim': nn.Conv2d(...),       # 尺寸
            'rot': nn.Conv2d(...),       # 旋转角 (sin, cos)
            'vel': nn.Conv2d(...),       # 速度 (仅 nuScenes)
        })

        # Transformer 解码器
        self.transformer = nn.ModuleList([
            TransformerDecoderLayer(...) for _ in range(num_decoder_layers)
        ])

        # 损失函数
        self.loss_cls = FocalLoss()
        self.loss_bbox = L1Loss()
        self.loss_heatmap = GaussianFocalLoss()

    def forward(self, x, batch_data_samples):
        """
        Args:
            x: 融合特征 [B, C, H, W]
            batch_data_samples: 数据样本

        Returns:
            预测结果
        """
        # 1. 共享卷积
        x = self.shared_conv(x)

        # 2. 生成 heatmap 和初始预测
        heatmap = self.heatmap_head(x)

        # 3. Transformer 解码器精炼
        for layer in self.transformer:
            x = layer(x, heatmap)

        # 4. 回归预测
        preds = {
            'heatmap': heatmap,
            'center': self.task_head['center'](x),
            'height': self.task_head['height'](x),
            'dim': self.task_head['dim'](x),
            'rot': self.task_head['rot'](x),
            'vel': self.task_head['vel'](x),
        }

        return preds
```

---

## 4. 数据处理流程

### 4.1 完整训练 Pipeline

```python
train_pipeline = [
    # ======== 1. 数据加载 ========
    dict(
        type='BEVLoadMultiViewImageFromFiles',  # 加载 6 个相机的图像
        to_float32=True,
        color_type='color',
    ),

    dict(
        type='LoadPointsFromFile',  # 加载点云
        coord_type='LIDAR',
        load_dim=5,      # x, y, z, intensity, ring_index
        use_dim=5,      # 使用前 5 维
    ),

    dict(
        type='LoadPointsFromMultiSweeps',  # 加载历史帧
        sweeps_num=9,   # 加载 9 帧历史点云
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
    ),

    dict(
        type='LoadAnnotations3D',  # 加载 3D 标注
        with_bbox_3d=True,
        with_label_3d=True,
    ),

    # ======== 2. 数据增强 ========
    dict(
        type='ImageAug3D',  # 图像增强
        final_dim=[256, 704],      # 输出尺寸
        resize_lim=[0.38, 0.55],   # 缩放范围
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],       # 旋转范围 (度)
        rand_flip=True,             # 随机翻转
        is_train=True,
    ),

    dict(
        type='BEVFusionGlobalRotScaleTrans',  # 全局旋转缩放
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.785, 0.785],  # 弧度
        translation_std=0.5,
    ),

    dict(type='BEVFusionRandomFlip3D'),  # 随机翻转

    # ======== 3. 过滤 ========
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),

    # ======== 4. 打包 ========
    dict(type='Pack3DDetInputs'),
]
```

### 4.2 数据加载详解

#### 4.2.1 BEVLoadMultiViewImageFromFiles

```python
class BEVLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """
    加载多视角相机图像并计算相机参数

    nuScenes 有 6 个相机:
    - CAM_FRONT: 前视
    - CAM_FRONT_LEFT: 左前视
    - CAM_FRONT_RIGHT: 右前视
    - CAM_BACK: 后视
    - CAM_BACK_LEFT: 左后视
    - CAM_BACK_RIGHT: 右后视
    """

    def transform(self, results):
        # 1. 加载 6 张图像
        imgs = [load_image(path) for path in results['img_filename']]

        # 2. 计算相机参数
        for cam_name in results['images'].keys():
            # lidar2cam: LiDAR 到相机的变换
            lidar2cam = results['images'][cam_name]['lidar2cam']

            # cam2lidar: 相机到 LiDAR (外参的逆)
            cam2lidar = inverse(lidar2cam)

            # cam2img: 相机内参
            cam2img = results['images'][cam_name]['cam2img']

            # lidar2img: LiDAR 到图像的投影
            lidar2img = cam2img @ lidar2cam

        # 3. 存储相机参数
        results['cam2img'] = np.stack(cam2img)      # [6, 4, 4]
        results['lidar2cam'] = np.stack(lidar2cam)  # [6, 4, 4]
        results['cam2lidar'] = np.stack(cam2lidar)  # [6, 4, 4]
        results['lidar2img'] = np.stack(lidar2img)  # [6, 4, 4]

        return results
```

### 4.3 数据增强详解

#### 4.3.1 ImageAug3D

```python
class ImageAug3D:
    """
    3D 场景的图像增强

    关键点: 图像增强必须与点云增强一致!

    增强参数:
    - final_dim: 输出图像尺寸
    - resize_lim: 缩放因子范围
    - rot_lim: 旋转角度范围
    - rand_flip: 是否随机翻转
    """

    def transform(self, data):
        # 1. 采样增强参数
        resize = np.random.uniform(0.38, 0.55)  # 缩放因子
        rotate = np.random.uniform(-5.4, 5.4)   # 旋转角度 (度)
        flip = np.random.choice([0, 1])         # 是否翻转

        # 2. 对每张图像应用变换
        for i, img in enumerate(data['imgs']):
            img = resize(img, resize)
            img = rotate(img, rotate)
            if flip:
                img = horizontal_flip(img)

        # 3. 记录变换矩阵 (用于后续坐标变换)
        # 这个矩阵用于将增强后的坐标转换回原始坐标系
        data['img_aug_matrix'] = compute_aug_matrix(resize, rotate, flip)

        return data
```

**一致性保证**:
```
图像增强 → 应用变换矩阵
    │
    └──► 计算 img_aug_matrix
             │
             ▼
    view_transform 使用此矩阵
    逆向变换采样点坐标
             │
             ▼
    确保图像和点云的增强一致!
```

#### 4.3.2 BEVFusionGlobalRotScaleTrans

```python
class BEVFusionGlobalRotScaleTrans:
    """
    全局旋转、缩放、平移

    同时应用于点云和 3D 边界框
    """

    def transform(self, input_dict):
        # 1. 采样变换参数
        scale = np.random.uniform(0.9, 1.1)  # 缩放
        rotation = np.random.uniform(-0.785, 0.785)  # 旋转 (弧度)
        translation = np.random.normal(0, 0.5, 3)  # 平移

        # 2. 应用到点云
        input_dict['points'] = transform_points(
            input_dict['points'],
            scale=scale,
            rotation=rotation,
            translation=translation
        )

        # 3. 应用到 3D 边界框
        input_dict['gt_bboxes_3d'] = transform_bboxes_3d(
            input_dict['gt_bboxes_3d'],
            scale=scale,
            rotation=rotation,
            translation=translation
        )

        # 4. 记录变换矩阵 (用于 view_transform)
        lidar_aug_matrix = np.eye(4)
        lidar_aug_matrix[:3, :3] = rotation_matrix * scale
        lidar_aug_matrix[:3, 3] = translation
        input_dict['lidar_aug_matrix'] = lidar_aug_matrix

        return input_dict
```

---

## 5. 配置文件解析

### 5.1 LiDAR + Camera 融合配置

```python
# bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py

# 基础配置 (继承自 LiDAR-only 配置)
_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# 检测范围 (LiDAR 坐标系)
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
# x: [-54, 54], y: [-54, 54], z: [-5, 3] (单位: 米)

# 传感器配置
input_modality = dict(use_lidar=True, use_camera=True)

# 模型配置
model = dict(
    type='BEVFusion',

    # ========== 数据预处理器 ==========
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],  # ImageNet 均值
        std=[58.395, 57.12, 57.375],      # ImageNet 标准差
        bgr_to_rgb=False,
    ),

    # ========== 图像分支 ==========
    img_backbone=dict(
        type='mmdet.SwinTransformer',  # 来自 mmdet
        embed_dims=96,                  # 嵌入维度
        depths=[2, 2, 6, 2],           # 4 个 stage 的层数
        num_heads=[3, 6, 12, 24],      # 每 stage 的注意力头数
        window_size=7,                  # 窗口大小
        out_indices=[1, 2, 3],         # 输出多尺度特征
        # 预训练权重
        init_cfg=dict(
            type='Pretrained',
            checkpoint='swin_tiny_patch4_window7_224.pth'
        ),
    ),

    img_neck=dict(
        type='GeneralizedLSSFPN',  # LSS 风格的 FPN
        in_channels=[192, 384, 768],  # SwinTransformer 输出通道
        out_channels=256,              # 输出统一通道
        start_level=0,
        num_outs=3,                  # 输出 3 个尺度
    ),

    view_transform=dict(
        type='DepthLSSTransform',  # 核心! 2D → BEV
        in_channels=256,            # 输入通道
        out_channels=80,            # 输出 BEV 通道

        # BEV 空间定义
        image_size=[256, 704],       # 输入图像尺寸
        feature_size=[32, 88],       # 特征图尺寸 (图像/8)

        # x, y, z 范围 [min, max, step]
        xbound=[-54.0, 54.0, 0.3],   # 360 个格子
        ybound=[-54.0, 54.0, 0.3],   # 360 个格子
        zbound=[-10.0, 10.0, 20.0],  # z 方向 slice

        # 深度范围 [min, max, step]
        dbound=[1.0, 60.0, 0.5],     # 118 个深度层

        downsample=2,                 # 下采样
    ),

    # ========== 融合层 ==========
    fusion_layer=dict(
        type='ConvFuser',
        in_channels=[80, 256],   # [图像特征, 点云特征]
        out_channels=256,
    ),

    # ========== 点云分支 (略,同 LiDAR-only) ==========
    pts_voxel_encoder=...,
    pts_middle_encoder=...,
    pts_backbone=...,
    pts_neck=...,

    # ========== 检测头 ==========
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=200,         # Transformer 查询数
        in_channels=512,           # 融合特征通道 (256 + 256)
        hidden_channel=128,        # Transformer 隐层
        num_classes=10,             # nuScenes 10 类
        num_decoder_layers=1,      # Transformer 层数
        ...
    ),
)
```

### 5.2 关键参数说明

| 参数 | 值 | 含义 |
|------|-----|------|
| `point_cloud_range` | [-54, -54, -5, 54, 54, 3] | 检测范围 (米) |
| `xbound/ybound` | [-54, 54, 0.3] | 每个 BEV 格子 0.3m |
| `voxel_size` | [0.075, 0.075, 0.2] | 体素大小 |
| `dbound` | [1, 60, 0.5] | 深度范围 1-60m |
| `feature_size` | [32, 88] | 特征图尺寸 |

**网格尺寸计算**:
```
BEV 网格: (54 - (-54)) / 0.3 = 360
但实际是 downsample=2 后: 180 x 180

体素网格: (54 - (-54)) / 0.075 = 1440
深度层数: (3 - (-5)) / 0.2 = 40 ≈ 41
```

---

## 6. 代码流程分析

### 6.1 训练流程

```
用户执行:
bash tools/dist_train.sh projects/BEVFusion/configs/xxx.py 8

    │
    ▼
tools/train.py
    │
    ├── Runner.from_cfg(cfg)  # 从配置创建 Runner
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
    │       │   model.forward(batch_inputs, batch_data_samples, mode='loss')
    │       │       │
    │       │       ├── data_preprocessor.forward()  # 归一化
    │       │       │
    │       │       ├── extract_feat()              # 特征提取
    │       │       │       │
    │       │       │       ├── extract_img_feat()  # 图像分支
    │       │       │       │       ├── img_backbone (SwinTransformer)
    │       │       │       │       ├── img_neck (FPN)
    │       │       │       │       └── view_transform (2D→BEV)
    │       │       │       │
    │       │       │       └── extract_pts_feat()  # 点云分支
    │       │       │               ├── voxelize
    │       │       │               └── pts_middle_encoder
    │       │       │
    │       │       ├── fusion_layer()              # 特征融合
    │       │       │
    │       │       ├── pts_backbone()              # SECOND
    │       │       │
    │       │       ├── pts_neck()                  # FPN
    │       │       │
    │       │       └── bbox_head.loss()            # 计算损失
    │       │
    │       ▼
    │   ValLoop (每 val_interval 个 epoch)
    │
    ▼
checkpoint 保存
```

### 6.2 推理流程

```
用户执行:
python demo/multi_modality_demo.py ...

    │
    ▼
加载模型
init_model(config_file, checkpoint)

    │
    ▼
inference_detector(model, data)
    │
    ├── model.forward(batch_inputs, batch_data_samples, mode='predict')
    │       │
    │       ├── extract_feat()
    │       │
    │       └── bbox_head.predict()
    │               │
    │               ├── heatmap → NMS → topk proposals
    │               │
    │               └── 边界框解码
    │
    ▼
结果后处理
    │
    ▼
可视化或保存
```

---

## 7. 实践指南

### 7.1 环境搭建

```bash
# 1. 创建 conda 环境
conda create -n bevfusion python=3.8
conda activate bevfusion

# 2. 安装 PyTorch
pip install torch==1.12.0 torchvision

# 3. 安装 MMCV
pip install openmim
mim install mmcv-full

# 4. 安装 MMDET 和 MMEngine
pip install mmdet mmengine

# 5. 克隆 MMDetection3D
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -v .
```

### 7.2 编译 BEV Pooling 算子

```bash
cd projects/BEVFusion
python setup.py develop

# 或使用
pip install -e .
```

### 7.3 运行 Demo

```bash
# 下载测试数据和模型
# 准备 nuScenes mini 数据

# 运行多模态 demo
python projects/BEVFusion/demo/multi_modality_demo.py \
    demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin \
    demo/data/nuscenes/ \
    demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    checkpoint.pth \
    --score-thr 0.2 \
    --show
```

### 7.4 训练模型

```bash
# 1. 先训练 LiDAR-only 模型
bash tools/dist_train.sh \
    projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    8

# 2. 下载图像预训练权重
# SwinTransformer 权重

# 3. 训练 LiDAR + Camera 融合模型
bash tools/dist_train.sh \
    projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
    8 \
    --cfg-options \
    load_from=/path/to/lidar_checkpoint.pth \
    model.img_backbone.init_cfg.checkpoint=/path/to/swin_checkpoint.pth
```

### 7.5 常用调试技巧

```python
# 1. 打印中间特征形状
def extract_feat(self, batch_inputs_dict):
    x = self.extract_img_feat(...)
    print(f"图像 BEV 特征: {x.shape}")  # [B, 80, 180, 180]

    pts_feature = self.extract_pts_feat(...)
    print(f"点云 BEV 特征: {pts_feature.shape}")  # [B, 256, 180, 180]
    ...

# 2. 可视化 BEV 特征
import matplotlib.pyplot as plt
bev_feature = x[0].cpu().numpy()  # 取第一帧
bev_sum = bev_feature.sum(axis=0)  # 沿通道求和
plt.imshow(bev_sum)
plt.savefig('bev_feature.png')

# 3. 减少数据快速迭代
# 修改 dataloader
train_dataloader = dict(batch_size=1, ...)
```

---

## 附录 A: 常见问题

**Q1: CUDA out of memory**
```python
# 方案1: 减小 batch size
train_dataloader = dict(batch_size=2, ...)

# 方案2: 启用混合精度
optim_wrapper = dict(..., amp=True)

# 方案3: 减小 BEV 分辨率
view_transform = dict(xbound=[-54, 54, 0.6], ...)  # 格子更大
```

**Q2: 训练 loss 不下降**
```python
# 检查:
# 1. 学习率是否合适
# 2. 数据标注是否正确
# 3. 损失权重是否合理
```

**Q3: 多 GPU 训练失败**
```bash
# 检查:
# 1. NCCL 是否正确安装
# 2. 网络是否正常
# 3. 使用正确的启动脚本
bash tools/dist_train.sh configs/xxx.py 8
```

---

## 附录 B: 参考资源

- [BEVFusion 论文](https://arxiv.org/abs/2205.13542)
- [BEVFusion 官方代码](https://github.com/mit-han-lab/bevfusion)
- [MMDetection3D BEVFusion](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion)
- [nuScenes 数据集](https://www.nuscenes.org/)
- [LSS 论文](https://arxiv.org/abs/2008.02759)

---

*文档版本: v1.0*
*更新日期: 2024*
*参考: BEVFusion 论文, MMDetection3D 官方代码*
