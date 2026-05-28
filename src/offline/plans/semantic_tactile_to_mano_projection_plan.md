# 语义约束 Tactile-To-MANO 投影计划

## Summary

当前 `src/offline` 中的 tactile sensor layout 到 fitted MANO mesh 投影，本质是 reference pose 下的位置几何投影：

```text
XHand tactile taxel 3D point
  -> robot_to_mano
  -> 全局最近 MANO triangle
  -> barycentric weights to 3 MANO vertices
```

这可以生成可用的静态 `sensor_to_mano_vertex_weights`，但它没有强制保持手指语义。由于 XHand 和 MANO 的手部构型不同，尤其是 XHand 除大拇指外各手指长度更接近，而 MANO 中人手各手指长度差异更明显，全局最近点投影会出现一个问题：**XHand 上命名为 fingertip tactile sensor 的点，映射到 MANO 后不一定仍位于对应手指的 fingertip 区域。**

下一步目标是引入语义约束，使 tactile taxel 不只按空间最近投影，还按 finger / segment / fingertip 语义投影。

## Implementation Status

- 第一版 `semantic-distal` 已实现：同名 finger + distal/tip candidate faces 内最近点投影。
- 第二版 `semantic-normalized` 已实现：先计算 XHand finger-local normalized coordinate，再映射到 MANO 同名 finger segment，并继续复用 `semantic-distal` 的 candidate faces 与 fallback 机制。
- 第三版 `block-preserving` 已实现：按 `layout.sensor_slices` 中的 sensor block 单独处理，先保持 block 内 taxel 的局部 2D 分布，再 snap 到 MANO 同名 finger distal/tip candidate faces。
- 第四版 `graph-preserving` 已实现：在 `block-preserving` 结果上建立 block 内 kNN graph，增加近邻边长、邻接关系、局部角度和 Laplacian 结构保持约束。
- `graph-preserving-no-block` 已设为当前默认候选方案：禁用 `block-preserving`，直接在 `semantic-normalized` 结果上做 kNN graph refinement。
- 当前模式：`global-nearest`、`semantic-distal`、`semantic-normalized`、`block-preserving`、`graph-preserving`、`graph-preserving-no-block`。

## Key Observation

现有 pipeline 已经有两类可复用信息：

- `offline_shape_alignment` 已经建立 XHand 21 语义关键点与 MANO 21 语义关键点的对应关系。
- `tactile_layout_3d_projection` 读取 XHand tactile XML 时已经保留 sensor name / finger name / semantic region。

因此，问题不是完全缺少语义信息，而是当前 `tactile_to_mano_projection` 在最后投影阶段没有使用这些语义约束。

## Proposed Direction

保留当前全局最近点投影作为 baseline，新增语义约束投影模式。

### 1. Semantic-Constrained Nearest Projection

第一版先实现保守 MVP：

```text
taxel sensor name / semantic_region
  -> infer finger: thumb / index / middle / ring / pinky
  -> infer intended region: distal / fingertip
  -> restrict candidate MANO triangles to same finger + distal/tip region
  -> nearest triangle inside semantic candidate set
  -> barycentric weights
```

例如：

```text
right_index_tactile_sensor
  -> MANO right index finger
  -> index DIP-to-tip / fingertip patch
  -> only project to this semantic region
```

这样可以避免 `index fingertip` taxel 因为几何上更近而落到 middle/ring 或 index 中段区域。

### 2. Normalized Finger-Segment Projection

第二版再做更稳的归一化语义段映射。

对每根手指使用 21 语义关键点建立局部坐标：

```text
MCP -> PIP -> DIP -> tip
```

对 XHand taxel 计算 finger-local 坐标：

```text
u = 沿语义骨架方向的位置比例
v = 指宽方向偏移
w = 掌心/手背方向偏移
```

然后映射到 MANO 对应语义段：

```text
XHand index distal segment u=0.8
  -> MANO index distal segment u=0.8
  -> project to MANO index distal/tip region
```

这比单纯最近点更能抵抗 robot hand 和 human hand 的长度比例差异。

### 3. Block-Preserving Projection

第三版解决的是 sensor block 内部形状被最近点投影打散的问题。当前 XHand 的 block 定义为 `layout.sensor_slices` 中的单个 tactile sensor，例如：

```text
right_index_tactile_sensor -> 120 taxels
```

处理流程：

```text
同一个 sensor block 的 XHand taxels
  -> robot_to_mano
  -> PCA / finger DIP-to-tip axis 建立 block-local (u, v)
  -> block centroid 使用 semantic-normalized 找到 MANO target center
  -> 按 MANO/XHand distal length 比例重建整个 block
  -> snap 到同名 finger distal/tip candidate faces
  -> barycentric weights
```

它的固定 fallback 链为：

```text
block-preserving -> semantic-normalized -> semantic-distal -> global-nearest
```

JSON / NPZ 会额外记录 `block_ids`、`block_local_uv`、`block_target_points`、`block_similarity_scale`、`block_fallback_reasons` 和 `block_layout_preservation`。这里的取舍是：它优先保留同一 tactile sensor block 内部的相对布局，因此 nearest distance 指标不一定优于 `semantic-distal`。

### 4. Graph-Preserving Projection

第四版在 `block-preserving` 基础上进一步约束 block 内局部拓扑。对每个 sensor block 建立 kNN graph：

```text
block-local taxel coordinates
  -> kNN graph
  -> block-preserving surface points as initialization
  -> edge length / adjacency / local angle / Laplacian diagnostics
  -> light graph refinement + surface snap
  -> barycentric weights
```

当前实现采用保守策略：

- `k=4`，只在同一个 `sensor.name` block 内建图。
- 优化项包含 data anchor、kNN edge-length spring、graph Laplacian preservation。
- 每轮更新后 snap 回同名 finger distal/tip candidate faces。
- 如果 refinement 后 graph preservation score 变差，则拒绝该 refinement，回退到 `block-preserving`。

固定 fallback 链：

```text
graph-preserving -> block-preserving -> semantic-normalized -> semantic-distal -> global-nearest
```

JSON / NPZ 额外记录：

```text
graph_knn_indices
graph_edge_length_error
graph_neighbor_mismatch
graph_angle_error_rad
graph_laplacian_error
graph_fallback_reasons
```

## MVP Implementation Plan

### New Projection Modes

在 `offline-tactile-projection project-xhand-tactile` 中新增参数：

```bash
--projection-mode global-nearest|semantic-distal|semantic-normalized|block-preserving|graph-preserving|graph-preserving-no-block
```

默认可先保持 `global-nearest`，方便和已有结果对比。确认效果后再考虑把默认值切到 `semantic-distal`。

### MANO Semantic Region Builder

新增一个 MANO semantic region 构建器：

```text
MANO mesh + MANO 21 keypoints
  -> per-finger candidate vertices / faces
```

第一版可使用简单规则：

- 对每个 MANO vertex，计算它到 5 根手指骨架线段的距离。
- 将 vertex 分配给最近的 finger。
- 对 fingertip/distal region，只保留靠近 `DIP -> tip` 或 `tip` 附近的 vertices。
- 一个 face 只要三个顶点中有足够多顶点属于该 semantic region，就纳入候选 face。

### XHand Taxel Semantic Assignment

当前 XHand tactile layout 已有：

```text
right_thumb_tactile_sensor
right_index_tactile_sensor
right_mid_tactile_sensor
right_ring_tactile_sensor
right_pinky_tactile_sensor
```

需要统一命名：

```text
mid -> middle
```

并记录每个 taxel 的语义投影目标：

```text
finger = thumb / index / middle / ring / pinky
segment = distal / fingertip
```

### Projection Report

JSON 中需要同时记录：

- `projection_mode`
- 每个 sensor 的 candidate face count
- global nearest distance
- semantic constrained nearest distance
- semantic projection 是否 fallback
- 每个 sensor 的 mean / max distance
- 每个 taxel column sum 是否接近 1

推荐保留 baseline 对比：

```text
global_nearest_distance_mm
semantic_nearest_distance_mm
semantic_distance_delta_mm
```

### Visualization

PNG 可视化中建议区分：

- colored taxel points
- semantic projected points
- optional faint line from taxel to projected point
- fallback taxels 用特殊颜色或更大的点标记

第一版先继续沿用当前 2D software render，不引入新可视化依赖。

## Fallback Rules

语义约束可能出现候选区域太小或局部几何异常，因此需要 fallback：

- 如果某个 finger 的 candidate face 数量为 0，fallback 到该 finger 全段。
- 如果 finger 全段也为空，fallback 到 global nearest。
- 如果 semantic nearest distance 比 global nearest distance 大得过多，标记 warning，但不一定自动 fallback。

建议第一版 warning 阈值：

```text
semantic_distance_mm > global_distance_mm + 15
```

或：

```text
semantic_distance_mm > 20
```

阈值需要通过实际 PNG 人工验收后再固定。

## Tests

新增 `tests/test_semantic_tactile_projection.py`：

- 测试 `mid` 归一化为 `middle`。
- 构造简单 mesh + finger region，验证 semantic mode 只在候选 face 中搜索。
- 测试 candidate face 为空时 fallback 到 global nearest。
- 测试 COO 权重列和仍为 1。
- 有本地 XHand/MANO 资产时跑 smoke test；资产缺失时跳过。

## Expected Benefits

- XHand 指尖 tactile sensor 更稳定地映射到 MANO 对应手指 fingertip / distal 区域。
- 避免不同手指长度差异导致的全局最近点串区。
- 后续生成 MANO UV tactile map 时，触觉热区语义更可信。
- 保留 global nearest baseline，方便对比和回退。

## Known Limitations

- 语义约束不能保证物理接触面积完全正确，只能保证更合理的语义落点。
- MANO mesh 没有原生 tactile patch 标注，semantic region 仍是根据关键点和几何启发式推断。
- 如果 fitted MANO pose 本身和 robot tactile layout 差异过大，semantic projection 会比 global nearest 有更大的几何距离。
- 该方案仍是 reference pose 静态映射，不解决 online pose retarget 或动态接触变形。

## Recommended Next Step

当前推荐把 `graph-preserving-no-block` 作为主候选 baseline，并继续和 `graph-preserving` / `block-preserving` 同场对比：

```text
results/tactile_to_mano_projection/
  *_block_preserving_*.json/png/npz
  *_graph_preserving_*.json/png/npz
  *_graph_preserving_no_block_*.json/png/npz
```

评价时不再只看单一 graph 指标，而是分三组：

```text
A. surface_fitting
  mean / max / rms distance
  warning ratio

B. graph_preservation
  edge length error
  neighbor mismatch
  angle error
  Laplacian error

C. distribution_quality
  nearest-neighbor distance CV
  collapse ratio
  PCA coverage area
  occupied face / vertex count
```

人工重点检查每根手指的 120 个 taxel 是否语义正确、分布自然、不塌缩，并结合 JSON 中 `quality_evaluation` 判断是否需要启用 `block-preserving`。
