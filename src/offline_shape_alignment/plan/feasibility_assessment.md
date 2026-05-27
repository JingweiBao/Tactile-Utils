# shape_alignment_dexhand2mano 可行性评估

## 目标范围

本模块目标是离线求解一个形态上尽量贴近灵巧手的 MANO hand，用于后续把 xhand / Inspire 的触觉分布、传感器区域和 MANO 表面建立更稳定的几何对应。

本阶段只解决形态对齐，不直接解决 tactile sensor 到 MANO UV 的投影、每帧触觉值保存或 Being-H0.5 数据接入。

## 当前已有资源

| 类别 | 当前状态 | 路径 | 评估 |
|---|---|---|---|
| MANO 参数模型 | 已有左右手 pkl | `assets/hands/mano/mano_v1_2/models/MANO_LEFT.pkl`, `assets/hands/mano/mano_v1_2/models/MANO_RIGHT.pkl` | 足够支撑 beta/pose 形态优化，但需要现代 PyTorch loader 或兼容旧 MANO webuser 代码。 |
| MANO UV atlas | 已有左右手 OBJ | `assets/hands/mano/MANO_UV_left.obj`, `assets/hands/mano/MANO_UV_right.obj` | 足够用于后续把拟合后的 MANO mesh 继承 UV；本阶段只做 3D shape alignment 时不是核心瓶颈。 |
| Inspire URDF | 已有左右手 URDF | `assets/hands/inspire_hand_dexsuite/inspire_hand_left.urdf`, `assets/hands/inspire_hand_dexsuite/inspire_hand_right.urdf` | 足够读取 link / joint tree / mimic joint / mesh 引用。 |
| Inspire mesh | 已有 collision OBJ 和 visual GLB | `assets/hands/inspire_hand_dexsuite/meshes/` | collision OBJ 可直接采样；visual GLB 更完整但当前项目未引入 GLB mesh loader。 |
| XHand URDF | 已有左右手 URDF | `assets/hands/xhand/Xhand-urdf/xhand1_left(1)/urdf/xhand_left.urdf`, `assets/hands/xhand/Xhand-urdf/xhand1_right(1)/urdf/xhand_right.urdf` | 足够读取完整手部结构。路径包含空格和括号，实现时要统一做 Path 处理。 |
| XHand mesh | 已有 STL mesh | `assets/hands/xhand/Xhand-urdf/.../meshes/` | 足够进行 robot surface sampling。 |
| XHand tactile 资料 | 已有触觉点 JSON / XML / PDF | `assets/hands/xhand/tactile sensor/` | 对本模块不是必需资源，但后续 tactile-to-MANO 投影会用到。 |
| 现有代码 | 已有 URDF/mesh/3D 可视化基础 | `src/tactile_layout_3d_projection/` | 可复用 URDF 解析、mesh 读取和 transform 逻辑，但 shape optimization 应独立成新模块。 |

## 可行性结论

该计划在本项目中 **可实现**，但不是“直接跑现有代码即可”的状态。资源层面已经具备 MANO、Inspire、XHand 的主要几何文件；主要缺口在算法实现依赖、语义 keypoint 标定文件，以及左右手/尺度/姿态规范。

建议把 MVP 定义为：

1. 加载指定 dexhand 的 reference pose mesh。
2. 加载 MANO 左/右手模型，在固定 pose 下只优化 shape parameter `beta`。
3. 使用 robot surface points 与 MANO surface points 的 Chamfer loss。
4. 加入手工配置的 21 个 robot-to-MANO semantic keypoints。
5. 输出 `beta_star`、拟合后的 MANO OBJ、loss 曲线和对齐可视化。

## 当前缺失资源与配置

| 缺失项 | 为什么需要 | 建议保存位置 |
|---|---|---|
| `robot_mano_keypoint_mapping.yaml` | 定义 wrist、MCP、PIP、DIP、tip 在 robot link/local frame 与 MANO joint/vertex 上的语义对应；这是防止串指和错误局部最优的关键资源。 | `src/shape_alignment_dexhand2mano/config/` 或 `assets/hands/<hand>/calibration/` |
| `robot_reference_pose.yaml` | 固定 Inspire/XHand 的对齐姿态；不同 reference pose 会改变 surface sampling 和 keypoint 位置。 | `assets/hands/<hand>/calibration/` |
| `mano_reference_pose.yaml` | 固定 MANO 的对齐姿态，例如 open hand / canonical flat hand。 | `assets/hands/mano/calibration/` |
| 统一尺度与坐标约定 | URDF mesh、MANO mesh 的单位和朝向需要显式记录，否则 Chamfer loss 会被尺度/朝向误差主导。 | `src/shape_alignment_dexhand2mano/config/alignment_conventions.yaml` |
| MANO 现代 loader | 原始 MANO webuser 代码较旧，工程上最好转成 PyTorch forward，便于优化 beta。 | 新模块代码或依赖 `smplx` / `manopth` |
| mesh sampling / Chamfer 实现 | 当前项目依赖只有 numpy、Pillow、PyYAML，缺少可微优化和 surface sampling 工具。 | 新模块代码；必要时补充依赖 |

## 依赖评估

当前 `pyproject.toml` 依赖很轻，只包含：

```text
numpy
Pillow
PyYAML
```

离线 shape alignment 至少需要补充以下能力：

- `torch`：优化 `beta` 与计算可微 loss。
- `scipy` 或自写 KD-tree/nearest neighbor：用于评估与非可微诊断。
- `trimesh`：读取 OBJ/STL/GLB、surface sampling、导出 OBJ；也能减少本项目重复造 mesh 轮子。
- 可选 `smplx` 或 `manopth`：加载 MANO pkl 并做 forward；如果不用外部包，则需要自己实现 MANO LBS。

如果只做一个离线工具，建议先不要把 heavy dependency 强塞进主功能；可以把 shape alignment 的依赖放到 optional extra，例如 `tactile-utils[shape-align]`。

## 风险点

- **最高风险：21 个语义关键点配置。** 没有这个文件时，只靠 Chamfer Distance 会很容易出现手指对应错位、左右镜像错位、掌心/手背混淆。
- **中高风险：MANO loader 兼容性。** 现有 MANO pkl 是可用资源，但原始 webuser 代码老旧，直接运行可能遇到 chumpy / Python 版本问题。
- **中等风险：Inspire visual GLB 与 collision OBJ 选择。** collision OBJ 更容易读，但 palm/base 的视觉完整性可能不如 GLB；如果使用 GLB，需要引入 trimesh 或 pygltflib。
- **中等风险：XHand 路径与 mesh 格式。** 资源路径包含空格、括号，mesh 多为 STL；代码必须用 `pathlib.Path`，不要手写字符串拼接。
- **中等风险：shape-only 对齐能力有限。** 只优化 MANO `beta` 无法完全拟合机器人手的非人体比例和机械外形，因此结果应理解为“形态接近的 MANO reference”，不是精确几何复制。

## 建议落地顺序

1. 先实现只读资产检查 CLI：确认 MANO pkl/UV OBJ、dexhand URDF、mesh 引用、左右手路径均可读。
2. 实现 robot reference mesh assembly：把 URDF link mesh 在 reference pose 下合并成一个 3D mesh。
3. 实现 MANO forward：先固定 `theta_ref`，只暴露 `beta`。
4. 增加 keypoint mapping 配置和可视化检查。
5. 跑 beta optimization，输出 fitted MANO OBJ 与对齐诊断 JSON。
6. 加人工验收图：灰色 dexhand mesh + 半透明 MANO mesh + 21 keypoints 连线。

## 本项目当前判断

MANO 相关文件 **基本够用**：已有左右手 pkl 和 UV OBJ。真正需要补的是现代 MANO forward 实现或依赖，而不是再找一套 MANO 资源。

Inspire / XHand 几何资源 **基本够用**：URDF 和 mesh 都在。真正需要补的是每种手的 `robot_mano_keypoint_mapping.yaml` 和 `robot_reference_pose.yaml`。

因此，本子模块可以启动 MVP，但第一版应先以“诊断可视化 + 可复现输出”为目标，不宜直接把输出当成稳定 MANO 投影 baseline。
