# offline_shape_alignment 可行性评估

## 目标范围

本模块目标是先诊断并统一灵巧手 URDF 与 MANO reference 之间的语义关键点、尺度、朝向、左右手镜像、root frame 和 reference pose，再在后续阶段离线求解一个形态上尽量贴近灵巧手的 MANO hand。

第一阶段只做诊断、对齐变换输出和保守的 robot reference pose 拟合。第二阶段先推进 beta-only MANO shape optimization，并在其后补充强正则、受限 pose residual 的 beta+pose MVP；当前仍不直接解决 tactile sensor 到 MANO UV 的投影、每帧触觉值保存或 Being-H0.5 数据接入。

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

该计划在本项目中 **可实现**，但不是“直接跑现有代码即可”的状态。资源层面已经具备 MANO、Inspire、XHand 的主要几何文件；第一阶段主要缺口在轻量 MANO reference reader、XHand 语义 keypoint 推断、诊断报告与可视化，以及左右手/尺度/姿态规范。

建议把第一阶段 MVP 定义为：

1. 加载 XHand reference pose mesh 与 MANO reference mesh。
2. 基于 XHand URDF joint/link 命名和 FK 生成 21 个 robot semantic keypoints。
3. 从 MANO reference 中生成 21 个 MANO semantic keypoints。
4. 计算并输出 `robot_to_mano` 相似变换、关键点距离、尺度/镜像/姿态诊断状态。
5. 当语义关键点基本正确但掌根展开角与 MANO reference 不一致时，搜索少量 XHand robot reference `qpos`，再重新输出诊断。
6. 输出 JSON 报告、robot reference pose JSON 和 MANO 基准下的关键点/mesh 对比 PNG。

## 当前缺失资源与配置

| 缺失项 | 为什么需要 | 建议保存位置 |
|---|---|---|
| `robot_mano_keypoint_mapping.yaml` | 定义 wrist、MCP、PIP、DIP、tip 在 robot link/local frame 与 MANO joint/vertex 上的语义对应；可先基于 URDF joint/link 命名自动生成候选 mapping，再人工确认。这是防止串指和错误局部最优的关键资源。 | `src/offline_shape_alignment/config/` 或 `assets/hands/<hand>/calibration/` |
| `robot_reference_pose.yaml` | 固定 Inspire/XHand 的对齐姿态；不同 reference pose 会改变 surface sampling 和 keypoint 位置。当前 XHand 可先由 `fit-xhand-reference-pose` 输出 JSON，再人工确认后固化。 | `assets/hands/<hand>/calibration/` |
| `mano_reference_pose.yaml` | 固定 MANO 的对齐姿态，例如 open hand / canonical flat hand；第一阶段默认使用 MANO pkl template / zero pose，并在报告里记录。 | `assets/hands/mano/calibration/` |
| 统一尺度与坐标约定 | URDF mesh、MANO mesh 的单位和朝向需要显式记录；第一阶段不改原始资产，只输出并应用 `robot_to_mano` 相似变换。 | `src/offline_shape_alignment/config/alignment_conventions.yaml` |
| MANO 轻量 reference reader | 第一阶段只需读取 MANO pkl 的 template vertices、faces、J、kintree；应避免依赖旧 `chumpy` 或完整 PyTorch MANO forward。 | 新模块代码 |
| mesh sampling / Chamfer 实现 | 当前 beta-only MVP 已实现固定 seed surface sampling 和 PyTorch Chamfer loss；后续如需更大规模点云或 GLB 支持，再考虑引入 `trimesh` / KD-tree。 | `src/offline_shape_alignment/sampling.py`, `src/offline_shape_alignment/shape_optimization.py` |

## 依赖评估

当前 `pyproject.toml` 依赖很轻，只包含：

```text
numpy
Pillow
PyYAML
```

第一阶段诊断子模块保持轻依赖，不新增 `torch` / `scipy` / `trimesh`。项目使用 Conda 环境 `tactile_utils`，后续依赖优先通过 `conda install -n tactile_utils <package>` 添加。

第二阶段 beta-only shape optimization 当前已补充 PyTorch CPU 版。更完整的离线 shape optimization 至少还可能需要以下能力：

- `torch`：优化 `beta` 与计算可微 loss；当前已通过 Conda 加入 `tactile_utils` 环境。
- `scipy` 或自写 KD-tree/nearest neighbor：用于评估与非可微诊断。
- `trimesh`：读取 OBJ/STL/GLB、surface sampling、导出 OBJ；也能减少本项目重复造 mesh 轮子。
- 可选 `smplx` 或 `manopth`：加载 MANO pkl 并做 forward；如果不用外部包，则需要自己实现 MANO LBS。

诊断工具仍保持轻依赖；shape optimization 的 PyTorch 依赖放到 optional extra，例如 `tactile-utils[shape-align]`，实际运行依赖安装到 `tactile_utils` Conda 环境中。

## 风险点

- **最高风险：21 个语义关键点配置。** 没有这个文件时，只靠 Chamfer Distance 会很容易出现手指对应错位、左右镜像错位、掌心/手背混淆。URDF 命名可以解决手指和关节层级的语义分类，适合自动生成初稿；但它不能完全保证 MANO 语义等价，尤其是二关节机械手指、耦合关节或缺少真实 DIP 的结构，仍需要 virtual keypoint / 人工标定和 3D 可视化验收。
- **中高风险：MANO loader 兼容性。** 现有 MANO pkl 是可用资源，但原始 webuser 代码老旧，直接运行可能遇到 chumpy / Python 版本问题。
- **中等风险：Inspire visual GLB 与 collision OBJ 选择。** collision OBJ 更容易读，但 palm/base 的视觉完整性可能不如 GLB；如果使用 GLB，需要引入 trimesh 或 pygltflib。
- **中等风险：XHand 路径与 mesh 格式。** 资源路径包含空格、括号，mesh 多为 STL；代码必须用 `pathlib.Path`，不要手写字符串拼接。
- **中等风险：shape-only 对齐能力有限。** 只优化 MANO `beta` 无法完全拟合机器人手的非人体比例、机械外形和 palm / finger pose 差异，因此 beta-only 结果应理解为“形态接近的 MANO reference”，不是精确几何复制。
- **中高风险：pose residual 吸收形态差异。** beta+pose 可以明显降低误差，但如果 pose 约束过松，优化器会用 MANO 手指弯曲/扭转来补偿机器人与人手的结构差异；因此当前必须保留 `pose_limit_rad`、pose L2、root fixed，以及 beta-only / beta+pose 对比报告。

## 建议落地顺序

1. 先实现只读资产检查 CLI：确认 MANO pkl/UV OBJ、XHand URDF、mesh 引用、左右手路径均可读。
2. 实现 XHand reference mesh assembly：把 URDF link mesh 在 reference pose 下合并成一个 3D mesh。
3. 实现轻量 MANO reference reader：读取 template mesh、16 skeleton joints 和 kintree，不做 MANO forward。
4. 基于 XHand URDF joint/link 命名自动生成 21 keypoints，并生成 MANO 21 keypoints。
5. 计算 `robot_to_mano` 相似变换和关键点距离，输出 JSON 诊断报告。
6. 针对 reference pose mismatch，先用少量 XHand 关节做保守 `qpos` 搜索，并输出 robot reference pose JSON。
7. 加人工验收图：MANO mesh + 对齐后的 XHand mesh + 21 keypoints 连线。
8. 诊断通过后进入 beta-only optimization，输出 fitted MANO OBJ、loss JSON / PNG 与 fixed-frame PNG。
9. beta-only 通过后进入受限 beta+pose MVP，输出 pose residual label / axis-angle / norm，并和 beta-only 指标对比。

## 本项目当前判断

MANO 相关文件 **基本够用**：已有左右手 pkl 和 UV OBJ。真正需要补的是现代 MANO forward 实现或依赖，而不是再找一套 MANO 资源。

Inspire / XHand 几何资源 **基本够用**：URDF 和 mesh 都在。真正需要补的是每种手的 `robot_mano_keypoint_mapping.yaml` 和 `robot_reference_pose.yaml`。

因此，本子模块可以启动 MVP。当前目标已经推进到“诊断可视化 + 可复现 `robot_to_mano` 输出 + 可人工确认的 robot reference pose + beta-only MANO shape fitting + 受限 beta+pose fitting”。在 UV 投影接入和更多人工验收前，不宜直接把输出当成最终稳定的 tactile-to-MANO baseline。
