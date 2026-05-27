# offline_shape_alignment 诊断子模块实现计划

## Summary

- 第一版不做 MANO shape 优化，只实现两个诊断能力：XHand 21 语义关键点推断与 MANO 关键点差距报告；MANO 基准下的尺度、朝向、左右手镜像、root frame、reference pose 一致性检查。
- 针对已观察到的 XHand / MANO 掌根展开角差异，补充一个保守的 XHand reference `qpos` 拟合步骤；该步骤只调整 robot reference pose，不求解 MANO `theta_ref`。
- 第二阶段推进 beta-only MANO shape optimization：固定 MANO template / zero pose，只优化 `beta`，输出 fitted MANO OBJ、loss JSON / PNG 和最终对齐 PNG。
- 在 beta-only 验证后，补充“强正则、受限 pose residual”的 beta+pose MVP：MANO root 固定，15 个非 root joint residual 通过 `pose_limit_rad * tanh(raw_pose)` 限幅。
- 诊断阶段保持轻依赖；beta-only shape optimization 额外依赖 PyTorch，已通过 Conda 加入 `tactile_utils` 环境。
- 项目使用 Conda 环境 `tactile_utils`；后续依赖优先通过 `conda install -n tactile_utils <package>` 加入该环境。
- 坐标统一不修改原始资产；计算 `robot_to_mano` 相似变换，并把它用于 JSON 报告、PNG 可视化和后续模块复用。

## Key Changes

### Python 包与接口

新增 `src/offline_shape_alignment/` Python 包，提供库 API 和 CLI：

- `infer_xhand_semantic_keypoints(urdf_path, side, qpos=None)`：基于 XHand URDF FK 生成 21 个语义关键点。
- `load_mano_reference(side, mano_root)`：读取 MANO pkl 的 `v_template` / `f` / `J` / `kintree`，不依赖 `scipy` / `chumpy`；自动推断 5 个 fingertip vertex。
- `diagnose_alignment(robot_keypoints, mano_keypoints, robot_mesh, mano_mesh)`：计算 `robot_to_mano`、关键点距离、尺度比例、镜像嫌疑、pose mismatch 指标。
- `fit_xhand_reference_pose(urdf_path, side, mano_keypoints, ...)`：在 XHand URDF 关节 limit 内搜索少量可解释的掌根 / 拇指参考关节，生成更接近 MANO reference 的 robot `qpos`。
- `load_mano_beta_model(side, mano_root)`：读取 MANO `v_template` / `shapedirs` / `J_regressor`，构建 PyTorch beta-only forward。
- `fit_mano_beta_to_xhand(side, ...)`：把 fitted XHand reference mesh 通过 `robot_to_mano` 放入 MANO frame，固定 MANO pose，只优化 `beta`。
- `fit_mano_beta_pose_to_xhand(side, ...)`：先跑 beta-only 初始化，再联合优化 `beta` 和受限 MANO pose residual；输出 pose residual label / axis-angle / norm，便于检查是否跑飞。
- `render_alignment_report(...)`：输出一张 MANO 基准下的 PNG，对比 MANO mesh/keypoints、对齐后的 XHand mesh/keypoints 和对应连线。

### CLI

新增 CLI：`offline-shape-alignment diagnose-xhand-reference`

- 参数：`--side left|right|both`、`--xhand-root`、`--mano-root`、`--results-root`、`--timestamp`、`--out-json`、`--out-png`。
- 默认 XHand 路径使用现有 `assets/hands/xhand/...`，MANO 路径使用 `assets/hands/mano/...`。
- 输出 JSON + PNG 到 `results/offline_shape_alignment/`。

新增 CLI：`offline-shape-alignment fit-xhand-reference-pose`

- 参数：在诊断参数基础上增加 `--iterations`、`--initial-step-rad`、`--min-step-rad`、`--out-pose-json`。
- 输出 baseline 诊断、拟合后的诊断、PNG 对比图，以及可复用的 XHand robot reference `qpos` JSON。
- 该 CLI 不修改 URDF / STL / MANO pkl，只在诊断过程中把 fitted `qpos` 应用于 XHand FK 和 mesh assembly。

新增 CLI：`offline-shape-alignment fit-mano-shape`

- 参数：`--side left|right|both`、`--iterations`、`--robot-surface-points`、`--mano-surface-points`、`--learning-rate`、`--beta-l2-weight`、`--key-decay-steps`、`--reference-pose-iterations`、`--no-fit-reference-pose`、`--out-json`、`--out-png`、`--out-loss-png`、`--out-obj`。
- 默认 CPU 友好配置为 `1000` iterations、`2048` robot surface points、`2048` MANO surface points；计划级重跑可以手动使用 `10000` / `4096`。
- 输出 aggregate JSON、fitted MANO OBJ、最终 fixed-frame PNG 和 loss curve PNG 到 `results/offline_shape_alignment/`。

新增 CLI：`offline-shape-alignment fit-mano-shape-pose`

- 参数：在 `fit-mano-shape` 基础上增加 `--beta-init-iterations`、`--pose-learning-rate`、`--pose-limit-rad`、`--pose-l2-weight`。
- 默认先跑 `400` 步 beta-only 初始化，再跑 `600` 步 beta+pose；pose residual 默认限制在单轴 `0.25rad` 内。
- 输出 aggregate JSON、fitted MANO OBJ、最终 fixed-frame PNG 和 loss curve PNG；JSON 中包含 beta-only summary 与 beta+pose summary，方便对比联合优化是否真正改善。

### XHand 21 点规则

- `wrist` = `{side}_hand_link` root origin。
- Thumb：`thumb_bend_joint`、`thumb_rota_joint1`、`thumb_rota_joint2`、`thumb_rota_joint3`。
- Index：`index_joint1`、`index_joint2`、`index_rota_joint3`，其中 `index_dip` 用 `index_joint2 -> tip` 的 virtual keypoint。
- Middle / Ring / Pinky：`joint1`、`joint2`、`joint3`，其中 `dip` 用 `joint2 -> tip` 的 virtual keypoint。
- URDF 中的 `mid` 输出统一命名为 `middle`。
- 关键限制：URDF 命名能解决语义分类，但不能完全保证 MANO 语义等价；自动生成结果必须通过可视化和报告检查。

### MANO 21 点规则

- 使用 MANO 16 skeleton joints：`0=wrist`，`1-3=index`，`4-6=middle`，`7-9=pinky`，`10-12=ring`，`13-15=thumb`。
- 五个 fingertip 从 template mesh 自动推断：沿每根手指 `pip -> dip` 方向，选择 distal projection 最大的 vertex；报告中记录 vertex index。
- MANO reference 使用 pkl 中的 template / zero pose；第一版不求解新的 `theta_ref`。

### 对齐诊断逻辑

- 用 Umeyama / Kabsch 相似变换把 XHand 21 点对齐到 MANO 21 点，禁止 reflection；另算一次允许 reflection 的误差用于镜像嫌疑判断。
- JSON 报告包含 raw / aligned keypoints、每点距离 mm、mean / max / rms、scale、4x4 `robot_to_mano`、raw bounds、basis determinant、mirror / unit / pose status。
- 默认 warning 阈值：`mean_distance_mm > 25` 或 `max_distance_mm > 50` 标记 reference pose / semantic mismatch；reflection RMS 比 proper RMS 低 20% 以上标记 mirror suspect；scale 超出 `[0.5, 2.0]` 标记 unit suspect。

## Test Plan

- 新增 `tests/test_offline_shape_alignment.py`。
- 测试 MANO reader 在没有 `scipy` / `chumpy` 时仍能读取本地 MANO pkl，并得到 `778` vertices、`1538` faces、`16` joints。
- 测试 XHand 左右手均能生成完整 21 点，label 顺序稳定，路径中括号 / 空格不影响读取。
- 测试 alignment report 对 XHand right + MANO right 产生有限的 `robot_to_mano`、非负距离、合理 scale，并导出 JSON / PNG。
- 测试 XHand reference pose fit 至少不劣化 objective，并能用 fitted `qpos` 重新生成诊断报告。
- 测试 MANO beta-only reader 能从 pkl 读取 `shapedirs` / `J_regressor`，并生成 `beta -> vertices / joints / 21 keypoints`。
- 测试 MANO zero-pose LBS 与 beta-only forward 等价，避免 pose forward 引入额外漂移。
- 测试 surface sampling deterministic，固定 seed 下采样稳定，barycentric 权重合法。
- 测试 beta-only optimizer smoke：小采样 / 少迭代下能输出有限 `beta`、fitted MANO mesh 和 fixed-frame report。
- 测试 beta+pose optimizer smoke：小采样 / 少迭代下能输出有限 `beta`、`15x3` pose residual，并满足 `pose_limit_rad`。
- 测试 CLI smoke：资产存在时跑 `diagnose-xhand-reference --side right`；资产缺失时跳过，保持仓库在无本地资产环境也可测试。
- 测试 CLI smoke：资产存在时跑 `fit-xhand-reference-pose --side right`，并确认 JSON / pose JSON / PNG 均写出。
- 测试 CLI smoke：资产存在且 PyTorch 可用时跑 `fit-mano-shape --side right`，并确认 JSON / PNG / loss PNG / OBJ 均写出。
- 测试 CLI smoke：资产存在且 PyTorch 可用时跑 `fit-mano-shape-pose --side right`，并确认 JSON / PNG / loss PNG / OBJ 均写出。
- 测试命令在 `tactile_utils` Conda 环境中执行：

```bash
conda activate tactile_utils
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Assumptions

- 第一版只支持 XHand，不扩展 Inspire；但 API 命名保留 `robot_keypoints`，方便后续接 Inspire。
- “统一处理”指输出并应用 `robot_to_mano` 变换，不重写 URDF、STL、MANO pkl 或 UV OBJ。
- XHand reference 使用 `qpos={}`，即 URDF zero pose。
- 如果诊断提示 pose mismatch，当前实现可以保守搜索新的 XHand robot reference `qpos`；MANO `theta_ref` 仍固定为 pkl template / zero pose。
- 当前 beta-only MANO shape optimization 作为第二阶段 baseline；受限 beta+pose MVP 在此基础上只优化小范围 non-root pose residual，不引入 `scipy` / `trimesh`。
- beta+pose MVP 只允许小的 non-root pose residual；root frame 仍由诊断阶段的 `robot_to_mano` 固定，不优化 MANO global rotation / translation。

## Beta-Only Shape Optimization

诊断子模块验收后，先进入 beta-only 离线 shape optimization：

```text
MANO vertices       : 778
MANO faces          : 1538
Robot surface points: 4096
MANO surface points : 4096
Optimization iters  : 10000 for plan-scale rerun; 1000 for current CPU-friendly default
Keypoint weight     : w(t) = max(0, 1.0 - t / 2500)
```

优化目标保持为：

$$
L_{align} = L_{CD} + w(t)L_{key}
$$

其中 `L_CD` 用于表面对齐，`L_key` 使用诊断阶段确认过的 21 个语义关键点。

当前实现补充一个 `beta` L2 regularization，防止 MANO shape 在机械手几何上过度外推。MANO pose 固定为 template / zero pose，XHand 使用 fitted robot reference `qpos` 后再通过 `robot_to_mano` 进入 MANO frame。

## Limited Beta + Pose Optimization

beta-only 通过后，进入受限 beta+pose MVP：

```text
Beta init iters      : 400
Pose joint count     : 15 non-root MANO joints
Pose residual limit  : pose = 0.25rad * tanh(raw_pose)
Pose optimization    : beta + pose residual
Root pose            : fixed zero
Regularization       : beta L2 + pose L2
```

优化目标为：

$$
L = L_{CD} + w(t)L_{key} + \lambda_\beta||\beta||^2 + \lambda_\theta||\theta_{res}||^2
$$

其中 `theta_res` 是受限的 MANO non-root pose residual。该阶段必须同时报告 beta-only 与 beta+pose 指标；如果 Chamfer 降低但 keypoint / pose residual 明显异常，需要回退到 beta-only 或收紧 `pose_limit_rad` / 提高 `pose_l2_weight`。
