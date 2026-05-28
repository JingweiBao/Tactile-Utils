# offline_shape_alignment

`offline_shape_alignment` 用于把灵巧手 URDF 几何与 MANO hand reference 对齐，并逐步生成可用于后续触觉投影的 MANO 形状结果。

当前实现已经跑通 XHand：

- XHand 21 个语义关键点推断
- MANO 21 个语义关键点生成
- `robot_to_mano` 相似变换诊断
- XHand reference `qpos` 拟合
- MANO beta-only shape optimization
- MANO beta + 受限 pose residual 联合优化
- JSON / PNG / fitted MANO OBJ / loss curve 输出

## Quick Start

### Environment

项目使用 Conda 环境 `tactile_utils`：

```bash
conda activate tactile_utils
```

诊断功能依赖项目基础包：

```bash
conda install -n tactile_utils numpy pillow pyyaml
```

MANO shape optimization 需要 PyTorch：

```bash
conda install -n tactile_utils pytorch cpuonly -c pytorch
```

如果 CPU PyTorch 遇到 MKL runtime symbol error，保持 MKL 在 2025 以下：

```bash
conda install -n tactile_utils "mkl<2025"
```

从仓库根目录运行源码版命令时，使用：

```bash
PYTHONPATH=src python -m sub_modules.offline_shape_alignment.cli <command> ...
```

如果项目已安装为 package，也可以直接使用 console script：

```bash
offline-shape-alignment <command> ...
```

### Common Commands

1. 生成 XHand / MANO 诊断报告：

```bash
PYTHONPATH=src python -m sub_modules.offline_shape_alignment.cli diagnose-xhand-reference \
  --side both \
  --xhand-root assets/hands/xhand \
  --mano-root assets/hands/mano \
  --results-root results
```

2. 拟合 XHand reference `qpos`：

```bash
PYTHONPATH=src python -m sub_modules.offline_shape_alignment.cli fit-xhand-reference-pose \
  --side both \
  --xhand-root assets/hands/xhand \
  --mano-root assets/hands/mano \
  --results-root results
```

3. 跑 MANO beta-only shape optimization：

```bash
PYTHONPATH=src python -m sub_modules.offline_shape_alignment.cli fit-mano-shape \
  --side both \
  --xhand-root assets/hands/xhand \
  --mano-root assets/hands/mano \
  --iterations 1000 \
  --robot-surface-points 2048 \
  --mano-surface-points 2048 \
  --results-root results
```

4. 跑受限 beta + pose residual optimization：

```bash
PYTHONPATH=src python -m sub_modules.offline_shape_alignment.cli fit-mano-shape-pose \
  --side both \
  --xhand-root assets/hands/xhand \
  --mano-root assets/hands/mano \
  --beta-init-iterations 400 \
  --iterations 600 \
  --robot-surface-points 2048 \
  --mano-surface-points 2048 \
  --pose-limit-rad 0.25 \
  --pose-l2-weight 1e-3 \
  --results-root results
```

输出默认保存到：

```text
results/offline_shape_alignment/
```

## Design Goal

核心目标不是精确复制机器人手的机械外形，而是得到一个语义合理、姿态可解释、可进入 MANO UV / tactile projection 流程的 MANO reference。

因此当前设计分成四级递进：

1. 先确认语义关键点和坐标系是否可信。
2. 再调整 robot reference pose，而不是一开始就扭 MANO。
3. 再只优化 MANO `beta`，得到 shape-only baseline。
4. 最后才允许小范围 MANO pose residual，并用强正则约束它。

这个顺序的取舍是：宁愿每一步能力保守、可解释，也不要让后续优化器用 MANO pose 掩盖关键点错误、左右镜像错误或 reference pose 不一致。

## Pipeline

### 1. XHand / MANO Diagnostic

入口：

```bash
offline-shape-alignment diagnose-xhand-reference \
  --side both \
  --xhand-root assets/hands/xhand \
  --mano-root assets/hands/mano
```

实现：

- XHand 由 `xhand.py` 基于 URDF joint/link naming 和 FK 推断 21 个语义关键点。
- MANO 由 `mano.py` 读取 `v_template / f / J / kintree_table`，并自动推断 5 个 fingertip vertex。
- `alignment.py` 用 Umeyama/Kabsch 相似变换估计 `robot_to_mano`。
- `render.py` 输出 MANO frame 下的 mesh/keypoint 对比图。

取舍：

- URDF 命名可以解决语义分类，但不能完全保证 MANO 语义等价。
- 诊断阶段不修改 URDF、STL、MANO pkl，只输出变换和报告。
- reflection 默认禁止，只额外计算 reflected RMS 用于镜像嫌疑判断。

### 2. XHand Reference Pose Fitting

入口：

```bash
offline-shape-alignment fit-xhand-reference-pose \
  --side both \
  --xhand-root assets/hands/xhand \
  --mano-root assets/hands/mano
```

实现：

- `reference_pose.py` 只搜索少量 XHand 近掌根关节和拇指相关关节。
- 每个候选 `qpos` 都重新做 URDF FK，生成 XHand 21 点。
- 对齐到 MANO 后计算加权关键点误差，使用保守 coordinate descent。

默认优化关节包括：

```text
thumb_bend_joint
thumb_rota_joint1
index_bend_joint
index_joint1
mid_joint1
ring_joint1
pinky_joint1
```

取舍：

- 只调整 robot reference pose，不优化 MANO `theta_ref`。
- 不把 `joint2 / joint3` 全部放开，避免为了降低误差把手指整体弯坏。
- 中指、无名指、小指没有额外 `bend_joint`，因此只能调它们的 `joint1`。

### 3. MANO Beta-Only Shape Optimization

入口：

```bash
offline-shape-alignment fit-mano-shape \
  --side both \
  --xhand-root assets/hands/xhand \
  --mano-root assets/hands/mano \
  --iterations 1000 \
  --robot-surface-points 2048 \
  --mano-surface-points 2048
```

实现：

- `mano_torch.py` 从 MANO pkl 读取 `v_template / shapedirs / J_regressor`，实现 `beta -> vertices / joints / 21 keypoints`。
- `sampling.py` 做固定 seed 的 surface sampling。
- `shape_optimization.py` 使用 PyTorch Adam 优化 MANO `beta`。
- XHand mesh/keypoints 先经过 `robot_to_mano` 放入 MANO frame。

目标函数：

```text
L = L_CD + w(t) * L_key + lambda_beta * ||beta||^2
w(t) = max(0, 1.0 - t / key_decay_steps)
```

取舍：

- MANO pose 固定为 template / zero pose。
- 先建立 shape-only baseline，避免 MANO pose 过早吸收机器人与人手的结构差异。
- 当前不引入 `scipy / trimesh`，Chamfer 和 sampling 都在本模块内实现。

### 4. Limited Beta + Pose Optimization

入口：

```bash
offline-shape-alignment fit-mano-shape-pose \
  --side both \
  --xhand-root assets/hands/xhand \
  --mano-root assets/hands/mano \
  --beta-init-iterations 400 \
  --iterations 600 \
  --robot-surface-points 2048 \
  --mano-surface-points 2048 \
  --pose-limit-rad 0.25 \
  --pose-l2-weight 1e-3
```

实现：

- 先跑 beta-only 初始化。
- 然后从 beta-only 的 `beta` 开始继续优化。
- `pose_residual` 从 0 开始，只作用于 15 个 non-root MANO joints。
- root pose 固定，不优化 global rotation / translation。
- pose residual 通过 `pose_limit_rad * tanh(raw_pose)` 限幅。

目标函数：

```text
L = L_CD + w(t) * L_key
    + lambda_beta * ||beta||^2
    + lambda_pose * ||theta_res||^2
```

取舍：

- 允许小幅 MANO pose residual，是为了处理机械手与 MANO reference 仍存在的局部姿态差异。
- 不直接放开全 MANO pose，因为那会让优化器用手指弯曲/扭转来掩盖形态差异。
- JSON 中记录每个 pose residual 的 label、axis-angle 和 norm，用于人工检查是否跑飞。

## Key Files

```text
types.py               Canonical 21 keypoint labels and shared dataclasses
xhand.py               XHand URDF FK, mesh assembly, semantic keypoint inference
mano.py                Lightweight MANO pkl reader without scipy/chumpy runtime dependency
mano_torch.py          PyTorch MANO beta and beta+pose forward
alignment.py           robot_to_mano similarity transform and diagnostics
reference_pose.py      XHand reference qpos search
sampling.py            Deterministic surface sampling
shape_optimization.py  beta-only and limited beta+pose optimizers
render.py              alignment PNG and loss curve rendering
cli.py                 offline-shape-alignment CLI
```

## Outputs

All CLI outputs are written under:

```text
results/offline_shape_alignment/
```

Common output types:

- `*.json`: metrics, transforms, beta, pose residual, qpos, loss history
- `*.png`: final alignment visualization
- `*_loss.png`: optimization loss curve
- `*.obj`: fitted MANO mesh

The current strongest result is from `fit-mano-shape-pose`. In the latest run, beta+pose improved keypoint and surface metrics over beta-only, but `pinky_mcp` pose residual approached the configured limit. That means the result is useful, but should still be visually inspected before treating it as a stable downstream baseline.

## Environment

The project uses the Conda environment:

```bash
conda activate tactile_utils
```

Diagnostic commands need only the project base dependencies. Shape optimization requires PyTorch:

```bash
conda install -n tactile_utils pytorch cpuonly -c pytorch
```

If CPU PyTorch hits an MKL runtime symbol error, keep MKL below 2025:

```bash
conda install -n tactile_utils "mkl<2025"
```

## Tests

Run from the repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The tests cover:

- MANO pkl loading without runtime scipy/chumpy
- XHand 21 keypoint inference
- alignment JSON / PNG generation
- XHand reference pose fitting
- MANO beta forward
- zero-pose LBS equivalence with beta-only forward
- deterministic surface sampling
- beta-only optimizer smoke
- limited beta+pose optimizer smoke
- CLI smoke tests for diagnostic, reference pose, beta-only, and beta+pose commands

## Current Limitations

- Only XHand is wired end-to-end.
- URDF naming helps semantic classification but does not prove MANO semantic equivalence.
- MANO is still an anatomical hand model; it cannot exactly represent robot mechanical geometry.
- Beta-only cannot express pose differences.
- Beta+pose can reduce loss, but pose residual must be constrained and manually inspected.
- No MANO UV / tactile projection output is generated yet.
