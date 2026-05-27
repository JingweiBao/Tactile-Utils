## 4. 离线阶段一：MANO 与 Inspire 形态对齐

### 4.1 准备数据

需要准备：

1. **MANO 标准模型**
   - vertices：778
   - faces：1538
   - UV map
   - shape parameter $\beta$
   - pose parameter $\theta$

2. **Inspire URDF 几何**
   - link mesh
   - joint tree
   - reference pose 下的 mesh surface

3. **21 个语义关键点**
   - wrist
   - 各手指 MCP / PIP / DIP 或对应机器人关节关键点
   - 各 fingertip
   - 要保证语义一致，不建议只用最近点匹配

4. **表面采样点**
   - 从 Inspire URDF mesh 采样 4096 点
   - 从 MANO mesh 采样 4096 点

### 4.2 优化 MANO shape 参数

论文使用一次性 offline shape optimization，求解最适合机器人手形态的 MANO shape：

$$
\beta^* = \arg\min_{\beta} L_{align}
$$

其中：

$$
L_{align} = L_{CD} + w(t)L_{key}
$$

- $L_{CD}$：Chamfer Distance，用于表面对齐。
- $L_{key}$：21 个语义关键点的位置差。
- $w(t)$：关键点 loss 权重，随优化逐渐衰减。

附录中给出的工程参数：

```text
MANO vertices       : 778
MANO faces          : 1538
Robot surface points: 4096
MANO surface points : 4096
Optimization iters  : 10000
Keypoint weight     : w(t) = max(0, 1.0 - t / 2500)
```

### 4.3 优化伪代码

```python
mano = load_mano_model()
robot = load_inspire_urdf()

q_ref = get_reference_pose()
theta_ref = get_mano_reference_pose()

robot_pts = sample_urdf_surface(robot, q_ref=q_ref, n=4096)
robot_kpts = compute_robot_21_keypoints(robot, q_ref)

beta = init_mano_beta()
optimizer = Adam([beta], lr=lr)

for t in range(10000):
    mano_mesh = mano.forward(beta=beta, theta=theta_ref)
    mano_pts = sample_mesh_surface(mano_mesh, n=4096)
    mano_kpts = mano.get_21_keypoints(beta, theta_ref)

    loss_cd = chamfer_distance(robot_pts, mano_pts)
    loss_key = l2_loss(robot_kpts, mano_kpts)
    w = max(0.0, 1.0 - t / 2500.0)

    loss = loss_cd + w * loss_key
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

beta_star = beta.detach()
save(beta_star, "beta_star.pkl")
```

### 4.4 离线输出

```text
calibration/
  beta_star.pkl
  mano_reference_mesh_beta_star.obj
  robot_mano_keypoint_mapping.json
  robot_reference_pose.json
```
