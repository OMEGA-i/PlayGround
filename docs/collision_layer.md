# 场景唯一碰撞层

Vibe-Campus 的生产端负责发布碰撞资产；OmegaPlayground 只消费，不按策略改写几何。

## 交付契约

`scene.usdc` 仍是视觉与场景组合的入口，引用独立的 `collisions.usdc`。
`meta.json` 增加：

```json
{"collision": {"schema": "omega.scene-collision/1", "entry": "collisions.usdc", "sha256": "<SHA256>"}}
```

碰撞层采用米、Z-up，只有静态碰撞体，绑定场景物理材质。`SHA256SUMS.json` 必须覆盖
碰撞层、入口和 meta。入口组合出的碰撞体路径与变换必须和独立层一致。
新场景身份由完整 checksum manifest 计算，修改 collision layer 会使旧地图、旧计划失效。

当前消费端支持 box、sphere、capsule、cylinder、ellipsoid、明确凸包。
普通非凸 mesh 明确拒绝，避免 MuJoCo 将其静默凸包化；不做运行时网格重建、道路加厚、
高度场替换、填洞或降低支撑面。几何表示的改变必须由场景生产端发布新 revision。

## 运行配置

场景运行统一采用 MuJoCo-Warp 原生接触，以及 analytic box-box manifold。
该选择属于仿真配置，不由策略选择。机器人控制器、频率、PD 和其他求解参数仍由交付包声明；
新接触配置下的策略质量需要分别验收，不能沿用训练或旧场景结果。

`SceneGround` 直接导入碰撞层，保留形状、变换、材料；独立静态查询模型由同一 builder 建立。
高度查询使用原始碰撞形状的向下射线，运行观测从机器人高度向下查询，以区分桥面与桥下。
不存在支撑时明确报错，不返回虚构地面高度。

正式 CLI 不再接受 `--road-boxes`、`--road-strips`、`--hfield-depth`。
旧几何变换实现位于 `legacy_ground.py`，仅供历史记录的无动力学回放使用。
旧 revision 不允许启动新模拟；旧录像仍能使用原来的资产复现。

## 验证

本地回归覆盖 USD 材质/尺寸保留、多层射线、缺失支撑、碰撞文件修改与场景身份、
Cube 道路地图烘焙。发布资产后在 测试主机 顺序执行启动、道路转弯、完整环线及小队测试，
保存实际接触日志和状态录像；验收结果单独记录，不引用旧 road-boxes 的通过结果。

## 当前冻结资产：Vibe-Campus revision 6

正式配置指向 `artifacts/scenes/VibeCampus-v20260923-collision-v4/`，内容等于生产端冻结的
`img2threejs/data/assets/VibeCampus-v20260923-collision-v8/`。构建候选目录仅保留历史证据，
运行时只加载这一份 `collisions.usdc`。

- 6451 个静态形状：619 box、432 cylinder、5400 闭合凸 mesh。
- 道路顶面约 0.23 m，厚度 0.20 m；交叉口使用连续厚箱体，剩余铺装离线分块。
- 道路分块统一水平重叠 1 mm，避免序列化和 float32 精确边界漏交；顶面高度不变。
- Collision SHA256：`7b5adcb24af8d34031094f40c45d1f2f4fcee76b9ab64e0ba67dfb2d2e6c2986`。
- SHA256SUMS.json 文件 SHA256：`a0339f76a52749eabf626a24216b96add6ec804012858789f4e999e4fdfa73c3`。
- manifest identity：`752186cd58cc14179944a4a7c4e2f3b3c0b31887fcd0373b3928801237a48770`。
- 地图：`artifacts/maps/VibeCampus-v20260923/collision-v4/`。
- 计划：`artifacts/plans/sonic_campus_{ring,north_corner}_collision_v4.json`，环线 382 m，转弯路线 40 m。
- 生产端代码提交 `f0ea9e9`。

资产是结构级碰撞近似：小型建筑装饰被剔除，闭合非凸障碍离线分解，开放表面厚化。
边界坐标采用毫米格，简化容差 1 cm；来源、近似和剔除清单在包内
`qc/collision_geometry.json`、`qc/collision_omissions.json`、`qc/collision_sources.json`。
23896 个视觉 prim 保持不变。几何 QC 不等于任意地形运动能力验收。

### 当前验证

- `artifacts/qc/collision_layer_v4/reference_support.json`：环线、单人转弯、三人转弯共 10845 个参考点，支撑均为 0.23 m，无高度跳变。
- `artifacts/qc/collision_layer_v4/support_gpu.json`：测试主机 十点薄鞋底落体通过，最大高度误差 0.108 mm，东北路口约 0.036 mm。
- `artifacts/qc/collision_layer_v4/old_pose_gpu.json`：历史 526.615 s 故障姿态 CPU/Warp 支撑法线均为 +Z，最深接触 -21.59 mm，通过。
- `artifacts/runs/collision_v4_routes/campus_corner/metrics.json`：单人转弯 64.51 s 完成，无倒地、无中止，位置误差 P95 0.245 m。
- `artifacts/runs/collision_v4_routes/campus_ring/metrics.json`：382 m 环线 586.665 s 完成，117333 个物理步，无倒地/中止，覆盖 100%，位置误差 P95 0.184 m、最大 0.301 m，最低根部离地 0.698 m。
- 三人转弯 60 s 通过，12000 个物理步均检查，无倒地/中止/机器人间接触，最小间距 2.845 m，单人误差 P95 最大 0.212 m。
- 六人、九人各 60 s 通过，均无倒地/中止/机器人间接触；最小间距分别 2.902/2.924 m，单人误差 P95 最大 0.276/0.295 m。完整检查见 `artifacts/runs/collision_v4_squads/report.json`。
- 保存状态回放：`artifacts/qc/collision_layer_v4/ring_1fps.mp4`（全环线 587 帧，1 fps）、`squad_turn_10fps.mp4`（35–60 s，251 帧，10 fps）。已检查环线 6 个关键位置和三人转弯逐秒抽帧；画面无调试横线，人物保持直立，物理通过结论来自完整日志。
- 本地回归 `artifacts/collision_layer_final_tests.log`：48 passed；测试主机 接触和选择性重置回归 7 passed。

### 保留的失败证据

第一消费候选 `collision-v1`（生产 v5）的东北路口 `[122.75, 3.75]` 下陷 16.1 mm，
超出 3 mm 阈值。生产端通过延长道路主体和完整路口箱体修复。
第三消费候选 `collision-v3`（生产 v7）的精确 `y=8` 分块接缝存在浮点缝隙，
射线在 x=21.25/22.75/24.25 查询到基础地面 0.02 m，路线预检阻断；生产 v8 的统一重叠修复。
这些旧候选不得作为已验收资产使用。

历史 526.615 s 故障姿态在第三候选上的 CPU/Warp 接触查询法线均为 +Z，最深 -21.59 mm，
未出现旧 hfield 的 -Z / -111.2 mm 异常。这不代表上游 heightfield bug 已修复。

```bash
.venv/bin/python -m omega_playground.eval.scene_support_probe \
  --scene artifacts/scenes/VibeCampus-v20260923-collision-v4 \
  --device cuda:0 --output artifacts/qc/collision_layer_v4/support_gpu.json

.venv/bin/python -m omega_playground.eval.route_gate \
  --case campus_corner --case campus_ring --device cuda:0 \
  --output artifacts/runs/collision_v4_routes

.venv/bin/python -m omega_playground.eval.squad_gate \
  --device cuda:0 --output artifacts/runs/collision_v4_squads
```
