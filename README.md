# OmegaPlayground

[Reproduce the G1 diagonal example](examples/g1-diagonal/README.md) · [Release candidate and dependencies](RELEASE.md)

OmegaPlayground 是一个单机器人仿真作业台。选择 Campus 或 Obstacle，在二维地图上画出目标轨迹，或用键盘直接控制机器人。页面同时显示机器人头部前视相机、第三人称仿真画面和事件回执。

## 操作

- **Auto · 手绘**　在二维地图上添加、移动或删除关键点，预览平滑的 B 样条，点击「执行任务」。地图叠加红色障碍/无支撑区域和橙色 0.6 m 净空边界；预览会标红不可通行部分并禁止执行。FFPD 以 1 Hz 读取实际位姿，先调整朝向，再跟踪轨迹。
- **Manual · 键盘**　W/S 前进或后退，A/D 转向。松键将指令归零，实际停稳效果取决于策略。
- **仿真配置**　工作流（SONIC / SuperMimic）、物理本体和外观分别选择。SONIC 支持 G1；SuperMimic 支持 G1/H2/A3/SIH2300，SIH2300 可选原生/UE4/UE5 外观。装备为空手或预制盾牌+警棍。工作流通过独立按钮切换，需先暂停，并进入目标工作流保存的状态；切换场景或应用机器人配置会重建仿真并清除任务。所有浏览器共享当前环境。
- **运动指令**　SONIC 支持 Walk/Run 和速度；SuperMimic 仅提供速度，其 Joystick 没有独立步态输入。Auto 由 FFPD 控制方向，Manual 由键盘控制方向；暂停/继续在两种模式下均可见。
- **恢复**　原地恢复使用失败位置，原点恢复使用任务起点。恢复后 FFPD 从重置位置继续执行已有轨迹。

Ego View 与 Main View 显示「视频 接收 / 渲染 FPS」：接收值按浏览器收到的完整帧统计（最近 2 秒），渲染值为服务端实际输出帧率。暂停或断流后接收值会归零。Main View 使用 H.265/WebSocket，不支持解码时回退 JPEG；Ego View 当前使用 MJPEG。

## 控制与物理

FFPD 是单人的程序化 Player。Auto 由它把手绘轨迹转换为移动速度和朝向，Manual 由键盘提供指令。两种模式都经 Joystick 生成全身参考，再由 Tracker 输出关节控制，物理引擎推进实际状态。线上每个环境只有一个活动本体。

| 工作流 | Joystick 参考帧率 | Tracker 控制频率 | 可绑定的外观 |
|---|---|---|---|
| SONIC · G1 | SonicG1Joystick，30 Hz | SonicG1Tracker，50 Hz | G1 |
| SuperMimic · G1 / H2 / A3 | 对应本体的 SuperMimic Joystick，25 Hz | 同一本体的 SuperMimic Tracker，50 Hz | 对应本体的原生外观 |
| SuperMimic · SIH2300 | SuperMimicSIH2300Joystick，25 Hz | SuperMimicSIH2300Tracker，50 Hz | SIH2300、UE4 或 UE5 |

Joystick 频率在这里指生成的参考轨迹帧率，不是重新调用规划器的频率。参考轨迹按 Tracker 的 50 Hz 时钟采样。UE4 和 UE5 是 SIH2300 的外观，不另选 Tracker，也不改变控制频率。

SuperMimic Tracker 固定为 BONES 表现优选版本：G1 c45k、H2 c35k、A3 c30k、SIH2300 c35k（UE4/UE5 共用）。A3 由用户明确选择失败更少的 c30k。course 内按 validation 选 checkpoint，跨 course 选择参考了 BONES test；不把该 test 当作独立选择后评测。在线与离线读取同一清单，启动校验完整 SHA-256，未修改 Joystick 或模型参数。[版本与四本体回放结果](docs/supermimic_bones_validation.md)。

接入验收采用五层门禁：**Tracker 原生 plane 一致性 → Joystick 独立验收 → Plane 物理闭环 → 场景接触对照 → 完整导航**。前两项可独立执行，后层不能覆盖前层失败。当前已实际运行前两项，第三层执行器受前置门禁约束，第四、五层尚未接入统一执行器；生产任务完成不等于这些门禁通过。入口与证据说明见 [分层验收](docs/layered_acceptance.md)。

逐位一致性只适用于已验证的确定性运行时。旧 MJWarp 3.8 非确定性对照不能作为接入失败的证据；MJWarp 4.0 的接触/约束排序修复、GPU 重放结果与排序开销见 [deterministic 对照](docs/determinism_20261108.md)。修复在隔离环境验证，线上运行时尚未升级。

两个工作流都以 200 Hz 在 GPU 上推进物理，每个 Tracker tick 执行四个子步。SONIC G1 和 SuperMimic 仅使用 MuJoCo Warp（MJWarp）执行 GPU 物理仿真。两者都向独立的 MJWarp 渲染进程提供物理姿态。策略频率按仿真时间计，实时性能取决于每秒墙钟时间实际完成的步数和画面数。[工作流与策略交付说明](docs/supermimic.md) · [MJWarp 后端与验收](docs/mjwarp_backend.md)

## 部署与开发

在有 CUDA GPU 和已准备资产的机器上，从仓库根目录运行：

```bash
bash deploy/start.sh
curl -fsS http://127.0.0.1:7777/api/state
```

启动后默认为暂停状态。`configs/scenes/` 定义场景初态，`configs/supermimic/` 绑定模型和本体，`configs/mjwarp-assets.json` 选择已准备的物理与渲染资产。部署脚本、解释器路径和端口见[部署说明](deploy/README.md)。

## SDG 任务采集

OmegaSDG 统筹 task、LeRobot v2.1 导出和加载；Playground 的 `tools/sdg/run_episode.py` 执行独立 G1/Obstacle task，`render_episode.py` 从实际姿态生成同步 Ego RGB-D，`check_contacts.py` 检查保存状态的几何接触。原始采集保持 `omega.playground.execution/1`，发布 metadata 采用 [SDG 契约](docs/sdg_eawm_export_schema.md)，两者版本分别管理。

每次执行有独立 `sdg_task_uid`。M 个 50 Hz 控制周期记录 M+1 个实际 body/qpos/qvel 状态，包含终点；实际 body pose 使用 tracker-world XYZ 与 xyzw 四元数。仅控制帧 0、2、…、小于 M 的状态生成 25 FPS RGB-D，共 `ceil(M/2)` 对，终点不额外编码。`terminal_state.npz` 保留实际 proprio、body、qpos/qvel 和 physics step。

任务记录明确的 `end_reason`；SIGINT/SIGTERM 在控制边界封存为 aborted，异常数据不会标记为完整 RGB-D。零控制步只保留诊断，不能作为有效 episode 导出。当前任务成功条件是进入目标位置容差，不等价于机器人已停稳。接触检查覆盖 50 Hz 保存状态，不代表已检查全部 200 Hz 子步。

## 验证进度

| L20 实测项目 | 结果 |
| --- | --- |
| SONIC G1 · Campus | MJWarp 下完成 90°/180° 转向和 2 m 手绘路线，终点误差约 0.11/0.29 m；双相机 1280×720 实测约 20 FPS，尚未达到 25 FPS。 |
| SONIC G1 · Obstacle | MJWarp 下完成手绘、键盘和两种恢复；正式入口实测 24.98 FPS，实时倍率 0.999。 |
| SuperMimic G1 · Campus / Obstacle | Campus 手绘路线终点误差约 0.10 m。Obstacle 完成 90° 和 180° 转向，两种恢复方式均可继续任务。一次短程运行约 24 FPS。 |
| VibeObstacle 场景渲染 | 在 L20 实拍核对了混凝土地纹、START 透明贴花和树冠遮罩，覆盖头部前视、跟随和全场俯视。 |
| SuperMimic A3 · Obstacle | 修复场景地面碰撞掩码后，Armed 版本在 4.22 s 仿真中前进约 1.08 m，未再穿过地面。 |
| SuperMimic H2 / SIH2300 | H2 使用固定的候选 Joystick 权重。两者的零速停稳仍待验收。 |

本地实测画面和状态记录位于 `artifacts/qc/l20-workbench/`。资产已准备和短程实测分别记录，完整外观与场景组合的运动质量仍需逐项验收。

## 开发与历史资料

Playground 可独立于 ProtoMotions 安装和运行。

开发入口按职责划分：`tools/assets/` 构建离线资产，`tools/supermimic/` 负责运行与回放，`tools/qc/` 验证视频和渲染，`tests/` 按 control、maps、runtime、viewer、workbench 组织。运行 `python -m pytest tests` 和 `node --test tests/workbench/*.cjs`。场景使用发布包中的碰撞层，运行产物写入 Git 忽略的 `artifacts/`。仿真质量以实际物理状态和对应录像核查。[场景碰撞约定](docs/collision_layer.md)

开发和验证以 [MJWarp 后端说明](docs/mjwarp_backend.md) 与 [部署入口](deploy/README.md) 为准。

右侧面板分为四组：仿真配置、仿真控制、恢复设置、运行指标。仿真控制默认展开，包含 Auto/Manual、任务执行、暂停/继续和速度滑轨；其他组默认折叠。

速度滑轨实时显示目标数值，停止拖动 120 ms 后自动下发，无需应用按钮；持续拖动合并请求，保留最后一次速度选择。
