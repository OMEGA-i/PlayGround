# OmegaPlayground SuperMimic SDG 导出 Schema

更新：2026-11-06。设计版本：`omega.sdg.eawm/2.0-draft`。本地推理代码已改为执行完整段后重新规划；GPU闭环质量尚待回放验收。当前逐chunk原始记录版本为 `omega.playground.executed-chunk/1`，完整LeRobot分片发布与unified adapter尚未实现。

## 1 核心约定

EAWM第一阶段替代Joystick，action标签是motion reference。Tracker输出的PD target是执行记录，不是本阶段EAWM监督目标。

```text
实际state[k] + 同一物理时间的ego frame + task / command
       ↓ 机器人实际history → FK → Joystick context
Joystick → 全部未来60帧（25 FPS）
       ↓ 当前实测姿态作插值锚点
121个reference采样点（50 Hz，含t=0锚点）
       ↓ 固定 world_from_chunk 坐标变换
Tracker流式取当前+未来10帧 → 120次PD控制 → state[k+120]
       ↓ 同步获取新的video/proprio，物理时钟不在等待期间推进
下一次Joystick推理
```

不预取下一段，不沿用预测尾帧作为实测context，不在chunk边界reset物理或Tracker history。主数据单位为一次决策chunk，用physics timestamp关联输入、reference和执行反馈。1.1版本中的提前23帧请求、40帧释放、跨chunk前瞻方案废止。

## 2 帧数与执行语义

| 数据 | 当前长度 | 时间意义 |
| --- | --- | --- |
| Joystick模型窗口 | 64帧，25 FPS | 前4帧context + `out[:,4:]`的60帧未来 |
| Context | 4帧实际状态特征 | k−6、k−4、k−2、k，间隔0.04秒 |
| Reference | 121点，50 Hz | 当前锚点r[0]和未来r[1…120] |
| PD控制 | 120步，50 Hz | 区间[k,k+120)，总计2.4秒 |
| Physics | 480子步，200 Hz | 每个PD目标保持四个物理子步 |
| Tracker窗口 | 11点 | 每步current + 10 future |

原始格式记录实际`planner_future_frames`和`planned_control_steps`，不将120写死到文件读取器。当前权重/配置要求返回完整64帧，短输出直接拒绝，不能静默截断或拿旧预测补齐。

每个控制步i用reference窗口r[i:i+11]计算action[i]，推进state[k+i]到state[k+i+1]。最后一步i=119仍消费真实目标r[120]，之后才规划下一段。r[0]为时间锚点，不额外占用一次PD执行。

### 2.1 段尾流式前瞻

当前段没有下一段reference，因此在窗口超过r[120]时：重复最后姿态，并将补帧root线/角速度和joint velocity设零；`lookahead_valid[11]`标记真实点与补帧。补帧只存在于临时actor输入，不入reference队列、不增加PD次数、不作为动作标签。

这保留现有Tracker网络的11帧输入维度，但改变段尾输入分布，可能影响停步、接缝和跟踪质量。CPU结构测试不能证明物理质量，必须做同步视频/日志回放。未来如更换末端策略，发布新的`lookahead_policy`；不能悄悄生成下一段来填补前瞻。

## 3 Proprio history 与坐标

### 3.1 实测历史构造

每次物理控制返回后保存owned state快照，不能引用随后会被模拟器改写的buffer。取最近7个50 Hz状态中的隔帧4点，经现有`full_body_motion()`与`features()`进行native joints→全body FK→MotionBricks特征转换，使用本体原生joint_names。

初始或reset后不足历史时重复最早实测state，并保存`valid`与实际重复tick。不得伪造负时间实测记录。生成器末帧速度特征遵循现有特征提取器的末端规则；没有访问未来实测状态。

当前Joystick并不理解图像：它使用proprio构造的运动上下文和速度/方向条件。ego图像在同一决策物理时间采集并保存供EAWM训练，`video_conditioned=false`。仿真root pose属于privileged输入；将来真实机器人需有对应状态估计接口。

### 3.2 每段独立heading局部坐标系

新context按最新观测heading规范化；输出native reference属于该次决策的局部坐标。当前转换器的原点可能位于历史首帧，因此保留reference的当前锚点，不假设其XY恒为零。

通过reference锚点与实测当前root求固定刚体变换：

```text
R = Rz(yaw(actual_root[k]) − yaw(reference_anchor))
t = actual_root_position[k] − R @ reference_anchor_position
world_from_chunk = [[R,t],[0,1]]
p_world = R @ p_chunk + t
orientation_world = R @ orientation_chunk
v_world = R @ v_chunk
omega_world = R @ omega_chunk
```

四元数为wxyz，距离m、角度rad，列向量变换约定。joint angles/velocities不受root刚体变换影响。该变换在一个chunk内固定，新段从新的实际root重新求解，不从旧reference末端接续。

转向command作为生成条件：将desired-facing速度旋转到当前heading坐标，目标姿态heading单独条件化。正常段不再使用旧HeadingFrame逐帧slew轨迹；HeadingFrame仅保留bootstrap/reset初始化用途。

Tracker仍保留每步窗口首帧yaw锚定到实测heading的逻辑。这是低层goal构造，不修改不可变world reference。保存实际state与adapter版本可重建；生产exporter应同时记录有效首帧quaternion或完整goal输入用于审计。

### 3.3 三个坐标空间

必须区分`chunk_local`、`tracker_world`和`scene_world`。LiveView在tracker姿态上增加spawn/height offset后渲染，记录`scene_from_tracker`才能把视频与物理轨迹放在同一场景中。skin/装备外观单独标记，不改变native关节语义。

## 4 Physics timestamp

权威时间来自已完成物理步数：

```text
physics_step = control_tick * 4
timestamp = physics_step / 200 = control_tick / 50
```

这是物理步计数的逻辑时间，不是PC时间，也不是渲染完成时间。推理、磁盘写入、同步渲染等待和用户暂停均不推进它。墙钟只供性能统计/文件唯一命名，不能用于训练对齐。

每段起点为k，终点为k+120：输入state和ego图像都标记k/50；reference采样点覆盖[k,k+120]；控制区间为[k,k+120)；执行state包含两端共121点。下一段输入必须是state[k+120]和该状态重新渲染的图像。

reset产生新的物理连续段，不能跨reset切history。当前服务tick维持单调递增，reset事件不增加物理步；后续exporter通过episode_uid和episode起点映射成episode内时间，不依赖MuJoCo reset后的data.time。

## 5 Video 与 proprio 对齐

在规划边界执行render barrier：提交当前实际body poses及其physics timestamp，等待对应渲染token完成，验证返回timestamp与当前tick一致，再冻结ego JPEG与proprio history。期间不运行物理。因此相机PC延迟不会转化为物理时间错位。

边界snapshot必须有ego frame；缺失则停止该次规划并报错，不拿旧图代替。历史中的较早proprio仍带自身timestamp，不能全部标成当前时刻。

执行视频按其所渲染的物理状态tick关联，不按编码完成时间关联。reference和video之间允许真实tracking误差：reference是指令，video描述实际结果，不能用移动视频时间轴来隐藏误差。

主视频名义每两次control输出一帧（25 Hz）。决策边界额外渲染为snapshot，不重复写入主MP4。完整SDG视频索引必须记录camera/frame/PTS/capture physics timestamp；当前实现优先落盘每段边界JPEG，连续视频的逐帧索引尚待完整exporter接入，不能把MP4文件名当作时间索引。

## 6 当前原始产物

在线运行路径：`artifacts/runs/supermimic/live/<robot>-<scene>-<run>/chunk-000000/`，与同名前缀主MP4配套。

```text
manifest.json
input.npz
ego.jpg
main.jpg
reference_local.npz
reference.npz
execution.npz
```

### 6.1 manifest.json

当前实现字段：

| 字段 | 含义 |
| --- | --- |
| schema | `omega.playground.executed-chunk/1` |
| chunk_index | 本次run内递增索引 |
| start_tick / physics_start_step / timestamp | 输入快照和控制起点 |
| end_tick / physics_end_step / end_timestamp | 实际结束时刻，finish后写入 |
| planner_future_frames | 本次释放的25 FPS未来帧数，当前60 |
| planned_control_steps / executed_control_steps | 预期/实际控制步数，完整段当前120 |
| embodiment / joint_names | 本体ID与native顺序 |
| scene_from_tracker | tracker-world到渲染scene-world的固定4×4变换 |
| tracker_sha256 | 实际加载的Tracker checkpoint |
| command | vx、vy、facing |
| camera_timestamp | 输入ego对应物理时间，必须等于timestamp |
| context_source | `executed_proprio` |
| lookahead_policy | `terminal_hold_zero_velocity` |
| video_conditioned | false，旧Joystick不消费视觉 |
| status | executing/complete/reset/closed；非complete不能视作全段执行 |

### 6.2 input.npz

| 字段 | 类型/形状 | 语义 |
| --- | --- | --- |
| states | float[4,Dstate] | 实测root、joint和body状态；布局绑定Tracker schema |
| ticks | int64[4] | 实际历史样本tick，bootstrap可重复 |
| valid | bool[4] | 重复填充false |
| physics_steps / timestamp | int64[4] / float[4] | 各历史样本物理时间 |
| joint_names | string[J] | native关节顺序 |
| command_heading_delta | float | desired facing减实际root yaw |
| camera_physics_step / camera_timestamp | int / float | 同目录ego.jpg/main.jpg对应的当前状态 |

State前缀为root xyz(3)、wxyz(4)、world linear velocity(3)、world angular velocity(3)、joint position(J)、joint velocity(J)，之后为Tracker FK body数据。消费者须绑定布局版本，不能只根据数组宽度猜。

### 6.3 reference_local.npz 与 reference.npz

两个文件保留相同native关节序列，分别对应局部和tracker-world root：

```text
qpos[N,7+J]                       # root xyz,wxyz + joints
root_linear_velocity[N,3]
root_angular_velocity_world[N,3]  # local文件中表示局部参考坐标，world文件中表示tracker world
joint_velocity[N,J]
timestamps[N]
fps=50
joint_names[J], embodiment
includes_current_anchor=true
planner_future_frames=60
observation_tick=k
world_from_chunk[4,4]             # 仅world文件
```

local文件timestamps从0开始，world文件timestamps为`k/50+arange(N)/50`。完整exporter应将速度字段改为中性名称并增加`frame_id`，当前读取器必须按文件坐标约定解释，不能仅凭字段中的world字样判断。

### 6.4 execution.npz

```text
actual[M+1,Dstate]
policy_action[M,J]
pd_target[M,J]
lookahead_valid[M,11]
state_ticks[M+1]
state_physics_steps[M+1]
state_timestamps[M+1]
```

M为实际执行步数。actual[0]为输入state，actual[-1]为结束state；policy_action[i]和pd_target[i]作用于两个相邻state之间。PD target直接从实际后端ctrl复制，不从实际关节位置倒推。

中途reset/关闭保存已完成部分，reference仍保留完整规划。异常掉电可留下executing产物；导出器必须将其判为incomplete，不当作完整示范。

## 7 完整SDG发布所需metadata

当前推理记录是第一层原始证据，正式训练release还必须补齐以下信息：

- run_uid、episode_uid、chunk_uid、task文本/revision/source、任务成功判据与结果。
- 实际本体MJCF和mesh hashes、关节轴/零位/限位/拓扑、质量/惯量、相机挂点与标定。
- native↔unified映射版本、sign/offset、joint masks、归一化参数。
- default pose、action scale、calibration、kp/kd、effort limits、backend config。
- Joystick/Tracker checkpoint和代码版本、native转换/重采样/坐标adapter版本。
- scene与visual hashes、scene_from_tracker、装备/skin、任务/人工干预事件。
- 连续相机frame索引与physics timestamp、各文件hash、QC、split。

物理PD映射为 `q_PD = default + calibration + scale * policy_action`，其中每关节scale由现有后端配置产生。state、reference、policy action与PD target必须区分，不能归一化后混为同一action。

原始实例产物按本体/episode集中归档；release可以采用LeRobot v3的多episode Parquet和MP4 shards。逻辑chunk靠索引关联，文件无需一chunk一份。自定义local reference和transform扩展需要专用loader，不能宣称原生LeRobot自动解释。

## 8 训练样本

```text
input:
  task
  embodiment_features + masks
  ego_image_at_timestamp
  executed_proprio_history + timestamps + history_mask
  optional deploy-available navigation command
label:
  reference_local future[60] at25FPS
  或 reference_local future[120] at50Hz
metadata:
  current_anchor
  world_from_chunk
  physics_timestamp
feedback:
  actual / pd_target / task_outcome / execution_video_refs
```

50 Hz action标签为reference[1:]，当前锚点reference[0]不是未来动作标签；25 FPS标签从reference的2,4,…120点取得。采样选择和速度差分规则在训练配置中固定。

统一action以native reference关节通道映射到semantic slots，加joint mask；root单独编码，坐标锚点可逆。损失屏蔽不存在关节和无效时间。不能用执行期间video作为生成该chunk的输入；它是反馈/未来监督。

当前teacher依赖速度/目标primitive，训练数据必须保留这些条件。若EAWM不输入同样条件，应有task/导航信息能够解释动作，否则同一可见输入可能对应不同teacher指令。

按源motion、场景变体、采集session和配对本体分组split，相邻chunk不跨split。统计仅来自train；另设未见scene/embodiment评估。失败样本保留但默认不作成功模仿示范。

## 9 验证与验收

### CPU契约检查

- proprio context来自实际state history；下一轮即使上一段reference偏离也不能读取预测尾帧。
- 当前64帧输出释放全部60未来帧，带当前锚点转换为121个点，恰好120次PD调用。
- Tracker边界保留history；最后九个控制步逐渐增加terminal hold补帧，队列不增加补帧。
- 物理时间只在成功控制step后增加；推理/渲染等待不会使时钟漂移。
- 新段world_from_chunk来自新实测锚点；刚体往返、速度方向、quaternion和native关节顺序一致。
- snapshot等待正确render token，camera timestamp等于当前proprio timestamp。
- pause不丢当前段；reset断history并保存partial；结束后才允许下一次规划。

### GPU与视觉验收

保存并实际查看ego/exo视频、reference/actual叠加和控制日志，检查长2.4秒chunk的响应延迟、转向、停步、段尾hold、接缝、滑步、跌倒与任务完成。QC使用2.5m竖直范围及0.5m网格。数值测试通过不等于Tracker已适应实测context或末端补帧。

## 10 实现位置与后续

- [chunk.py](../src/omega_playground/policy/supermimic/chunk.py)：实际history与流式前瞻窗口。
- [live_planner.py](../src/omega_playground/policy/supermimic/live_planner.py)：实测context、完整future输出。
- [heading.py](../src/omega_playground/policy/supermimic/heading.py)：固定world_from_chunk。
- [rollout.py](../src/omega_playground/policy/supermimic/rollout.py)：Tracker连续history和PD记录。
- [live_view.py](../src/omega_playground/policy/supermimic/live_view.py)：physics-time render barrier。
- [在线入口](../tools/supermimic/serve_supermimic_live.py)：执行结束后规划、逐chunk原始证据。

后续：GPU短程回放验收→补齐正式SDG metadata与连续相机索引→实现unified adapter与训练shard exporter。此次改造不包含训练新模型、替换Tracker权重或远程部署。
