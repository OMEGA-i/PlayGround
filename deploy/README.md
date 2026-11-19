# Playground 部署

在仓库根目录执行 `bash deploy/start.sh`。启动器复用已就绪的服务，进程日志位于 `artifacts/runs/services/`。部署前需准备 CUDA、Nginx、FFmpeg NVENC 以及与模型匹配的资产。

| 服务 | 默认端口 | 用途 |
| --- | --- | --- |
| Nginx | 7777 | 对外作业台，`OMEGA_PUBLIC_PORT` 可覆盖 |
| 工作流选择器 | 18784 | 切换当前工作流 |
| SONIC | 18782 | G1，GPU MJWarp |
| SuperMimic | 18783 | 每个本体使用自己的 Joystick 和 Tracker |

浏览器访问宿主机映射后的地址。Pod 内用 `curl -fsS http://127.0.0.1:7777/api/state` 检查当前工作流就绪状态。

`OMEGA_SHARED_ROOT` 指向含 `joystick/` 和 `supermimic/` 的资产根目录，默认 `~/omega-assets`。可通过 `OMEGA_MJWARP_PYTHON`、`OMEGA_RENDER_PYTHON`、`OMEGA_JOYSTICK_PYTHON`、`OMEGA_SUPERMIMIC_PYTHON` 指定对应的解释器。初始工作流由 `OMEGA_WORKFLOW` 指定（默认 `sonic`），SuperMimic 初始本体由 `OMEGA_SUPERMIMIC_ROBOT` 指定（默认 `g1`）。初始场景由 `OMEGA_SCENE` 选择（`campus` 或 `obstacle`）。

场景初态放在 [configs/scenes](../configs/scenes/)，Joystick、Tracker 与 SIH2300 本体绑定放在 [configs/supermimic](../configs/supermimic/)，[资产清单](../configs/mjwarp-assets.json) 中的路径相对于清单自身。
