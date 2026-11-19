# 作业台视频

Main View 通过 WebSocket 传输 H.265，Ego View 通过 MJPEG 传输。前端分别显示收到的帧率与服务端实测渲染输出帧率。物理与 Tracker 在 GPU 上推进，渲染进程读取最新的姿态。主视角使用 FFmpeg NVENC 编码，需在部署机器上提供可用的 NVIDIA 编码器及驱动库。
