# FR3 + Wuji EEF replay

回放 `/home/user/tomato_eef` 中的一段 episode（可用 `--dataset` 指定其他数据目录）。
离线完成 cam0 → 各自机械臂底座 → link8 IK → 有界平滑与时间缩放；
真机端复用 `openpi/deploy/fr3_wuji_runtime` 的控制器、网关、Wuji SDK 和反馈保护。
无需模型服务、checkpoint、视频或实时相机。

默认回放 `action`（目标 EEF + 手指指令）；用 `--field observation.state` 可回放实测轨迹。
一次回放一个 episode/帧区间，不会自动循环或跳到下一段。

代码仓库：<https://github.com/YAnG-0419/fr3_wuji_eef_replay>。
仓库包含源码、测试和限位快照；数据集、`outputs/` 回放包及本地 Python 环境不提交到 Git。

```bash
git clone https://github.com/YAnG-0419/fr3_wuji_eef_replay.git
cd fr3_wuji_eef_replay
```

## 本机直接试用

```bash
cd /home/user/lpy/fr3_wuji_eef_replay
export REPLAY_PYTHON=/home/user/lpy/Pi05/.venv/bin/python

# 生成整段 episode 0 的离线回放包，不连接设备。
# output 必须不存在；省略它会生成唯一目录。
bash run.sh plan --episode 0 --output outputs/my-episode0

# 只取前 90 帧（end-frame 不包含该帧），适合先做短段检查。
bash run.sh plan --episode 0 --start-frame 0 --end-frame 90 \
  --output outputs/my-first90

# 回放实测状态，而不是动作指令。
bash run.sh plan --episode 1 --field observation.state \
  --output outputs/my-measured-episode1
```

独立安装离线环境：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash run.sh plan --dataset /path/to/tomato_eef_cam0 --episode 0
```

推荐 Python 3.10+。离线只依赖 `numpy/scipy/pyarrow/PyYAML`。
`run.sh` 默认用当前目录 `.venv/bin/python`，也可以指定 `REPLAY_PYTHON`。
入口禁用 Python 字节码缓存；复用 Pi05 Python 时无需安装包或修改其代码、环境。

## 当前工作站：Home 后回放一条完整轨迹

当前默认数据目录为 `/home/user/tomato_eef`，部署目录为 `/home/user/lpy/Pi05`，
右手地址为 `192.168.2.111:50001`，左手仍为 `192.168.1.110:7447`。
这些是本工作站设置，其他机器应显式覆盖对应参数。
现场 Home 使用右手端口 `7447` 曾连接超时，显式指定 `50001` 后成功；
不要仅凭能扫描到设备或 ping 通就认定 SDK 连接可用。

```bash
# 现场值守：此命令会使双臂双手回到 Home，不修改 Pi05 配置。
bash /home/user/lpy/Pi05/deploy/fr3_wuji_home/home.sh \
  --wuji-right-address 192.168.2.111:50001

cd /home/user/lpy/fr3_wuji_eef_replay
export REPLAY_PYTHON=/home/user/lpy/Pi05/.venv/bin/python

# 离线生成回放包，输出目录必须不存在。
bash run.sh plan --episode 57 --field action --output outputs/episode57-full

# 不启动设备的配置预检。
bash run.sh run --plan outputs/episode57-full \
  --wuji-right-address 192.168.2.111:50001 --hand-control slider

# 现场值守：确认运动区域无障碍后执行，结束时双手释放。
bash run.sh run --plan outputs/episode57-full \
  --wuji-right-address 192.168.2.111:50001 --hand-control slider \
  --execute --supervised-trial
```

episode 57 的 559 帧 `action` 已通过离线规划及部署配置预检：原时长 18.6s，
规划回放约 117.67s，平滑后 EEF 采样点最大位置偏差约 3.15mm。
与本次成功 Home 的末帧反馈比较，双臂首帧最大关节差约 0.099rad；
episode 75 则约为 0.433rad，超过默认 0.3rad 接近范围，因此选用 episode 57。
这些历史结果不替代执行时的新鲜反馈检查，也不代表已完成真机 replay 验证。

## 输出内容与运动语义

每个回放包包含：

- `report.json`：源文件/标定哈希、字段、帧区间、IK/FK 误差、平滑偏差和实际慢放时长。
- `trajectory.npz`：原始 EEF、各底座下的 link8 目标、IK 后关节角、时间轴、关节限位及 FK 误差。

离线关节顺序为 **左臂7、左手20、右臂7、右手20**，与部署设备接口一致。
数据集 EEF 的顺序则是 **左臂 xyz+xyzw、右臂 xyz+xyzw、左手20、右手20**。
转换公式为：`T_base_link8 = T_base_cam0 @ T_cam0_link8`。
这里控制的是 link8，不是假定的手掌中心/TCP。

FR3 是 7 自由度，单个末端位姿不能唯一决定关节姿态。默认 `--seed-mode recorded` 使用
数据中的 `action_joint` 或 `observation.joint_state` 位置作为 IK 种子，保留采集时的冗余构型。
每一帧都会对 **EEF 目标做 FK 残差校验**；种子不符合目标时才数值求解。
也可用 `--seed-mode previous` 从上一帧解继续求 IK；可能产生与采集时不同的肘部姿态。
种子关节字段必须存在，不会凭空猜一个初始构型。

`--calibration` 默认取数据集内 `conversion/calibration.yaml`。它是**源数据的变换**：
使用它将记录的 cam0 位姿恢复到记录时的两侧底座坐标，现场可以不启动相机。
要在另一套安装上复现相同空间路径，需要确认两侧底座/工具/工件之间的布置相符。
替换标定会改变目标底座位姿；不要仅因现场相机换了位置就替换源数据标定。

默认 `--interpolation bounded` 使用五次 B-spline 的非负基函数平滑关节目标，
轨迹处于附近关节目标的凸包内；起止速度/加速度均为零，内部具有连续高阶导数。
因此原始 EEF/IK 采样点会有小量偏差，报告中 `playback.max_eef_position_knot_deviation_m`
和 `max_eef_rotation_knot_deviation_rad` 给出**原采样时刻**的最大偏差。
它们不是整个连续 EEF 路径的误差上界。

需要插值经过每个 IK 点时，可用 `--interpolation interpolate`；若五次插值在点间越过关节限位，
工具会拒绝该段，不会裁剪指令或静默换模式。原始手指指令也必须在关节限位内。
EEF 只在采样点做逆解，执行的是关节曲线；点间不是笛卡尔直线，也未做双臂/环境碰撞检测。

默认 `--speed 0.4` 表示请求最多 0.4 倍原速。全轨迹会根据解析计算的速度、加速度、jerk 极值
进一步统一慢放，而不是单纯把 30Hz 数据改成另一频率。当前规划上限：

| 阶段 | 臂速度 rad/s | 臂加速度 rad/s² | 臂 jerk rad/s³ | 手速度 rad/s |
| --- | ---: | ---: | ---: | ---: |
| 从实时姿态接近首帧 | 0.18 | 0.432 | 3.456 | 0.35 |
| 回放 | 0.42 | 1.44 | 17.28 | π/4 |

手部接近加速度/jerk 为 1/5；回放为 3/30。所有轴共用同一时间缩放，保持双臂双手同步目标。
这些是软件规划上限；真机仍由现有设备进程检查实时状态。
默认全程跟随选中的目标，包括 `observation.engaged=0` 的区间，不根据 engaged 值停开某个设备。

## 迁移到真机电脑

复制本工具目录和生成的回放包即可，不必复制视频或整个数据集。
若要在现场重新选 episode，再复制数据集的 `meta/`、`data/`、`conversion/`。
本机另有 `../fr3_wuji_eef_replay.tar.gz`，包含代码、测试、限位快照和已验证的示例包。
该压缩包与示例回放包不在 Git 仓库中；从 Git 克隆后，先运行 `plan` 生成回放包，或从数据电脑复制。

真机电脑需要已经可用的这份 **定制 openpi 部署栈**，包括：

- `deploy/fr3_wuji_runtime/{hardware,bridge,ipc,ros_boundary}.py` 及相关 `fr3_wuji_slow/`、`experiments/` 模块。
- 已在该电脑构建并能通过原部署校验的 `.deployment/oneshot-overlay`。
- 原部署的 `gello-retarget` 工作站配置、Docker 镜像 `franka-upper-body-teleop:latest`。
- 容器能使用的 Wuji SDK。`--sdk-site` 指向含 `wuji_sdk/`、`wuji_sdk.libs/` 的目录。

普通上游 openpi 不包含上述硬件模块。沿用原部署环境运行本工具，**不要为回放启动模型服务**。
硬件进程由这个入口启动；现场先结束原来的推理执行器、遥操作/手部 SDK owner 和机械臂控制进程，
避免两个进程同时控制设备。CPU 分配、ROS/DDS 和 Docker 权限沿用 `hardware.py` 的配置
（当前版本配置使用到 CPU 23），换机器时需要匹配原部署环境。

下面路径按真机的实际安装位置修改：

```bash
cd /path/to/fr3_wuji_eef_replay
export ROBOT_OPENPI=/home/user/lpy/Pi05
export ROBOT_REFERENCE=/home/user/lpy/gello-retarget
export WUJI_SDK_SITE=/home/user/miniconda3/envs/gello-upper-body-teleop/lib/python3.10/site-packages
export REPLAY_PYTHON="$ROBOT_OPENPI/.venv/bin/python"

# 默认仅校验控制器构建/限位并输出启动命令，不启动任何设备。
bash run.sh run --plan outputs/episode0-full \
  --openpi "$ROBOT_OPENPI" --reference "$ROBOT_REFERENCE" --sdk-site "$WUJI_SDK_SITE"
```

`run` 的配置检查也需要现场 overlay/工作站文件；本机只有训练/数据环境时，运行 `plan` 即可。
自定义 overlay 可传 `--overlay /path/to/oneshot-overlay`。

可选只读状态检查：在已有机械臂状态发布器的情况下，在上述命令末尾添加 `--read-only`。
该模式仅启动只读 bridge，记录 `feedback.json`，不使能双手。

在现场值守条件下执行短段回放：

```bash
# 先生成并复制一个短段包到 outputs/my-first90，再在现场执行。
bash run.sh run --plan outputs/my-first90 \
  --openpi "$ROBOT_OPENPI" --reference "$ROBOT_REFERENCE" --sdk-site "$WUJI_SDK_SITE" \
  --left-arm-ip 172.16.0.2 --right-arm-ip 172.16.1.2 \
  --wuji-left-address 192.168.1.110:7447 --wuji-right-address 192.168.2.111:50001 \
  --execute --supervised-trial
```

只有 `--execute` 会输出运动。它仍要求原部署的 `--supervised-trial`（现场值守模式），
或 `--qualification /path/to/qualification.json`（原有调试验证记录）；不会生成虚假的通过记录。
默认手部 `strict` 模式；需沿用原来的 UI slider 行为时，传 `--hand-control slider`，
此时 SDK 内部还会限制手指目标变化，实际手部轨迹可能滞后于回放曲线。

真机先从**新鲜实测姿态**生成首帧接近曲线，随后按 100Hz 下发回放。
默认首帧双臂最大关节偏差不得超过 `0.3rad`。若超限，先将机器移近采集起点；
确实需要更大的接近范围时，可以显式传 `--max-start-delta 1.5`，仍然通过速度/限位检查。
接近期间没有碰撞规划。采集/回放机器的关节极限不一致时，取两者更严格的范围。

运行中 Ctrl-C/SIGTERM 会请求停止；通信丢失、20ms 指令过期、反馈超时或跟踪异常由
现有设备/控制器保护处理。调度错过 100Hz 截止时间后不会突发补发旧帧。
末帧稳定至少 0.5s 后按原部署 `finish-policy=disable` 释放双手并清理本次设备进程；
两侧手部会失去保持力，现场应事先安排好物体支撑。
停止请求或 RPC 成功不等于物理停止验证，结果会保留原设备返回的 `physical_stop_confirmed` 值。

现场输出目录包含 `commands.json`、`runtime.json`、控制器/bridge 日志及 `result.json`。
成功结束时还会写 `execution.npz`，包含时间、命令和反馈；异常退出时保留错误与设备日志。

## 已做的本机验证

使用本机 `openpi/.venv/bin/python`，未连接真机：

| 回放对象 | 帧数 | 原时长 | 规划回放时长 | 平滑后 EEF 采样点最大位置偏差 |
| --- | ---: | ---: | ---: | ---: |
| episode 0，action | 1278 | 42.57s | 255.86s | 2.41mm |
| episode 190，action | 955 | 31.80s | 197.54s | 2.58mm |
| episode 1，observation.state | 1069 | 35.60s | 511.34s | 1.33mm |

原始 IK 节点的 FK 最大位置误差均小于 `4.1e-8m`，这与平滑后路径偏差是两项不同检查。
实际回放时间会另加实时接近与末端稳定时间。上表慢放较多来自原始指令/测量中的急变；
`--speed 1` 仍会服从导数上限，不能保证原速。

```bash
REPLAY_PYTHON=/home/descfly/lpy/openpi/.venv/bin/python
"$REPLAY_PYTHON" -m pytest -q tests
```

8 项测试覆盖：左右坐标变换和手指映射、EEF 目标驱动 IK、不可达目标、四元数顺序/正负、
构型跳变、限位/导数边界、回放包篡改、100Hz 超时不补帧、操作员停止。
真机环境的控制器构建、网络、Wuji 反馈与实际运动需要在另一台电脑验证。
