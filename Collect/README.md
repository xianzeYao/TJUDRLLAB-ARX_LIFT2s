# Collect

当前 `Collect/` 目录主要保留这些采集/重放入口：

- `collect_vr.py`
  - `collect_vr_episode(..., arm_mode="single" | "dual")`
- `collect_3dmouse.py`
  - `collect_3dmouse_episode(..., arm_mode="single" | "dual")`
- `collect_gravity.py`
  - `collect_gravity_episode(..., arm_mode="single" | "dual")`
- `replay.py`
  - 重放采集好的 episode

## 设计约定

- 不走命令行参数
- 通过 Python 函数参数控制保存路径、频率、相机、单/双臂模式
- 双臂数据维度固定是 `14`
- 单臂数据维度固定是 `7`
- `action_kind` 只使用 `joint` 或 `eef`
- `replay()` 会自动按 `7/14` 维判断单/双臂

## 最常用函数

### VR 采集

```python
from Collect.collect_vr import collect_vr_episode

collect_vr_episode(
    env,
    arm_mode="dual",
    out_dir="episodes_raw/vr_upper_only",
    camera_names=("camera_h", "camera_l", "camera_r"),
    action_kind="joint",
    frame_rate=15.0,
    include_base=False,
)
```

### 重力采集

```python
from Collect.collect_gravity import collect_gravity_episode

collect_gravity_episode(
    env,
    arm_mode="single",
    leader_side="left",
    out_dir="episodes_raw",
    action_kind="joint",
    frame_rate=15.0,
    mirror=True,
)
```

### 3D Mouse 采集

依赖：

- 需要额外安装 `pyspacemouse`

```python
from Collect.collect_3dmouse import collect_3dmouse_episode

collect_3dmouse_episode(
    env,
    arm_mode="single",
    leader_side="left",
    out_dir="episodes_raw",
    frame_rate=15.0,
    control_rate=60.0,
    task="pick up cup",
)
```

默认按键：

- `button0`: 夹爪收紧
- `button1`: 夹爪张开

默认按键：

- `button0`: 当前激活手臂夹爪收紧
- `button1`: 当前激活手臂夹爪张开
- `button0 + button1`: 切换当前激活手臂

### 重放

```python
from ARX_Realenv.ROS2.arx_ros2_env import ARXRobotEnv
from Collect.replay import replay_episode

env = ARXRobotEnv(camera_type="color", camera_view=(), dir=None, video=False, img_size=(224, 224))
replay_episode(
    env,
    episode_dir="episodes_raw/episode_000000",
    speed=1.0,
)
```

## 保存格式

每条数据保存在：

```text
episode_xxxxxx/
  episode.json
  low_dim.npz
  images/*.npz
  images_depth/*.npz
```

其中：

- `episode.json`
  - 记录 `mode=dual/single`、`side`、`action_kind`、`frame_rate`
- `low_dim.npz`
  - 保存 `qpos / qvel / effort / eef / action`
  - 双臂是 `14` 维，单臂是 `7` 维
- `robot_base / base_wheels / base_velocity / action_base`
  - 只有双臂且 `include_base=True` 时才保存
