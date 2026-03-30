# Realenv 公用方法说明

这里的 `Realenv` 对应当前代码里的 `ARXRobotEnv`，文件位置是 `ARX_Realenv/ROS2/arx_ros2_env.py`。

这份说明按实际公开接口整理，重点回答三件事：

1. 怎么初始化 `ARXRobotEnv`
2. `reset()` 到 `close()` 之间能做什么
3. 常用方法的参数、返回值和调用约定

## 1. 推荐程序骨架

推荐把资源申请和释放写完整，不要只写一半：

```python
import sys
import time
import numpy as np

sys.path.append("../ARX_Realenv/ROS2")
from arx_ros2_env import ARXRobotEnv


def main():
    env = None
    try:
        env = ARXRobotEnv(
            duration_per_step=1.0 / 20.0,
            min_steps=20,
            max_v_xyz=0.25,
            max_a_xyz=0.20,
            max_v_rpy=0.30,
            max_a_rpy=1.00,
            camera_type="all",
            camera_view=("camera_h",),
            dir=None,
            video=False,
            video_fps=20.0,
            video_name=None,
            img_size=(640, 480),
        )

        obs = env.reset()
        print(obs.keys())

        env.step_lift(18.0)
        env.step_smooth_eef(
            {
                "left": np.array(
                    [0.10, 0.00, 0.15, 0.0, 0.0, 0.0, -2.0],
                    dtype=np.float32,
                )
            }
        )

        env.step_base(0.2, 0.0, 0.0)
        time.sleep(1.0)
        env.step_base(0.0, 0.0, 0.0)

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
```

## 2. `ARXRobotEnv(...)` 初始化参数

`ARXRobotEnv` 初始化时会：

- 启动 ROS2
- 创建通信节点
- 启动相机/状态订阅
- enable 机器人

当前构造函数参数如下：

- `duration_per_step: float = 0.02`
  平滑插值控制里单步时长，`1/20` 就是 20Hz。
- `min_steps: int = 10`
  平滑轨迹最少插值步数。
- `max_v_xyz: float = 0.25`
  平移最大速度限制。
- `max_v_rpy: float = 0.3`
  姿态最大速度限制。
- `max_a_xyz: float = 0.20`
  平移最大加速度限制。
- `max_a_rpy: float = 1.00`
  姿态最大加速度限制。
- `camera_type: Literal["color", "depth", "all"] = "all"`
  要订阅的图像类型。
- `camera_view: Iterable[str] = ("camera_l", "camera_h", "camera_r")`
  要订阅的相机视角列表。
- `dir: Optional[str] = None`
  默认图片/视频保存目录；后续 `get_observation()`、`get_camera()` 传 `save_dir=None` 时会沿用这里。
- `video: bool = False`
  默认保存方式；`False` 保存单帧，`True` 保存视频。
- `video_fps: float = 20.0`
  保存视频时的帧率。
- `video_name: Optional[str] = None`
  保存视频时的视频名前缀。
- `img_size: Optional[Tuple[int, int]] = (224, 224)`
  统一缩放后的图像大小，传给相机读取接口。

## 3. `reset()` 到 `close()` 之间通常做什么

中间一般就三类动作：

- 取观测
- 发控制
- 切模式

典型流程：

```python
obs = env.reset()

# 1) 取状态 / 图像
obs = env.get_observation()
frames = env.get_camera()
status = env.get_robot_status()

# 2) 控制机械臂 / 升降 / 底盘
env.step_smooth_eef(
    {"left": np.array([0.10, 0.00, 0.15, 0.0, 0.0, 0.0, -2.0], dtype=np.float32)}
)
env.step_lift(18.0)
env.step_base(0.2, 0.0, 0.0)

# 3) 切模式
env.set_special_mode(1, side="left")
```

常见任务包括：

- 拍图并做检测
- 读取左右臂状态
- 控制单臂或双臂末端位姿
- 控制绝对关节目标或关节增量
- 控制底盘平移 / 旋转
- 控制升降高度
- 执行抓取、放置、巡检等任务

## 4. 观测与状态接口

## `reset() -> Dict[str, np.ndarray]`

作用：

- 等待约 `2.5s` 让状态稳定
- 双臂回初始位
- 升降回到 `0.0`
- 底盘速度清零
- 返回一份观测 `obs`

常见写法：

```python
obs = env.reset()
```

适合场景：

- 程序起始阶段做一次标准复位
- 新任务开始前统一回到已知状态

## `get_observation(save_dir=None, video=None, include_arm=True, include_camera=True, include_base=True)`

统一获取观测，最常用。

```python
obs = env.get_observation()
obs = env.get_observation(include_camera=False)
obs = env.get_observation(save_dir="debug_frames", video=False)
```

参数说明：

- `save_dir`
  当前调用的保存目录；`None` 时沿用实例化时的 `dir`。
- `video`
  当前调用的保存模式；`None` 时沿用实例化时的 `video`。
- `include_arm`
  是否包含左右臂状态。
- `include_camera`
  是否包含相机图像。
- `include_base`
  是否包含升降 / 底盘状态。

常见返回键：

- `left_end_pos` / `right_end_pos`
  末端位姿，格式为 `[x, y, z, roll, pitch, yaw]`
- `left_joint_pos` / `right_joint_pos`
- `left_joint_vel` / `right_joint_vel`
- `left_joint_cur` / `right_joint_cur`
- `base_height`
- `base_wheel1` / `base_wheel2` / `base_wheel3`
- `camera_h_color`
- `camera_h_aligned_depth_to_color`

适合场景：

- 想一次拿齐状态和图像
- 算法流程里直接读取统一观测
- 需要顺手保存调试图片 / 视频

## `get_camera(save_dir=None, video=None, target_size=None, return_status=False)`

只取相机，或者取相机加状态快照。

```python
frames = env.get_camera()
frames = env.get_camera(target_size=(640, 480))
frames, status = env.get_camera(return_status=True)
```

参数说明：

- `save_dir`
  本次保存目录，`None` 时沿用实例化参数。
- `video`
  本次是否保存视频，`None` 时沿用实例化参数。
- `target_size`
  本次图像缩放大小，`None` 时沿用实例化时的 `img_size`。
- `return_status`
  `False` 时只返回 `frames`；`True` 时返回 `(frames, status)`。

返回的 `frames` 是字典，键通常像：

- `camera_h_color`
- `camera_l_color`
- `camera_r_color`
- `camera_h_aligned_depth_to_color`

适合场景：

- 只关心图像，不需要完整 `obs`
- 想做视觉检测、点选、保存图片

## `get_robot_status()`

只取机器人状态，不取图像。

```python
status = env.get_robot_status()
left = status["left"]
right = status["right"]
base = status["base"]
```

常见可读字段：

- `left.end_pos`
- `left.joint_pos`
- `left.joint_vel`
- `left.joint_cur`
- `base.height`
- `base.chx / base.chy / base.chz`

适合场景：

- 判断当前机械臂或底盘状态
- 不想取图，只想轻量读取状态

## 5. 控制接口

### 5.1 机械臂动作的统一格式

机械臂控制接口里，`action` 统一都是字典：

```python
{
    "left": np.ndarray(shape=(7,), dtype=np.float32),
    "right": np.ndarray(shape=(7,), dtype=np.float32),
}
```

可以只传单臂，例如只传 `"left"`。

末端位姿类接口的 7 维含义：

- 前 6 维：`[x, y, z, roll, pitch, yaw]`
- 第 7 维：`gripper`

关节类接口的 7 维含义：

- 前 6 维：关节位置
- 第 7 维：`gripper`

如果维度不对、字典为空、或者键不是 `left/right`，接口会直接报错。

### 5.2 `step_smooth_eef(action, return_observation=False)`

最常用的末端控制接口，会根据当前状态做插值。

```python
env.step_smooth_eef(
    {
        "left": np.array([0.10, 0.00, 0.15, 0.0, 0.0, 0.0, -2.0], dtype=np.float32),
        "right": np.array([0.10, 0.00, 0.15, 0.0, 0.0, 0.0, -2.0], dtype=np.float32),
    }
)
```

说明：

- 会读取当前观测后做轨迹插值
- 受 `duration_per_step`、`min_steps`、速度 / 加速度限制影响
- `return_observation=True` 时，执行完会再取一次观测返回

适合场景：

- 常规末端位姿控制
- 需要更平滑的动作执行

### 5.3 `step_raw_eef(action, return_observation=False)`

直接发送绝对末端目标，不做上层插值规划。

```python
env.step_raw_eef(
    {
        "left": np.array([0.12, 0.00, 0.18, 0.0, 0.0, 0.0, -2.5], dtype=np.float32)
    }
)
```

适合场景：

- 你已经自己规划好了目标
- 想更直接地下发末端命令

### 5.4 `step_raw_joint(action, return_observation=False)`

直接发送绝对关节目标。

```python
env.step_raw_joint(
    {
        "right": np.array([0.0, 0.3, 0.8, 1.2, 0.0, 0.0, -2.0], dtype=np.float32)
    }
)
```

适合场景：

- 明确知道关节目标
- 做 joint 级调试

### 5.5 `step_smooth_joint(action, num_steps=5, step_sleep_s=0.01, return_observation=False)`

对绝对关节目标做线性插值后逐步发送。

```python
env.step_smooth_joint(
    {
        "left": np.array([0.0, 0.2, 0.6, 1.0, 0.0, 0.0, -2.0], dtype=np.float32)
    },
    num_steps=8,
    step_sleep_s=0.02,
)
```

参数说明：

- `num_steps`
  插值步数，必须大于 `0`。
- `step_sleep_s`
  两步之间的 sleep 时间，必须大于等于 `0`。

适合场景：

- 想用 joint 目标，但又不希望一步跳过去
- 机械臂调试时需要更可控的关节过渡

### 5.6 `step_delta_eef(action, return_observation=False)`

在当前末端位姿基础上做增量控制。

格式：

- `left/right -> [dx, dy, dz, droll, dpitch, dyaw, dgripper]`

```python
env.step_delta_eef(
    {
        "left": np.array([0.01, 0.00, 0.00, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    }
)
```

适合场景：

- 小步微调
- 视觉伺服
- 手工调位置

### 5.7 `step_delta_joint(action, return_observation=False)`

在当前关节位置基础上做关节增量控制。

格式：

- `left/right -> [dj0, dj1, dj2, dj3, dj4, dj5, dgripper]`

适合场景：

- 关节级微调
- 标定 / 调试

### 5.8 `step_lift(height, return_observation=False)`

控制升降高度，内部会从当前高度逐步走到目标高度。

```python
env.step_lift(18.0)
```

适合场景：

- 抓取前调工作高度
- 巡检 / 放置时切不同作业面

### 5.9 `step_base(vx, vy, vz, return_observation=False)`

发送一次底盘速度命令，立即返回。

```python
env.step_base(0.2, 0.0, 0.0)
```

当前项目里的常见理解方式：

- `vx`：前后
- `vy`：左右
- `vz`：旋转

注意：

- 这是一次发送，不会自动持续
- 想持续运动时，通常要配合 `time.sleep(...)`
- 结束时记得再发一次 `env.step_base(0.0, 0.0, 0.0)`

典型写法：

```python
env.step_base(0.2, 0.0, 0.0)
time.sleep(2.0)
env.step_base(0.0, 0.0, 0.0)
```

### 5.10 `set_special_mode(mode, side="both")`

给机械臂切特殊模式。

可用模式：

- `0`：soft
- `1`：home
- `2`：protect
- `3`：gravity

示例：

```python
env.set_special_mode(1, side="left")
env.set_special_mode(3, side="right")
env.set_special_mode(2, side="both")
```

说明：

- `side` 可选 `left`、`right`、`both`
- `mode=1` 不是单独发模式切换，而是直接走回初始位逻辑
- 返回值是 `(success, error_message)`

## 6. 关闭接口

## `close()`

作用：

- 底盘速度清零
- 双臂回初始位
- 升降归零
- 关闭 ROS2 通信资源

一般只在程序退出时调用一次，推荐放进 `finally`。

## 7. 使用建议

- 普通业务优先用 `env.get_observation()`、`env.get_camera()`、`env.get_robot_status()`，不要一开始就直接操作 `env.node`
- 常规末端控制优先用 `step_smooth_eef(...)`
- 需要 joint 级调试时再用 `step_raw_joint(...)` 或 `step_smooth_joint(...)`
- 需要微调时优先用 `step_delta_eef(...)` 或 `step_delta_joint(...)`
- 做底盘持续移动时，最后一定补一个零速度停止命令
- 机械臂动作接口的 `action` 一定传字典，不是单个数组
- 程序结束时一定调用 `close()`

## 8. 一句话总结

最推荐的使用方式就是：

```python
env = None
try:
    env = ARXRobotEnv(...)
    env.reset()
    # 中间做取图、读状态、控双臂、控底盘、控升降、切模式
finally:
    if env is not None:
        env.close()
```
