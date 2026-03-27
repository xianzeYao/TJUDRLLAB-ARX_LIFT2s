# Realenv 公用方法说明

这里的 `Realenv` 对应当前代码里的 `ARXRobotEnv`，文件位置是 `ARX_Realenv/ROS2/arx_ros2_env.py`。

本文主要说明三件事：

1. 一般程序怎么建立类实例
2. 推荐的 `try: ... reset ... finally: close()` 写法
3. `reset` 和 `close` 中间常见可以做什么，以及几个公用方法的用途

## 1. 一般程序骨架

推荐写法：

```python
import sys
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
            img_size=(640, 480),
        )

        obs = env.reset()
        print(obs.keys())
        # obs = env.reset()

        # 这里开始写你的业务逻辑

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
```


## 4. `reset()` 和 `close()` 中间可以干嘛

中间一般就是做三类事情：

- 取观测
- 发控制
- 切模式

典型流程：

```python
obs = env.reset()

# 1) 看当前状态/图像
obs = env.get_observation()
cams = env.get_camera()
status = env.get_robot_status()

# 2) 控制机械臂
action = {
    "left": np.array([0.10, 0.00, 0.15, 0.0, 0.0, 0.0, -2.0], dtype=np.float32),
}
env.step_smooth_eef(action)

# 3) 控制升降和底盘
env.step_lift(18.0)
env.step_base(0.2, 0.0, 0.0)

# 4) 切模式
env.set_special_mode(1, side="left")
```

常见任务包括：

- 拍图并做检测
- 读取左右臂状态
- 控制单臂或双臂末端位姿
- 直接发关节目标
- 做增量控制
- 控制底盘平移/旋转
- 控制升降高度
- 执行抓取、放置、巡检等任务

## 5. 常用公用方法

## `ARXRobotEnv(...)`

创建真实机器人环境对象，初始化时会启动 ROS2 通信并 enable 机器人。

常用参数：

- `duration_per_step`：插值控制里每一步的时间
- `min_steps`：最少插值步数
- `max_v_xyz / max_a_xyz`：末端平移速度/加速度限制
- `max_v_rpy / max_a_rpy`：末端姿态速度/加速度限制
- `camera_type`：`"color"`、`"depth"`、`"all"`
- `camera_view`：相机列表，例如 `("camera_h",)` 或 `("camera_l", "camera_h", "camera_r")`
- `dir`：取图时保存目录，默认 `None`
- `video`：是否保存视频
- `video_name`：视频文件名前缀
- `img_size`：图像缩放大小，例如 `(640, 480)`

## `reset()`

作用：

- 等待一小段时间让状态稳定
- 双臂回初始位
- 升降归零
- 底盘速度清零
- 返回一份观测 `obs`

返回值：

- `Dict[str, np.ndarray]`

常见写法：

```python
obs = env.reset()
```

## `close()`

作用：

- 停止底盘
- 双臂回初始位
- 升降归零
- 关闭 ROS2 通信资源

一般只在程序退出时调用一次。

## `get_observation(...)`

统一获取观测，最常用。

```python
obs = env.get_observation()
```

常用参数：

- `save_dir`：保存图像或视频的目录
- `video`：`True` 保存视频，`False` 保存单帧
- `include_arm`：是否包含双臂状态
- `include_camera`：是否包含图像
- `include_base`：是否包含底盘/升降状态

常见返回键：

- `left_end_pos` / `right_end_pos`：`[x, y, z, roll, pitch, yaw]`
- `left_joint_pos` / `right_joint_pos`
- `left_joint_vel` / `right_joint_vel`
- `left_joint_cur` / `right_joint_cur`
- `base_height`
- `base_wheel1` / `base_wheel2` / `base_wheel3`
- `camera_h_color`
- `camera_h_aligned_depth_to_color`

适合场景：

- 想一次把状态和图像都拿到
- 想保存当前图像
- 想在算法里直接读取统一观测

## `get_camera(...)`

只取相机，或者取相机加状态快照。

```python
frames = env.get_camera()
frames, status = env.get_camera(return_status=True)
```

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

适合场景：

- 判断当前机械臂或底盘状态
- 不想取图，只想轻量读取状态

## `step_smooth_eef(action)`

最常用的末端控制接口。

这里的 `action` 实际上传的是字典，不是单个数组。每个机械臂目标是 7 维：

- 前 6 维：`[x, y, z, roll, pitch, yaw]`
- 第 7 维：`gripper`

示例：

```python
action = {
    "left": np.array([0.10, 0.00, 0.15, 0.0, 0.0, 0.0, -2.0], dtype=np.float32),
    "right": np.array([0.10, 0.00, 0.15, 0.0, 0.0, 0.0, -2.0], dtype=np.float32),
}
env.step_smooth_eef(action)
```

特点：

- 会参考当前观测做插值规划
- 会受 `duration_per_step`、`min_steps`、速度加速度限制影响
- 适合正常的末端位姿控制

## `step_raw_eef(action)`

直接发送绝对末端目标，不走上层插值规划。

格式仍然是：

- `left/right -> [x, y, z, roll, pitch, yaw, gripper]`

示例：

```python
env.step_raw_eef({
    "left": np.array([0.12, 0.00, 0.18, 0.0, 0.0, 0.0, -2.5], dtype=np.float32)
})
```

适合场景：

- 你已经自己规划好了目标
- 你想更直接地下发末端命令

## `step_raw_joint(action)`

直接发送绝对关节目标。

每个目标也是 7 维：

- 前 6 维：关节位置
- 第 7 维：夹爪

示例：

```python
env.step_raw_joint({
    "right": np.array([0.0, 0.3, 0.8, 1.2, 0.0, 0.0, -2.0], dtype=np.float32)
})
```

适合场景：

- 你明确知道关节目标
- 你要做 joint 级调试

## `step_delta_eef(action)`

在当前末端位姿基础上做增量控制。

格式：

- `left/right -> [dx, dy, dz, droll, dpitch, dyaw, dgripper]`

示例：

```python
env.step_delta_eef({
    "left": np.array([0.01, 0.00, 0.00, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
})
```

适合场景：

- 小步微调
- 视觉伺服
- 手工调位置

## `step_delta_joint(action)`

在当前关节位置基础上做关节增量控制。

格式：

- `left/right -> [dj0, dj1, dj2, dj3, dj4, dj5, dgripper]`

## `step_lift(height)`

控制升降高度。

```python
env.step_lift(18.0)
```

说明：

- 内部会从当前高度逐步走到目标高度
- 常用于抓取前调工作高度

## `step_base(vx, vy, vz)`

发送一次底盘速度命令，立即返回。

```python
env.step_base(0.2, 0.0, 0.0)
```

当前常见理解方式：

- `vx`：前后
- `vy`：左右
- `vz`：旋转

注意：

- 这是一次发送，不会自动持续
- 想持续运动时，通常需要配合 `time.sleep(...)`
- 结束时记得再发一次 `env.step_base(0.0, 0.0, 0.0)`

典型写法：

```python
env.step_base(0.2, 0.0, 0.0)
time.sleep(2.0)
env.step_base(0.0, 0.0, 0.0)
```

## `set_special_mode(mode, side="both")`

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
- `mode=1` 本质上会走回初始位逻辑

## 6. 一个更完整的推荐模板

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
            img_size=(640, 480),
        )

        env.reset()
        env.step_lift(18.0)

        obs = env.get_observation()
        print(obs.keys())

        frames = env.get_camera()
        print(frames.keys())

        action = {
            "left": np.array([0.10, 0.00, 0.15, 0.0, 0.0, 0.0, -2.2], dtype=np.float32),
        }
        env.step_smooth_eef(action)

        env.step_base(0.2, 0.0, 0.0)
        time.sleep(1.0)
        env.step_base(0.0, 0.0, 0.0)

        env.set_special_mode(1, side="left")

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
```

## 7. 使用建议

- 普通业务优先用 `env.get_observation()`、`env.get_camera()`、`env.get_robot_status()`，不要一开始就直接操作 `env.node`
- 常规末端控制优先用 `env.step_smooth_eef(...)`
- 需要 joint 级调试时再用 `step_raw_joint(...)`
- 需要微调时优先用 `step_delta_eef(...)` 或 `step_delta_joint(...)`
- 任何程序都建议把 `close()` 放在 `finally` 里
- 做底盘持续移动时，最后一定补一个零速度停止命令

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
