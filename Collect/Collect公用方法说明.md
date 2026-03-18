# Collect 公用方法说明

这里的 `Collect` 是采集层，主要对应下面几个文件：

- `Collect/collect_gravity.py`
- `Collect/collect_vr.py`
- `Collect/collect_3dmouse.py`
- `Collect/collect_utils.py`
- `Collect/replay.py`
- `Collect/convert_lerobot.py`

和前面的 `Realenv` 不同，`Collect` 这一层主要做的是：

- 从真实机器人读取状态、动作、图像
- 组织成 episode
- 保存成统一数据格式
- 回放采集结果
- 转成 LeRobot 数据集格式

## 1. 一般采集程序骨架

最常见的写法是：

```python
from collect_gravity import collect_gravity_episode
from collect_utils import ARXRobotEnv


def main():
    env = None
    try:
        env = ARXRobotEnv(
            camera_type="color",
            camera_view=("camera_h", "camera_l", "camera_r"),
            dir=None,
            video=False,
            img_size=(640, 480),
        )

        env.reset()
        env.step_lift(14.5)

        collect_gravity_episode(
            env,
            arm_mode="single",
            out_dir="episodes_raw/gravity_single",
            frame_rate=20.0,
            action_kind="joint",
            camera_names=("camera_h", "camera_l", "camera_r"),
            with_depth=False,
            task="",
        )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
```

## 2. 一般程序的类建立

如果采集任务比较长，建议写成业务类。

```python
from collect_vr import collect_vr_episode
from collect_utils import ARXRobotEnv


class MyCollector:
    def __init__(self):
        self.env = ARXRobotEnv(
            camera_type="color",
            camera_view=("camera_h", "camera_l", "camera_r"),
            dir=None,
            video=False,
            img_size=(640, 480),
        )

    def run(self):
        self.env.reset()
        collect_vr_episode(
            self.env,
            arm_mode="dual",
            out_dir="episodes_raw/vr_dual",
            frame_rate=20.0,
            action_kind="joint",
            camera_names=("camera_h", "camera_l", "camera_r"),
        )

    def close(self):
        if self.env is not None:
            self.env.close()


def main():
    app = None
    try:
        app = MyCollector()
        app.run()
    finally:
        if app is not None:
            app.close()
```

## 3. 为什么还是建议 `try: ... finally: close()`

推荐结构：

```python
env = None
try:
    env = ARXRobotEnv(...)
    env.reset()

    # 中间开始采集
    ...
finally:
    if env is not None:
        env.close()
```

原因：

- 采集脚本通常会长时间运行
- 中间可能因为相机、VR、3D mouse、ROS 话题未就绪而报错
- `finally` 能保证机器人最后正常 `close()`

## 4. `reset()` 和 `close()` 中间可以干嘛

采集流程中间一般做这些事：

- 等待话题 ready
- 把机械臂切到特定模式
- 开始录制 episode
- 结束后选择保存或丢弃
- 回初始位
- 继续下一段 episode

典型流程：

```python
env.reset()
env.step_lift(14.5)

# 1) 开始一段采集
episode_dir = collect_gravity_episode(...)

# 2) 查看最近一次 episode
print(episode_dir)

# 3) 回放
replay_episode(env, episode_dir=episode_dir)

# 4) 转 LeRobot
convert_collect_to_lerobot(
    episodes_root="episodes_raw/gravity_single",
    output_root="lerobot_v3/gravity_single",
)
```

## 5. Collect 这一层的三个主入口

## `collect_gravity_episode(...)`

位置：

- `Collect/collect_gravity.py`

用途：

- 重力模式采集
- 支持单臂镜像采集
- 支持双臂重力采集

常见写法：

```python
collect_gravity_episode(
    env,
    arm_mode="single",
    out_dir="episodes_raw/gravity_single",
    frame_rate=20.0,
    action_kind="joint",
    camera_names=("camera_h", "camera_l", "camera_r"),
    with_depth=False,
    leader_side="left",
    mirror=True,
)
```

关键参数：

- `arm_mode`：`"single"` 或 `"dual"`
- `out_dir`：episode 保存目录
- `frame_rate`：录制帧率
- `action_kind`：`"joint"` 或 `"eef"`
- `camera_names`：采集哪些相机
- `with_depth`：是否同时录深度
- `leader_side`：单臂时由哪侧带动
- `mirror`：单臂时是否镜像到另一侧
- `control_rate`：内部控制循环频率

行为特点：

- 单臂模式用 `SingleArmMirrorCollector`
- 双臂模式用 `DualArmGravityCollector`
- 开始录制前会等待状态和相机 ready
- 录完后会询问 `save / discard / quit`

## `collect_vr_episode(...)`

位置：

- `Collect/collect_vr.py`

用途：

- 通过 VR 话题采集动作
- 支持单臂和双臂

常见写法：

```python
collect_vr_episode(
    env,
    arm_mode="dual",
    out_dir="episodes_raw/vr_dual",
    frame_rate=20.0,
    action_kind="joint",
    camera_names=("camera_h", "camera_l", "camera_r"),
    with_depth=False,
    leader_side="left",
    include_base=False,
)
```

关键参数：

- `arm_mode`：`"single"` 或 `"dual"`
- `action_kind`：这里只支持 `"joint"` 或 `"eef"`
- `leader_side`：单臂模式时保留哪一侧
- `include_base`：双臂模式下是否同时记录底盘动作

行为特点：

- 使用 `DualVRCollector`
- 底层会监听 `/ARX_VR_L` 和 `/ARX_VR_R`
- 单臂模式会从双臂 VR 帧里裁出一侧

## `collect_3dmouse_episode(...)`

位置：

- `Collect/collect_3dmouse.py`

用途：

- 用 3D mouse 控机械臂并采集
- 支持单臂和双臂

常见写法：

```python
collect_3dmouse_episode(
    env,
    arm_mode="single",
    out_dir="episodes_raw/3dmouse_single",
    frame_rate=20.0,
    action_kind="eef",
    camera_names=("camera_h",),
    leader_side="left",
    control_rate=60.0,
    translation_scale=0.10,
    rotation_scale=0.60,
    gripper_step=2.5,
    home_on_start=True,
)
```

关键参数：

- `action_kind`：这里只支持 `"eef"`
- `leader_side`：单臂侧，双臂时也是默认活动侧
- `translation_scale / rotation_scale`：平移和旋转速度缩放
- `gripper_step`：夹爪单步变化
- `translation_deadzone / rotation_deadzone`：死区
- `response_exponent`：响应曲线指数
- `home_on_start`：每次录制前是否先回 home

行为特点：

- 单臂用 `SingleArmSpaceMouseCollector`
- 双臂用 `DualArmSpaceMouseCollector`
- 依赖可选包 `pyspacemouse`

## 6. 自定义采集时最常用的公用方法

这些方法都在 `Collect/collect_utils.py`。

## `create_episode_buffer(...)`

作用：

- 创建一个内存中的 episode 容器

常见写法：

```python
episode = create_episode_buffer(
    episode_idx=0,
    mode="single",
    frame_rate=20.0,
    action_kind="joint",
    include_camera=True,
    include_base=False,
    camera_names=("camera_h",),
    config={"task": "pick cup"},
    side="left",
)
```

常用参数：

- `mode`：`"single"` 或 `"dual"`
- `action_kind`：`"joint"` 或 `"eef"`
- `include_camera`
- `include_base`
- `camera_names`
- `config`：任务说明和采集配置
- `side`：单臂时写 `left` 或 `right`

## `record_episode_interactive(...)`

作用：

- 按固定帧率循环调用 `capture_fn`
- 持续把帧塞进 `EpisodeBuffer`
- 在终端里通过回车停止，通过 `q` 停止并请求退出

常见写法：

```python
quit_requested = record_episode_interactive(
    episode=episode,
    capture_fn=collector.capture_frame,
    frame_rate=20.0,
    max_frames=0,
    prompt_start=False,
)
```

## `save_episode(...)`

作用：

- 把 `EpisodeBuffer` 落盘到一个 `episode_xxxxxx` 目录

常见写法：

```python
episode_dir = save_episode(episode, Path("episodes_raw/gravity_single"))
```

保存后目录里通常会有：

- `episode.json`
- `low_dim.npz`
- `images/*.npz`
- `images_depth/*.npz`

## `load_episode(...)`

作用：

- 从已保存的 episode 目录重新读回 `EpisodeBuffer`

```python
episode = load_episode("episodes_raw/gravity_single/episode_000000")
```

## `find_next_episode_index(root)`

作用：

- 找下一个可用 episode 编号

```python
next_idx = find_next_episode_index("episodes_raw/gravity_single")
```

## `latest_episode_dir(root)`

作用：

- 找某个目录下最新一个 episode

```python
latest = latest_episode_dir("episodes_raw/gravity_single")
```

## 7. Episode 里实际存了什么

每一帧核心字段是 `EpisodeFrame`，主要包括：

- `qpos`
- `qvel`
- `effort`
- `eef`
- `action`
- `images`
- `images_depth`
- `robot_base`
- `base_wheels`
- `base_velocity`
- `action_base`
- `topic_stamps`

其中常见理解：

- `qpos`：关节位置
- `qvel`：关节速度
- `effort`：关节电流/力矩相关量
- `eef`：末端 `[x, y, z, roll, pitch, yaw, gripper]`
- `action`：这一帧记录下来的控制目标

维度约定：

- 单臂是 `7`
- 双臂是 `14`

## 8. 自定义采集时，中间通常还能做什么

除了直接调 `collect_*_episode(...)`，你也可以自己控制流程：

- 先 `wait_until_ready()`
- 再 `prepare()`
- 然后自己调 `capture_frame(frame_idx)`
- 自己决定什么时候 `save_episode(...)`
- 录完后做筛选、重放、转换

更底层一点时，常见中间动作包括：

- `env.set_special_mode(1)` 回初始位
- `env.set_special_mode(3)` 切重力模式
- `env.step_lift(...)` 调工作高度
- `env.get_camera(...)` 看当前图像
- `env.get_robot_status()` 检查状态是否稳定

## 9. 回放方法

回放入口在 `Collect/replay.py`。

## `replay_episode(...)`

作用：

- 按时间戳把历史 episode 动作重新发给机器人

常见写法：

```python
from replay import replay_episode

replay_episode(
    env,
    episode_dir="episodes_raw/gravity_single/episode_000000",
    speed=1.0,
    start_index=0,
    end_index=-1,
    single_side=None,
)
```

参数说明：

- `speed`：回放速度倍率
- `start_index / end_index`：只回放一部分帧
- `single_side`：单臂回放时指定是左臂还是右臂

行为特点：

- 自动判断单臂还是双臂
- `joint` 数据走 `env.step_raw_joint(...)`
- `eef` 数据走 `env.step_raw_eef(...)`
- 如果 episode 里包含底盘数据，也会一起发底盘和升降

## 10. 转 LeRobot 方法

转换入口在 `Collect/convert_lerobot.py`。

## `convert_collect_to_lerobot(...)`

作用：

- 把 `episodes_raw/...` 转成 LeRobot 数据格式

常见写法：

```python
from convert_lerobot import convert_collect_to_lerobot

convert_collect_to_lerobot(
    episodes_root="episodes_raw/gravity_single",
    output_root="lerobot_v3/gravity_single",
    repo_id="tjudrllab/gravity_single",
    target_version="v3.0",
    fps=None,
    robot_type="arx",
    task_override=None,
    include_depth_images=False,
    max_episodes=0,
)
```

关键参数：

- `episodes_root`：原始 episode 根目录
- `output_root`：转换输出目录
- `repo_id`：LeRobot 数据集名
- `target_version`：支持 `v2.1` 或 `v3.0`
- `fps`：不写就沿用 episode 自带帧率
- `task_override`：统一覆盖任务文本
- `include_depth_images`：是否导出深度图
- `max_episodes`：最多转换多少个 episode

## 11. 一个更底层的自定义模板

如果你不想直接用 `collect_gravity_episode(...)` 这种高层封装，也可以自己写：

```python
from pathlib import Path
from collect_utils import (
    ARXRobotEnv,
    SingleArmMirrorCollector,
    create_episode_buffer,
    find_next_episode_index,
    record_episode_interactive,
    save_episode,
)


def main():
    env = None
    collector = None
    try:
        env = ARXRobotEnv(
            camera_type="color",
            camera_view=("camera_h",),
            dir=None,
            video=False,
            img_size=(640, 480),
        )
        env.reset()
        env.step_lift(14.5)

        collector = SingleArmMirrorCollector(
            env=env,
            leader_side="left",
            camera_names=("camera_h",),
            include_camera=True,
            use_depth=False,
            action_kind="joint",
            mirror=True,
            img_size=(640, 480),
        )

        collector.wait_until_ready()
        collector.prepare()

        episode = create_episode_buffer(
            episode_idx=find_next_episode_index(Path("episodes_raw/custom")),
            mode="single",
            frame_rate=20.0,
            action_kind="joint",
            include_camera=True,
            include_base=False,
            camera_names=("camera_h",),
            config={"task": "custom collect"},
            side="right",
        )

        record_episode_interactive(
            episode=episode,
            capture_fn=collector.capture_frame,
            frame_rate=20.0,
            max_frames=0,
            prompt_start=False,
        )

        if episode.frame_count > 0:
            save_episode(episode, Path("episodes_raw/custom"))

    finally:
        if collector is not None:
            try:
                collector.close()
            except Exception:
                pass
        if env is not None:
            env.close()
```

## 12. 使用建议

- 高层采集优先直接用 `collect_gravity_episode`、`collect_vr_episode`、`collect_3dmouse_episode`
- 只有在你要改采样逻辑时，再直接下钻到 `collect_utils.py`
- 录制前先确认 `env.camera_view` 和 `camera_names` 一致
- 想录深度时，`env.camera_type` 必须支持 `all` 或 `depth`
- `3dmouse` 采集前要先确认 `pyspacemouse` 已安装
- 自定义采集时，务必区分单臂 `7` 维和双臂 `14` 维
- 回放和转换前，先确认 episode 目录确实完整

## 13. 一句话总结

`Collect` 层最推荐的使用方式就是：

```python
env = None
try:
    env = ARXRobotEnv(...)
    env.reset()
    env.step_lift(...)
    collect_gravity_episode(...)   # 或 collect_vr_episode(...) / collect_3dmouse_episode(...)
finally:
    if env is not None:
        env.close()
```
