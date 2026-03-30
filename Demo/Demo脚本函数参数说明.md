# Demo 脚本函数参数说明

这份文档只整理 `Demo` 目录下复用度较高、会被其他脚本直接 import 的几个模块，方便后续写新 demo 时快速查函数入口。

## 1. `demo_utils.py`

用途：通用文本解析、底盘移动封装、抓取流程封装。

### `extract_numbered_sentences(raw)`

从大模型输出里提取编号步骤，并顺手抽取其中的 `xx cup` 描述。

- `raw: Optional[str]`
  大模型原始返回文本，可以包含代码块或 `<answer>...</answer>` 包裹。
- 返回值：`Tuple[List[str], List[str]]`
  第一个列表是步骤文本，第二个列表是提取出的杯子短语。

### `do_replan(color_img, planning_prompt, max_retries=5)`

针对当前彩色图反复请求规划结果，直到能解析出有效步骤。

- `color_img: np.ndarray`
  当前 RGB/BGR 图像。
- `planning_prompt: str`
  给多模态模型的完整规划提示词。
- `max_retries: int = 5`
  最大重试次数。
- 返回值：`Tuple[List[str], List[str], Optional[str]]`
  分别是步骤列表、杯子列表、最后一次原始回答文本。

### `step_base_duration(arx, vx, vy, vz, duration)`

对 `env.step_base()` 的时间封装。

- `arx`
  `ARXRobotEnv` 实例。
- `vx / vy / vz: float`
  底盘速度命令。
- `duration: float`
  持续时间，单位秒；小于等于 0 时直接返回。

### `estimate_lift_from_goal_z(goal_z, current_lift, target_goal_z=0.0, ...)`

根据目标点高度和当前升降高度，估算新的升降目标。

- `goal_z: float`
  当前视觉反算出的目标高度。
- `current_lift: float`
  当前升降高度。
- `target_goal_z: float = 0.0`
  希望目标点最终对齐到的参考高度。
- `meters_per_lift_unit`
  升降单位和米之间的换算系数。
- `min_lift / max_lift`
  结果裁剪范围。
- 返回值：`float`
  建议的升降目标值。

### `execute_pick_place_*_sequence(arx, pick_ref, place_ref, arm, do_pick=True, do_place=True)`

当前文档覆盖：

- `execute_pick_place_cup_sequence`
- `execute_pick_place_straw_sequence`
- `execute_pick_place_deepbox_sequence`
- `execute_pick_place_normal_object_sequence`

共用参数：

- `arx`
  `ARXRobotEnv` 实例。
- `pick_ref: Optional[np.ndarray]`
  抓取点在参考坐标系下的 3D 点。
- `place_ref: Optional[np.ndarray]`
  放置点在参考坐标系下的 3D 点。
- `arm: str`
  `"left"` 或 `"right"`。
- `do_pick / do_place: bool`
  是否执行抓取段 / 放置段。

说明：

- 这几个函数自己不做感知，只负责把动作序列逐个喂给 `arx.step_smooth_eef()`。
- 当 `do_pick=True` 且 `pick_ref is None` 时会报错；`do_place` 同理。

### `execute_move_away(arx, blocker_ref, arm)`

执行一个简单的“拨开障碍物”动作序列。

- `blocker_ref: Optional[np.ndarray]`
  障碍物参考点；为空会报错。
- `arm: str`
  `"left"` 或 `"right"`。

## 2. `point2pos_utils.py`

用途：把像素点和深度图转换成参考系或 base 系下的 3D 点。

### `get_aligned_frames(arx, depth_median_n=1)`

读取 `camera_h` 的彩色图和对齐深度图。

- `arx`
  `ARXRobotEnv` 实例。
- `depth_median_n: int = 1`
  大于 1 时会多次取深度并做逐像素中值融合。
- 返回值：`Tuple[np.ndarray | None, np.ndarray | None]`
  分别是彩色图和深度图。

### `load_intrinsics(path=None)`

读取 3x3 相机内参矩阵。

- `path: Path | str | None`
  不传时走默认 `camera_h` 内参文件。

### `load_cam2ref(path=None, side=None)`

读取相机到参考系的外参矩阵。

- `path`
  直接指定某个外参文件。
- `side`
  不传 `path` 时可传 `"left"` 或 `"right"` 读取默认外参。
- 返回值：
  传 `path` 或 `side` 时返回单个 `4x4` 矩阵；都不传时返回 `(T_left, T_right)`。

### `pixel_to_ref_point(pixel, depth_image, robot_part="left", ...)`

把像素点投影到左右臂参考系。

- `pixel: Tuple[int, int]`
  图像坐标 `(u, v)`。
- `depth_image: np.ndarray`
  对齐深度图。
- `robot_part`
  `"left"` 或 `"right"`。
- `K / T_left / T_right`
  可选的内参和外参缓存，传入可减少重复读文件。
- 返回值：`np.ndarray`
  shape 为 `(3,)` 的 3D 点。

### `pixel_to_base_point(pixel, depth_image, robot_part="center", ...)`

把像素点投影到工作坐标系。

- `robot_part`
  `"center"`、`"left"`、`"right"`。
- `offset`
  当 `robot_part="center"` 时使用的 ref 到工作中心补偿。

### `pixel_to_ref_point_safe(...)` / `pixel_to_base_point_safe(...)`

安全版本。深度无效或像素越界时返回 `None`，适合 demo 流程里直接做判空重试。

## 3. `motion_pick_place_cup.py`

用途：生成杯子抓取 / 放置动作序列，不直接执行。

### 动作格式

所有公开函数都返回：

```python
{"left": np.ndarray(shape=(7,), dtype=np.float32)}
```

或

```python
{"right": np.ndarray(shape=(7,), dtype=np.float32)}
```

7 维顺序固定为：

- `[x, y, z, roll, pitch, yaw, gripper]`

### 单步动作函数

- `make_pick_move_action(pt_ref, arm)`
- `make_pick_robust_action(pt_ref, arm)`
- `make_close_action(pt_ref, arm)`
- `make_pick_stop_action(pt_ref, arm)`
- `make_pick_back_action(pt_ref, arm)`
- `make_place_move_action(pt_ref, arm)`
- `make_place_robust_action(pt_ref, arm)`
- `make_down_action(pt_ref, arm)`
- `make_open_action(pt_ref, arm)`
- `make_place_stop_action(pt_ref, arm)`
- `make_release_action(pt_ref, arm)`

共用参数：

- `pt_ref: Optional[np.ndarray]`
  参考系下的 3D 目标点，预期 shape 为 `(3,)`。
- `arm: str`
  `"left"` 或 `"right"`。

说明：

- 这些函数只负责“算出动作”，不发给机器人。
- `pt_ref is None` 时默认按原点构造动作，通常只建议调试时这么用。

### `build_pick_cup_sequence(pt_ref, arm)`

返回杯子抓取动作列表。

### `build_place_cup_sequence(pt_ref, arm)`

返回杯子放置动作列表。

## 4. `motion_move_away.py`

用途：生成简单的“靠近障碍物并侧向推开”动作序列。

### `make_move_away_approach_action(pt_ref, arm)`

生成接近障碍物的姿态。

### `make_move_away_push_action(pt_ref, arm)`

生成实际推开的姿态。

### `build_move_away_sequence(pt_ref, arm)`

返回上面两步组成的动作列表。

共用参数：

- `pt_ref: Optional[np.ndarray]`
  障碍物 3D 点。
- `arm: str`
  `"left"` 或 `"right"`。

## 5. `nav_utils.py`

用途：把离散路径点转成底盘旋转 / 前进动作，并执行。

### `path_to_actions(path, init_yaw=0.0)`

将二维路径点序列转换为动作列表。

- `path: List[Tuple[float, float]]`
  平面路径点，至少两点才会产生动作。
- `init_yaw: float = 0.0`
  初始朝向，弧度制。
- 返回值：`List[Tuple[str, float]]`
  例如 `[("rotate", 1.57), ("forward", 0.8)]`。

### `execute_nav_actions(arx, actions, distance)`

执行 `path_to_actions()` 的输出。

- `actions`
  `("rotate", angle)` 或 `("forward", distance)` 组成的列表。
- `distance: float`
  前进时预留的停止距离；如果 `value - distance <= 0`，该段前进会被跳过。
- 返回值：`List[Tuple[float, float]]`
  实际执行过的旋转速度和持续时间，可用于后续反向恢复朝向。

### `recover_rotations(arx, executed_rotations)`

按逆序撤销前面执行过的旋转动作。

## 6. 其他同类脚本

`motion_pick_place_straw.py`、`motion_pick_place_deepbox.py`、`motion_pick_place_normal_object.py` 的接口风格与 `motion_pick_place_cup.py` 一致：

- 单步函数：输入 `pt_ref` 和 `arm`，输出单臂 7 维动作字典
- 序列函数：`build_pick_*_sequence()` / `build_place_*_sequence()`
- 区别主要在固定偏移量、夹爪开合值、抬升高度和姿态设置
