#!/bin/bash

workspace=$(pwd)
source ~/.bashrc

# CAN
gnome-terminal -t "can1" -x sudo bash -c "cd ${workspace};cd ..  ; cd Src/LIFT/ARX_CAN/arx_can; sudo bash arx_can1.sh; exec bash;"
sleep 0.5
gnome-terminal -t "can3" -x sudo bash -c "cd ${workspace};cd ..  ; cd Src/LIFT/ARX_CAN/arx_can; sudo bash arx_can3.sh; exec bash;"
sleep 0.5
gnome-terminal -t "can5" -x sudo bash -c "cd ${workspace};cd ..  ; cd Src/LIFT/ARX_CAN/arx_can; sudo bash arx_can5.sh; exec bash;"

sleep 1

#body&lift
gnome-terminal -t "body&lift" -x  bash -c "cd ${workspace}; cd .. ; cd Src/LIFT/body/ROS2; source install/setup.bash && ros2 launch arx_lift_controller lift.launch.py; exec bash;"

sleep 1

#left&right arm
gnome-terminal -t "left&right" -x  bash -c "cd ${workspace}; cd .. ; cd Src/LIFT/ARX_X5/ROS2/X5_ws; source install/setup.bash && ros2 launch arx_x5_controller v2_double_arm.launch.py; exec bash;"
# gnome-terminal -t "left" -x  bash -c "cd ${workspace}; cd .. ; cd Src/LIFT/ARX_X5/ROS2/X5_ws; source install/setup.bash && ros2 launch arx_x5_controller v2_single_arm.launch.py; exec bash;"

sleep 1

# camera_h
gnome-terminal -t "h_camera" -x  bash -c "cd ${workspace}; cd .. ; cd Src/LIFT/realsense; \
  source install/setup.bash && ros2 launch realsense2_camera rs_launch.py\
  align_depth.enable:=true \
  pointcloud.enable:=true \
  enable_sync:=true \
  publish_tf:=true \
  tf_publish_rate:=50.0 \
  camera_name:=camera_h \
  camera_namespace:=camera_h_namespace \
  serial_no:=_409122274317 \
  rgb_camera.enable_auto_white_balance:=false \
  rgb_camera.white_balance:=4600 \
  depth_module.color_profile:=640x480x60 \
  depth_module.depth_profile:=640x480x60 \
  depth_module.enable_auto_exposure:=false \
  depth_module.exposure:=12750 \
  depth_module.gain:=16;exec bash;"

sleep 1
# camera_l
gnome-terminal -t "l_camera" -x  bash -c "cd ${workspace}; cd .. ; cd Src/LIFT/realsense; \
  source install/setup.bash && ros2 launch realsense2_camera rs_launch.py\
  align_depth.enable:=true \
  pointcloud.enable:=true \
  enable_sync:=true \
  publish_tf:=true \
  tf_publish_rate:=50.0 \
  camera_name:=camera_l \
  camera_namespace:=camera_l_namespace \
  serial_no:=_409122272587 \
  rgb_camera.enable_auto_white_balance:=false \
  rgb_camera.white_balance:=4600 \
  depth_module.color_profile:=640x480x60 \
  depth_module.depth_profile:=640x480x60 \
  depth_module.enable_auto_exposure:=false \
  depth_module.exposure:=12750 \
  depth_module.gain:=16;exec bash;"

sleep 1
# camera_r
gnome-terminal -t "r_camera" -x  bash -c "cd ${workspace}; cd .. ; cd Src/LIFT/realsense; \
  source install/setup.bash && ros2 launch realsense2_camera rs_launch.py\
  align_depth.enable:=true \
  pointcloud.enable:=true \
  enable_sync:=true \
  publish_tf:=true \
  tf_publish_rate:=50.0 \
  camera_name:=camera_r \
  camera_namespace:=camera_r_namespace \
  serial_no:=_409122272707 \
  rgb_camera.enable_auto_white_balance:=false \
  rgb_camera.white_balance:=4600 \
  depth_module.color_profile:=640x480x60 \
  depth_module.depth_profile:=640x480x60 \
  depth_module.enable_auto_exposure:=false \
  depth_module.exposure:=12750 \
  depth_module.gain:=16;exec bash;"