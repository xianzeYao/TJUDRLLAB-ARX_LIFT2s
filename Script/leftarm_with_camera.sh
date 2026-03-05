#!/bin/bash 

workspace=$(pwd)
source ~/.bashrc

# 左臂can拉起
gnome-terminal -t "can1" -x sudo bash -c "cd ${workspace};cd .. ; cd ARX_CAN/arx_can; sudo bash arx_can1.sh; exec bash;"

sleep 1

# 左臂node拉起 
gnome-terminal -t "arm_L" -x  bash -c "cd ${workspace}; cd .. ; cd ARX_X5/ROS2/X5_ws; source install/setup.bash && ros2 launch arx_x5_controller v2_single_arm.launch.py; exec bash;"

# 左臂相机拉起
gnome-terminal -t "left_camera" -x  bash -c "cd ${workspace}; cd ../.. ; cd realsense; source install/setup.bash && ros2 launch realsense2_camera rs_launch.py \
  camera_name:=camera_l \
  serial_no:=_409122272587 \
  depth_module.color_profile:=640x480x90 \
  depth_module.depth_profile:=640x480x90; exec bash;"
