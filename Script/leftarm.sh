#!/bin/bash 

workspace=$(pwd)
source ~/.bashrc

# 左臂can拉起
gnome-terminal -t "can1" -x sudo bash -c "cd ${workspace};cd .. ; cd ARX_CAN/arx_can; sudo bash arx_can1.sh; exec bash;"

sleep 1

# 左臂node拉起 
gnome-terminal -t "arm_L" -x  bash -c "cd ${workspace}; cd .. ; cd ARX_X5/ROS2/X5_ws; source install/setup.bash && ros2 launch arx_x5_controller v2_single_arm.launch.py; exec bash;"

