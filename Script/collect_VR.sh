#!/bin/bash

workspace=$(pwd)
source ~/.bashrc

# CAN
gnome-terminal -t "can1" -x sudo bash -c "cd ${workspace};cd ../../.. ; cd ARX_CAN/arx_can; ./arx_can1.sh; exec bash;"
sleep 0.1
gnome-terminal -t "can3" -x sudo bash -c "cd ${workspace};cd ../../.. ; cd ARX_CAN/arx_can; ./arx_can3.sh; exec bash;"
sleep 0.1
gnome-terminal -t "can5" -x sudo bash -c "cd ${workspace};cd ../../.. ; cd ARX_CAN/arx_can; ./arx_can5.sh; exec bash;"
sleep 1

#body
gnome-terminal -t "body" -x  bash -c "cd ${workspace}; cd ../../..; cd body/ROS2; source install/setup.bash && ros2 launch arx_lift_controller lift.launch.py; exec bash;"
#AC_one pos control
sleep 3
gnome-terminal -t "L" -x  bash -c "cd ${workspace}; cd ../../..; cd ARX_X5/ROS2/X5_ws; source install/setup.bash && ros2 launch arx_x5_controller v2_pos_control.launch.py; exec bash;"

#VR
gnome-terminal -t "unity_tcp" -x  bash -c "cd ${workspace}; cd ../../..; cd ARX_VR_SDK/ROS2; source install/setup.bash && ros2 run serial_port serial_port_node; exec bash;"
sleep 0.1
gnome-terminal -t "arx5_pos_cmd" -x  bash -c "cd ${workspace}; cd ../../..; cd ARX_VR_SDK/ROS2; source install/setup.bash && ros2 topic echo /ARX_VR_L; exec bash;"