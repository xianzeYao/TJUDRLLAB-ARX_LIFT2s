#!/bin/bash

workspace=$(pwd)
source ~/.bashrc

# CAN
gnome-terminal -t "can5" -x sudo bash -c "cd ${workspace};cd ..  ; cd ARX_CAN/arx_can; sudo bash arx_can5.sh; exec bash;"

sleep 1

#body&lift
gnome-terminal -t "body&lift" -x  bash -c "cd ${workspace}; cd .. ; cd body/ROS2; source install/setup.bash && ros2 launch arx_lift_controller lift.launch.py; exec bash;"
