#!/bin/bash
echo "Setup Ubuntu and ROS Kinetic"
#  Define 
DOWNLOAD_DIR=/home/$USER/Downloads
## Add user to sudo
sudo usermod -aG sudo $USER
## Update
sudo apt-get update 
sudo apt-get upgrade
sudo apt-get install -y wget
## Download ROS Kinetics
FILE=$DOWNLOAD_DIR/$CONDA_FILENAME
if [ -f "$FILE" ]; then
    echo "$FILE exist"
else 
    echo "Download Anaconda File to $DOWNLOAD_DIR"
    wget -P $DOWNLOAD_DIR https://raw.githubusercontent.com/ROBOTIS-GIT/robotis_tools/master/install_ros_kinetic.sh
fi
## Install ROS
# cd $DOWNLOAD_DIR
# chmod 755 ./install_ros_kinetic.sh && bash ./install_ros_kinetic.sh
# Install ROS kinetics and Manipulator dependencies
# Install Manipulator packages
cd ~/catkin_ws/src/
git clone https://github.com/ROBOTIS-GIT/DynamixelSDK.git
git clone https://github.com/ROBOTIS-GIT/dynamixel-workbench.git
git clone https://github.com/ROBOTIS-GIT/dynamixel-workbench-msgs.git
git clone https://github.com/ROBOTIS-GIT/open_manipulator.git
git clone https://github.com/ROBOTIS-GIT/open_manipulator_msgs.git
git clone https://github.com/ROBOTIS-GIT/open_manipulator_simulations.git
git clone https://github.com/ROBOTIS-GIT/robotis_manipulator.git