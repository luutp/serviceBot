#!/bin/bash
echo "Enter catkin Workspace name:"
read varname
#  Define 
DOWNLOAD_DIR=/home/$USER/Downloads
## Add user to sudo
sudo usermod -aG sudo $USER
name_catkin_workspace=$varname

echo "[Make the catkin workspace: $name_catkin_workspace and test the catkin_make]"
source /opt/ros/kinetic/setup.bash
mkdir -p $HOME/$name_catkin_workspace/src
cd $HOME/$name_catkin_workspace/src
catkin_init_workspace
cd $HOME/$name_catkin_workspace
catkin_make
sh -c "echo \"source ~/$name_catkin_workspace/devel/setup.bash\" >> ~/.bashrc"
echo "[Complete!!!]"