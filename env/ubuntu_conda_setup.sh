#!/bin/bash
echo "Setup Ubuntu and Conda for ServiceBot Project"
#  Define 
DOWNLOAD_DIR=/home/$USER/Downloads
CONDA_FILENAME=Anaconda3-2019.10-Linux-x86_64.sh
CONDA_ENV=serviceBot
PIP_REQUIREMENTS=~/serviceBot/env/requirements.txt
## Add user to sudo
sudo usermod -aG sudo $USER
## Downloads
sudo apt-get install -y wget
# Install Nvidia driver
sudo apt-add-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install -y nvidia-418
# Install graphviz
sudo apt-get install -y graphviz
# Device manager
sudo apt-get install -y hardinfo
## Download Anaconda
FILE=$DOWNLOAD_DIR/$CONDA_FILENAME
if [ -f "$FILE" ]; then
    echo "$FILE exist"
else 
    echo "Download Anaconda File to $DOWNLOAD_DIR"
    wget -P $DOWNLOAD_DIR https://repo.continuum.io/archive/$CONDA_FILENAME
fi
## 
## Install Anaconda
cd $DOWNLOAD_DIR
bash $CONDA_FILENAME -b

## Settings
## Add Anaconda to PATH and update
PATH=/home/$USER/anaconda3/bin:$PATH
conda info
conda update -y conda
conda update -y anaconda
## Create Conda environment
conda create -y -n $CONDA_ENV python=3.7
conda env list
## Activate
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
# Conda Install
conda install -y nodejs -n $CONDA_ENV
conda install -y tensorflow-gpu -n $CONDA_ENV

# pip install
pip install -r $PIP_REQUIREMENTS
