#!/bin/bash
echo "Setup Conda Environment"
# #  Define 
DOWNLOAD_DIR=/home/$USER/Downloads
CONDA_ENV=tf2
PIP_REQUIREMENTS=~/serviceBot/env/requirements.txt
## Add user to sudo
sudo usermod -aG sudo $USER
## Downloads
sudo apt-get install -y wget
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
conda install -y -c conda-forge ipympl -n $CONDA_ENV

# # pip install
echo "Installing PIP requirements"
/home/$USER/anaconda3/envs/$CONDA_ENV/bin/pip install -r $PIP_REQUIREMENTS
# update ipympl for interactive plot
/home/$USER/anaconda3/envs/$CONDA_ENV/bin/pip git+https://github.com/matplotlib/jupyter-matplotlib.git#egg=ipympl 
# Jupyterlab setup
echo "Installing Jupyterlab Extension"
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyter-matplotlib
jupyter labextension install @jupyterlab/toc
# Update and rebuild jupyterlab extenstion
jupyter labextension update --all && jupyter lab build && jupyter labextension list