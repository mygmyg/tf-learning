#!/usr/bin/env bash
# This script creates a conda environment on a vanilla Ubuntu machine
# from scratch.

sudo apt-get -y install git
cd ~
git clone -q git@qianyi301.xicp.net:southlake/SR.git
cd SR
. install/install-conda-on-linux.sh
conda env create -f install/env-gpu.yml
source activate SR
pip install -e .