#!/bin/bash
#SBATCH -J pgan_cf_
#SBATCH -A eecs
#SBATCH -p dgx2
#SBATCH -o logs/out_tr.out
#SBATCH -e logs/out_tr.err
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 6-23:59:59
#SBATCH --gres=gpu:4
source ../stylegan2-ada-pytorch/env/bin/activate
#conda --init bash
#conda activate pg
CUDA_HOME=/apps/cuda/ python -W ignore train.py --outdir=./training-cifar-lt/ --cfg=stylegan2 --data=./datasets/cifar10  --fname=lt_100.json --gpus=4 --cond=True --mirror=True --snap=100 --kimg=25000 --dlr=0.002 --map-depth=2 --batch=128  --batch-gpu=32 --fp32=False --desc=normal_2
#python dataset_tool.py --source cifar-10-python.tar.gz --dest data/cifar10 --width 32 --height 32 --transform center-crop
