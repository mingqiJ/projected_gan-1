#!/bin/bash
#SBATCH -J pgan_cf_
#SBATCH -A eecs
#SBATCH -p gpu
#SBATCH -o logs/out_trw0.35.out
#SBATCH -e logs/out_trw0.35.err
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 6-23:59:59
#SBATCH --gres=gpu:4
conda activate pg
CUDA_HOME=/apps/cuda/ python -W ignore train.py --outdir=./training-cifar-lt/ --cfg=stylegan2 --data=../datasets/cifar10  --fname=lt_100.json --gpus=4 --cond=True --mirror=True --snap=200 --kimg=25000 --map-depth=2 --batch=128  --batch-gpu=32 --fp32=False --t_start_kimg=4000 --t_end_kimg=8000 --desc=w0.35
#python dataset_tool.py --source cifar-10-python.tar.gz --dest data/cifar10 --width 32 --height 32 --transform center-crop
