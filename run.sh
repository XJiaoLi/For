#!/bin/bash
#JSUB -q gpu
#JSUB -gpgpu 1
#JSUB -app default
#JSUB -n 20
#JSUB -e error.%J
#JSUB -o output.%J
#JSUB -J my_job

source /apps/software/anaconda3/etc/profile.d/conda.sh
conda activate  jd
unset PYTHONPATH

nohup python  -u train.py  >run.log 2>&1 &


