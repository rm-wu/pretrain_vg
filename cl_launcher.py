import os
import sys

folder = "/home/riccardom/pretrain_vg"
model = "r50"
batch_size = 64
train_batch_size = f"--train_batch_size {batch_size}"

if not os.path.abspath(os.curdir) == folder:
    sys.exit()

os.makedirs(f"{folder}/jobs", exist_ok=True)
os.makedirs(f"{folder}/out_job", exist_ok=True)


# for
exp_name = f"pretrain_gldv2_r50_heavy_bs64"
filename = f"{folder}/jobs/{exp_name}.job"

content = ("#!/bin/bash \n" +
           f"#SBATCH --job-name={exp_name} \n" +
           "#SBATCH --gres=gpu:8 \n" +
           "#SBATCH --cpus-per-task=24 \n" + 
           "#SBATCH --mem=55GB  \n" +
           "#SBATCH --time=48:00:00 \n" +
           f"#SBATCH --output={folder}/out_job/out_{exp_name}.txt \n"+
           f"#SBATCH --error={folder}/out_job/err_{exp_name}.txt \n" +
           "ml purge \nml Python/3.6.6-gomkl-2018b \n" +
           "source /home/riccardom/pretrain_vg/myvenv/bin/activate \n" +
           f"python main.py --exp_name {exp_name} --arch {model} {train_batch_size} ")

with open(filename, "w") as file:
    _ = file.write(content)

_ = os.system(f"sbatch {filename}") 

