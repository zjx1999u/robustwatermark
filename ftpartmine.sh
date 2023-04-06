#!/bin/bash
#SBATCH -p gpu2
#SBATCH -N 1-1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem 5G
#SBATCH --gres gpu:1
#SBATCH -o ftpart-method-%j.out

date
srun singularity exec --nv /app/singularity/deepo/20190624.sif python ftpartmethod.py --batch_size 100  --dataset cifar10 --max_epochs 30 --lr 0.001   \
--load_path ./chk5/wm/Logit_0.5_snnl_0.01_un_2/use_epoch_39model.pth --usepercent 1.0  --trg_set_size 500 --classes_num 10 \
--wm_path ./data/protecting_ipp/unrelated/cifar10  \
--log_dir ./chk5/wm/Logit_0.5_snnl_0.01_un_2 \
--save_dir ./chk5/wm/Logit_0.5_snnl_0.01_un_2 \
--runname allft --save_model rtal_logit.pth   --reinitll --tunealllayers --method ProtectingIP

srun singularity exec --nv /app/singularity/deepo/20190624.sif python ftpartmethod.py --batch_size 100  --dataset cifar10 --max_epochs 30 --lr 0.001   \
--load_path ./chk5/wm/Logit_0.5_snnl_0.01_un_2/use_epoch_39model.pth --usepercent 0.1  --trg_set_size 500 --classes_num 10 \
--wm_path ./data/protecting_ipp/unrelated/cifar10  \
--log_dir ./chk5/wm/Logit_0.5_snnl_0.01_un_2 \
--save_dir ./chk5/wm/Logit_0.5_snnl_0.01_un_2 \
--runname ft --save_model rtal_logit.pth   --reinitll --tunealllayers --method ProtectingIP
date