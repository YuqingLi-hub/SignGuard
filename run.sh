#!/bin/bash
#SBATCH --output=/fred/oz410/project/FL/SignGuard/outputs/%x-out.txt  # %x = job name
#SBATCH --error=/fred/oz410/project/FL/SignGuard/outputs/%x-error.txt # %x = job name
#SBATCH --ntasks=1
#SBATCH -c 6
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1   # Request GPU if needed
#SBATCH -A oz410
#SBATCH -D /fred/oz410/project/FL/SignGuard

cd /fred/oz410/project/FL/SignGuard

module load gcc/13.3.0
module load python/3.12.3
module load cuda/11.7.0

source /fred/oz410/venv/pytorch/bin/activate
JOB_NAME=$1
ATTACK_TYPE=$2
python -u federated_main.py --agg_rule SignGuard --attack "$ATTACK_TYPE"
# python -u federated_main.py
# python -u federated_main.py --agg_rule AlignIns --attack "$ATTACK_TYPE" --local_iter 2