#!/bin/bash
#SBATCH --output=/fred/oz410/project/FL/SignGuard/outputs/R-QIM/%x-out.txt  # %x = job name
#SBATCH --error=/fred/oz410/project/FL/SignGuard/outputs/R-QIM/%x-error.txt # %x = job name
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=6G
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
# ATTACK_TYPE=$2
# python -u federated_main.py --agg_rule SignGuard --attack "$ATTACK_TYPE"
# python -u federated_main.py --agg_rule SignGuard --alpha 0.7 --delta 1 --k 0
python -u federated_main.py --agg_rule SignGuard --alpha 0.51 --delta 0.05 --k 0
# python -u federated_main.py --agg_rule AlignIns --attack "$ATTACK_TYPE" --local_iter 2