#!/bin/bash

# Get job name from command-line argument or default to "IDNtest_default"
JOB_NAME=${1:-SignGuard_AlignIns_test_default}

# for a in random sign_flip noise label_flip lie byzMean min_max min_sum non nan zero
# # ['random', 'sign_flip', 'noise', 'label_flip', 'lie', 'byzMean', 'min_max', 'min_sum', 'non','nan','zero']
# # for a in nan zero
# do
#     echo "Submitting job with attack type: $a"
    
#     # Submit the job with job name and attack type as arguments
#     sbatch --job-name="$JOB_NAME-$a" run.sh "$JOB_NAME" "$a"
# done
# Submit the job dynamically with the correct job name
sbatch --job-name="$JOB_NAME" run.sh "$JOB_NAME" byzMean