#!/bin/bash

###
# CS236781: Deep Learning
# run_experiments.sh
#
# This script runs multiple experiments with varying configurations.
#

###
# Parameters for sbatch
#
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1
MAIL_USER="elad.sznaj@campus.technion.ac.il","roi.teichman@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=Tcs236781-hw

# Experiment configurations
Ks=(32)
Ls=(2 4 8 16)

ROOT_DIR=$(pwd)
HIDDEN_DIMS="512 256 64 16"

divide_by_2() {
  local val=$1
  echo $((val / 2))
}


# Loop through the configurations and submit jobs
for K in "${Ks[@]}"; do
  for L in "${Ls[@]}"; do
    RUN_NAME="exp1_1_L${L}_K${K}"
    JOB_NAME="job_${RUN_NAME}"

    POOL_EVERY_DIVIDED=$(divide_by_2 "${L}")

    sbatch \
      -N $NUM_NODES \
      -c $NUM_CORES \
      --gres=gpu:$NUM_GPUS \
      --job-name $JOB_NAME \
      --mail-user $MAIL_USER \
      --mail-type $MAIL_TYPE \
      -o "slurm-%N-%j.out" \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

echo "*** Python path: \$(python -c 'import sys; print(sys.path)') ***"

# Set PYTHONPATH to include the root directory
export PYTHONPATH=\$PYTHONPATH:$ROOT_DIR

# Run the experiment
# 1.1
python hw2/experiments.py run-exp --run-name $RUN_NAME --filters-per-layer $K --layers-per-block $L --pool-every 8 --hidden-dims $HIDDEN_DIMS --epochs 50 --early-stopping 5 --batches 120

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF

  done
done
