#!/bin/bash

#SBATCH --job-name=iliar_training
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_night
#SBATCH --time=6:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-1

current_dir=`pwd`
export PATH=$PATH:~/.local/bin

echo "Session " ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
echo "Running on " $(hostname)

# Copy the source directory and data
echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude logslurms --exclude configs . $TMPDIR/code

echo "Checking out the correct version of the code commit_id f33a0b7a401f50ffb0f0b858203f0d8fcbdba17e"
cd $TMPDIR/code
git checkout f33a0b7a401f50ffb0f0b858203f0d8fcbdba17e

# Set up the virtual environment
echo "Setting up the virtual environment"
python3 -m venv venv
source venv/bin/activate


echo "Upgrading pip and installing build tools"
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel

if ! python -m pip install .; then
    echo "ERROR: Failed to install the library"
    exit 1
fi


# WandB login with API key
export WANDB_API_KEY="57a2afe3e0bb21e7136a1fe7ae58990347fa95c7"  # Replace with your actual WandB API key
echo "Logging into WandB"
wandb login $WANDB_API_KEY
# Start training
echo "Training"
python -m torchtmpl.minimal_train /usr/users/avr/avr_11/hammou1/hammou/ILIAR/configs/tmpg8uqhhv6-config.yml train

# Exit with error code if training fails
if [[ $? != 0 ]]; then
    echo "Training script failed"
    exit 1
fi
