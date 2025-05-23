import os
import sys
import subprocess
import tempfile


def makejob(commit_id, configpath, nruns):
    return f"""#!/bin/bash

#SBATCH --job-name=iliar_training
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_night
#SBATCH --time=6:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-{nruns}

current_dir=`pwd`
export PATH=$PATH:~/.local/bin

echo "Session " ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
echo "Running on " $(hostname)

# Copy the source directory and data
echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude logslurms --exclude configs . $TMPDIR/code

echo "Checking out the correct version of the code commit_id {commit_id}"
cd $TMPDIR/code
git checkout {commit_id}

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
python -m torchtmpl.minimal_train {configpath} train

# Exit with error code if training fails
if [[ $? != 0 ]]; then
    echo "Training script failed"
    exit 1
fi
"""


def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")


# Ensure all the modified files have been staged and commited
# This is to guarantee that the commit id is a reliable certificate
# of the version of the code you want to evaluate
result = int(
    subprocess.run(
        "expr $(git diff --name-only | wc -l) + $(git diff --name-only --cached | wc -l)",
        shell=True,
        stdout=subprocess.PIPE,
    ).stdout.decode()
)
if result > 0:
    print(f"We found {result} modifications either not staged or not commited")
    raise RuntimeError(
        "You must stage and commit every modification before submission "
    )

commit_id = subprocess.check_output(
    "git log --pretty=format:'%H' -n 1", shell=True
).decode()

print(f"I will be using the commit id {commit_id}")

# Ensure the log directory exists
os.system("mkdir -p logslurms")

if len(sys.argv) not in [2, 3]:
    print(f"Usage : {sys.argv[0]} config.yaml <nruns|1>")
    sys.exit(-1)

configpath = sys.argv[1]
if len(sys.argv) == 2:
    nruns = 1
else:
    nruns = int(sys.argv[2])

# Copy the config in a temporary config file
os.system("mkdir -p configs")
tmp_configfilepath = tempfile.mkstemp(dir="./configs", suffix="-config.yml")[1]
os.system(f"cp {configpath} {tmp_configfilepath}")

# Launch the batch jobs
submit_job(makejob(commit_id, tmp_configfilepath, nruns))