#!/usr/bin/env bash
#SBATCH --job-name=gen_ptacek3000_decr_irony_toxic
#SBATCH --partition=shared_a5000
#SBATCH --qos=a6000_restricted
#SBATCH --account=research_vilenchik
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=80:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/truenas/home/chanel/BERTmodels/logs/gen_datasets_ptacek3000_decreased_irony_toxic/slurm_%j.out
#SBATCH --error=/truenas/home/chanel/BERTmodels/logs/gen_datasets_ptacek3000_decreased_irony_toxic/slurm_%j.err
#
# Submit via Open OnDemand Job Composer (or `sbatch submit_gen_datasets_ptacek3000_decreased_irony_toxic.sbatch`
# directly). Runs OUTSIDE any container by default on the compute node, so this
# wraps the actual training loop in the same apptainer image + GPU passthrough
# this dev session runs in. run_gen_datasets_ptacek3000_decreased_irony_toxic.sh
# handles conda activation itself once inside the container.
set -uo pipefail

mkdir -p /truenas/home/chanel/BERTmodels/logs/gen_datasets_ptacek3000_decreased_irony_toxic

IMAGE="/truenas/sif_images/pytorch_cuda12.6_ngc_conda_vscode.sif"

apptainer exec --nv --bind /truenas "${IMAGE}" \
  bash /truenas/home/chanel/BERTmodels/run_gen_datasets_ptacek3000_decreased_irony_toxic.sh
