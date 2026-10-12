#!/usr/bin/env bash
#SBATCH --job-name=ptacek_aug65_roberta
#SBATCH --partition=shared_a5000
#SBATCH --qos=a6000_restricted
#SBATCH --account=research_vilenchik
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=80:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/truenas/home/chanel/BERTmodels/logs/Ptacek_aug65_roberta-base/slurm_%j.out
#SBATCH --error=/truenas/home/chanel/BERTmodels/logs/Ptacek_aug65_roberta-base/slurm_%j.err
#
# Submit via Open OnDemand Job Composer (or `sbatch submit_ptacek_aug65_roberta.sbatch`
# directly). Runs OUTSIDE any container by default on the compute node, so this
# wraps the actual training loop in the same apptainer image + GPU passthrough
# this dev session runs in. run_ptacek_aug65_roberta.sh handles conda activation
# itself once inside the container.
set -uo pipefail

mkdir -p /truenas/home/chanel/BERTmodels/logs/Ptacek_aug65_roberta-base

IMAGE="/truenas/sif_images/pytorch_cuda12.6_ngc_conda_vscode.sif"

apptainer exec --nv --bind /truenas "${IMAGE}" \
  bash /truenas/home/chanel/BERTmodels/run_ptacek_aug65_roberta.sh
