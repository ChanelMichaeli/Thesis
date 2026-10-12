#!/usr/bin/env bash
# Fine-tune + cross-domain CV eval for every generated Ptacek training file
# under datasets/gen_datasets/Ptacek (recursively), one training run per file.
# Continues past a failed file (logged) rather than aborting the whole batch.
#
# Run from inside the apptainer terminal:
#   bash run_gen_datasets_ptacek.sh
set -uo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /truenas/home/chanel/conda_envs/bertmodels

cd /truenas/home/chanel/BERTmodels

ABS_PATH="/truenas/home/chanel/BERTmodels/"
GEN_DIR="datasets/gen_datasets/Ptacek"
LOG_DIR="${ABS_PATH}logs/gen_datasets_ptacek"
mkdir -p "$LOG_DIR"

# same cross-domain sarcasm test bench used for the earlier Ptacek augmentation run
TEST_DATASETS="datasets/MUStARD/test_data_filtered.csv datasets/Ptacek/test_data_filtered.csv datasets/Riloff/test_data_filtered.csv datasets/SARC/test_data_filtered.csv datasets/SPIRS/test_data_filtered.csv datasets/SemEval2022/test_data_filtered.csv datasets/iSarcasmEval/test_data_filtered.csv datasets/multimodal_sarcasm_detection/test_data_filtered.csv"

while IFS= read -r train_file; do
  rel_no_ext="${train_file#${ABS_PATH}${GEN_DIR}/}"
  rel_no_ext="${rel_no_ext%.csv}"

  train_arg="${GEN_DIR}/${rel_no_ext}.csv"
  output_model_dir="${ABS_PATH}savedModels_gen/Ptacek/${rel_no_ext}/"
  scores_output_file="${ABS_PATH}score_results_gen/Ptacek/${rel_no_ext}.csv"
  log_file="${LOG_DIR}/${rel_no_ext//\//_}.log"

  # skip if a previous run already finished: saveModel() writes
  # model_state_dict.pth as its very last step, after the scores CSV is
  # written, so requiring both present means the whole run completed
  if [ -s "${scores_output_file}" ] && [ -f "${output_model_dir}model_state_dict.pth" ]; then
    echo "=== [$(date '+%F %T')] skipping ${train_arg} (already trained) ==="
    continue
  fi

  echo "=== [$(date '+%F %T')] training on ${train_arg} -> ${log_file} ==="
  python runBERTmodels.py \
    --absolute_path "${ABS_PATH}" \
    --datasets "${train_arg} ${TEST_DATASETS}" \
    --model_name bert-base-uncased \
    --model_mode training \
    --n_folds 5 \
    --output_model_dir "${output_model_dir}" \
    --scores_output_file "${scores_output_file}" \
    > "${log_file}" 2>&1
  status=$?
  if [ $status -ne 0 ]; then
    echo "FAILED (exit ${status}): ${train_arg} -- see ${log_file}"
  fi
done < <(find "${ABS_PATH}${GEN_DIR}" -type f -name "*.csv" | sort)
