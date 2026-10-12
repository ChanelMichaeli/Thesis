#!/usr/bin/env bash
# Fine-tune + 5-fold CV for every generated Ptacek3000 training file under
# generated_datasets/Ptacek3000/added_sentences/decreased_irony_toxic/{50_percent,100_percent},
# one training run per file. Each fold is also evaluated cross-domain against
# all sarcasm test sets (Riloff, SPIRS, MUStARD, iSarcasmEval, SemEval2022,
# SARC, Ptacek, multimodal_sarcasm_detection).
# Models are not kept: each run is trained into a scratch dir that is removed
# right after the run finishes.
# Continues past a failed file (logged) rather than aborting the whole batch.
#
# Run from inside the apptainer terminal:
#   bash run_gen_datasets_ptacek3000_decreased_irony_toxic.sh
set -uo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /truenas/home/chanel/conda_envs/bertmodels

cd /truenas/home/chanel/BERTmodels

ABS_PATH="/truenas/home/chanel/BERTmodels/"
GEN_DIR="generated_datasets/Ptacek3000/added_sentences/decreased_irony_toxic"
LOG_DIR="${ABS_PATH}logs/gen_datasets_ptacek3000_decreased_irony_toxic"
SCRATCH_MODEL_DIR="${ABS_PATH}tmp_trainer/ptacek3000_decreased_irony_toxic_scratch"
mkdir -p "$LOG_DIR"

# cross-domain sarcasm test bench used for the other generated-dataset runs
TEST_DATASETS="datasets/MUStARD/test_data_filtered.csv datasets/Ptacek/test_data_filtered.csv datasets/Riloff/test_data_filtered.csv datasets/SARC/test_data_filtered.csv datasets/SPIRS/test_data_filtered.csv datasets/SemEval2022/test_data_filtered.csv datasets/iSarcasmEval/test_data_filtered.csv datasets/multimodal_sarcasm_detection/test_data_filtered.csv"

for percent_dir in 50_percent 100_percent; do
  while IFS= read -r train_file; do
    rel_no_ext="${train_file#${ABS_PATH}${GEN_DIR}/}"
    rel_no_ext="${rel_no_ext%.csv}"

    train_arg="${GEN_DIR}/${rel_no_ext}.csv"
    scores_output_file="${ABS_PATH}score_results_gen/Ptacek3000/added_sentences/decreased_irony_toxic/${rel_no_ext}.csv"
    log_file="${LOG_DIR}/${rel_no_ext//\//_}.log"
    output_model_dir="${SCRATCH_MODEL_DIR}/${rel_no_ext}"

    # skip if a previous run already finished
    if [ -s "${scores_output_file}" ]; then
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

    # model is not needed, drop it immediately to save disk space
    rm -rf "${output_model_dir}"
  done < <(find "${ABS_PATH}${GEN_DIR}/${percent_dir}" -type f -name "*.csv" | sort)
done

rmdir --ignore-fail-on-non-empty "${SCRATCH_MODEL_DIR}" 2>/dev/null || true
