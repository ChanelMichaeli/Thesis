# Thesis
Sarcasm detection poses a persistent challenge in natural language processing due to its inherent ambiguity and reliance on contextual and pragmatic cues. This study introduces an affective dimension-aware framework aimed at improving the generalizability of sarcasm detection models across diverse domains.
We propose a taxonomy of sarcasm affective dimensions—humor, irony, and toxicity and analyze their distribution across eight benchmark datasets. Our empirical analysis reveals significant affective dimensions distributional discrepancies that correlate with degradation in cross-domain performance.
To mitigate these challenges, we implement an affective dimensions-targeted upsampling strategy that selectively enriches underrepresented affective dimensions in the training data. Experimental results demonstrate that our approach improves cross-domain performance, and with a dramatic effect when discrepancies are large.

## Setup

```bash
pip install torch transformers accelerate tensorflow scikit-learn pandas numpy scipy statsmodels matplotlib seaborn emoji
```

All commands below are run from the repository root. `runBERTmodels.py` has Windows paths as defaults, so always pass `--absolute_path`, `--output_model_dir` and `--scores_output_dir` (or `--scores_output_file`) explicitly. Paths given to `--datasets` are relative to `--absolute_path`, and a trailing `/` on `--absolute_path` is required.

## Stage 1: Prepare the datasets

Every dataset is a CSV with a `text` column and a binary `label` column (`1` = sarcastic, `0` = not sarcastic), balanced between the two classes. The scripts expect this layout:

```
datasets/
  <DATASET>/train_data_filtered.csv
  <DATASET>/test_data_filtered.csv
```

with `<DATASET>` one of `Ptacek`, `Riloff`, `SARC`, `SPIRS`, `SemEval2022`, `iSarcasmEval`, `MUStARD`, `multimodal_sarcasm_detection`. The folder name is used as the dataset name in all result tables.

## Stage 2: Cross-domain baseline (train on one dataset, test on all others)

`runBERTmodels.py` fine-tunes a model with stratified k-fold CV (`bertFineTuneKfolds.py`). In `training` mode the **first** dataset in `--datasets` is the training set and all the others are cross-domain test sets. Each fold is scored on its held-out part (in-domain) and on every test set (cross-domain).

```bash
python runBERTmodels.py \
  --absolute_path "$PWD/" \
  --model_mode training \
  --model_name bert-base-uncased \
  --n_folds 5 \
  --datasets "datasets/Riloff/train_data_filtered.csv datasets/Ptacek/test_data_filtered.csv datasets/SARC/test_data_filtered.csv datasets/SPIRS/test_data_filtered.csv datasets/SemEval2022/test_data_filtered.csv datasets/iSarcasmEval/test_data_filtered.csv datasets/MUStARD/test_data_filtered.csv datasets/multimodal_sarcasm_detection/test_data_filtered.csv" \
  --output_model_dir "$PWD/savedModels_finetuned/Riloff" \
  --scores_output_file "$PWD/scores_results_finetuned/Riloff.csv"
```

Repeat with each dataset as the training set. The scores CSV has one row per test set (plus a `<DATASET> (CV heldout)` row) with mean and std of F1 (binary and macro), precision, recall and accuracy across folds. The model is saved to `--output_model_dir/model_state_dict.pth`.

Any Hugging Face model can be passed to `--model_name` (e.g. `roberta-base`).

## Stage 3: Train the affective-dimension classifiers

Train one classifier per affective dimension (humor, irony, toxicity) on an external labelled dataset for that dimension, in the same `text`/`label` format. Use the folder names `humor`, `irony` and `toxcity`: the folder name becomes the probability column name (`humor_probs`, `irony_probs`, `toxcity_probs`) that the later stages read.

```bash
for dim in humor irony toxcity; do
  python runBERTmodels.py \
    --absolute_path "$PWD/" \
    --model_mode training \
    --model_name bert-base-uncased \
    --datasets "aux_datasets/${dim}/train_data.csv aux_datasets/${dim}/test_data.csv" \
    --output_model_dir "$PWD/savedModels_finetuned_aux/${dim}" \
    --scores_output_file "$PWD/scores_results_aux/${dim}.csv"
done
```

## Stage 4: Label the sarcastic samples with affective dimensions

`affective_eval` mode loads a trained classifier, runs it on the **sarcastic** (`label == 1`) rows of every dataset given, and writes the per-sample probability to `<scores_output_dir>/<DATASET>/<file>.csv`. Running the three classifiers into the same directory adds one column per dimension to the same file.

```bash
SETS="datasets/Ptacek/train_data_filtered.csv datasets/Ptacek/test_data_filtered.csv datasets/Riloff/test_data_filtered.csv datasets/SARC/test_data_filtered.csv datasets/SPIRS/test_data_filtered.csv datasets/SemEval2022/test_data_filtered.csv datasets/iSarcasmEval/test_data_filtered.csv datasets/MUStARD/test_data_filtered.csv datasets/multimodal_sarcasm_detection/test_data_filtered.csv"

for dim in humor irony toxcity; do
  python runBERTmodels.py \
    --absolute_path "$PWD/" \
    --model_mode affective_eval \
    --model_name "$PWD/savedModels_finetuned_aux/${dim}" \
    --datasets "$SETS" \
    --scores_output_dir "$PWD/scores_results_finetuned/aux_labels"
done
```

Do not put a trailing `/` on `--model_name` here, since the column name is taken from its last path segment. Include the **train** split of any dataset you want to enrich in Stage 6.

## Stage 5: Affective discrepancy vs. cross-domain degradation

`sublabels_analysis.py` holds the analysis functions: KL divergence between the affective distributions of a train/test pair (`kl_div_f1`), correlation with the in-domain minus cross-domain F1 gap (`corr_p_val_plot`), partial correlations controlling for generic shift (`partial_corr_analysis`), QAP permutation tests (`qap_partial_corr_test`) and leave-one-dataset-out checks (`leave_one_dataset_out`).

First compute the generic (non-affective) domain-shift controls, lexical JS divergence and BERT-embedding Fréchet distance, for every dataset pair:

```bash
python -c "
import pandas as pd
from generic_shift import compute_generic_shift_table
names = ['Ptacek','Riloff','SARC','SPIRS','SemEval2022','iSarcasmEval','MUStARD','multimodal_sarcasm_detection']
data = {n: pd.read_csv(f'datasets/{n}/test_data_filtered.csv') for n in names}
compute_generic_shift_table(data).to_csv('generic_kl_divergence_results.csv', index=False)
"
```

Then test whether affective KL still explains the F1 gap after controlling for generic shift:

```bash
python generic_shift_control_analysis.py
```

This writes `generic_shift_control_analysis_table_demo46.csv`, `partial_corr_results_demo46.csv` and `partial_corr_leave_one_dataset_out_demo46.csv`. The train/test F1 pairs and the humor/toxicity distributions are defined at the top of the script; update them with your Stage 2 and Stage 4 results.

## Stage 6: Affective-targeted enrichment (upsampling)

`enrichment.py` builds new training sets by adding sarcastic samples from a source pool, selected by their affective probabilities (from Stage 4):

| Function | Samples added |
|---|---|
| `enrichment_single_label(..., label, pos_samples)` | `label > 0.75` (`pos_samples=True`) or `label < 0.25` (`pos_samples=False`) |
| `enrichment_two_label(..., label1, label2, pos_samples)` | both labels above 0.75 / below 0.25 |
| `enrichment_randomly(...)` | random sarcastic samples (control) |
| `enrichment_lexically_matched(..., target_domain_path)` | samples lexically closest to a target domain (control) |

`enrichment_size` is the fraction of the training set size to add. With `balance=True` an equal number of non-sarcastic samples is added too; with `balance=False` only sarcastic samples are added.

Example: enrich a Ptacek training set with 50% sarcastic samples that are low in both irony and toxicity, plus the matching random control:

```bash
mkdir -p datasets/Ptacek/upsampling_train_imbalanced

python -c "
from enrichment import enrichment_two_label
enrichment_two_label(
    train_data_path='datasets/Ptacek/train_data_filtered3000.csv',
    source_path='datasets/Ptacek/train_data_filtered.csv',
    prob_source_path='scores_results_finetuned/aux_labels/Ptacek/train_data_filtered.csv',
    enrichment_size=0.5, label1='irony_probs', label2='toxcity_probs',
    pos_samples=False, balance=False,
).to_csv('datasets/Ptacek/upsampling_train_imbalanced/Ptacek65_irony_txc_neg0.csv', index=False)
"

python -c "
from enrichment import enrichment_randomly
enrichment_randomly(
    train_data_path='datasets/Ptacek/train_data_filtered3000.csv',
    source_path='datasets/Ptacek/train_data_filtered.csv',
    enrichment_size=0.5, balance=False,
).to_csv('datasets/Ptacek/upsampling_train_imbalanced/Ptacek65_random0.csv', index=False)
"
```

`prob_source_path` must be the Stage 4 output for `source_path`, because its rows are aligned with the sarcastic rows of the source. Generate several files per condition with different seeds (suffixes `0`, `1`, ...) to average over runs.

## Stage 7: Train on the enriched sets and evaluate cross-domain

Run `runBERTmodels.py` in `training` mode with an enriched file as the training set (the first entry of `--datasets`), evaluated on the same cross-domain test sets. Run it once per enriched file:

```bash
python runBERTmodels.py \
  --absolute_path "$PWD/" \
  --model_mode training \
  --model_name roberta-base \
  --n_folds 5 \
  --datasets "datasets/Ptacek/upsampling_train_imbalanced/Ptacek65_irony_txc_neg0.csv datasets/MSTI/test_data_filtered.csv datasets/MUStARD/test_data_filtered.csv datasets/Ptacek/test_data_filtered.csv datasets/Riloff/test_data_filtered.csv datasets/SARC/test_data_filtered.csv datasets/SPIRS/test_data_filtered.csv datasets/SemEval2022/test_data_filtered.csv datasets/iSarcasmEval/test_data_filtered.csv datasets/multimodal_sarcasm_detection/test_data_filtered.csv" \
  --output_model_dir "$PWD/savedModels_finetuned/roberta-base/Ptacek_aug65/Ptacek65_irony_txc_neg0" \
  --scores_output_file "$PWD/scores_results_finetuned/roberta-base/Ptacek_aug65/Ptacek65_irony_txc_neg0.csv"
```

Repeat for every condition and seed (e.g. `Ptacek65_irony_neg<seed>`, `Ptacek65_txc_neg<seed>`, `Ptacek65_irony_txc_neg<seed>`, `Ptacek65_random<seed>`), changing the training file, `--output_model_dir` and `--scores_output_file` accordingly. Keep all score files of one experiment in the same directory so they can be aggregated.

Aggregate the results across seeds:

```bash
# per experiment (irony / toxic / both / random): mean F1 across seed files, per test set
python aggregate_ptacek_aug65_summary.py \
  --directory scores_results_finetuned/roberta-base/Ptacek_aug65 \
  --output scores_results_finetuned/roberta-base/Ptacek_aug65_summary.csv

# average all result CSVs in each directory into <dir>/aggregated.csv
python aggregate_gen_results.py scores_results_finetuned/roberta-base/Ptacek_aug65
```

## Stage 8: Which samples did the enrichment fix? (affective flip analysis)

Compares three BERT models on the sarcastic test samples: A (base Ptacek set), B (affective enrichment) and R (random enrichment). First save each model's per-sample predictions on the cross-domain test sets (`testing` mode with `--save_predictions`):

```bash
for m in Ptacek3000_retrained Ptacek65_irony_txc_neg0 Ptacek65_random0; do
  python runBERTmodels.py \
    --absolute_path "$PWD/" \
    --model_mode testing \
    --save_predictions \
    --model_name "$PWD/savedModels_finetuned/${m}" \
    --datasets "datasets/SPIRS/test_data_filtered.csv datasets/multimodal_sarcasm_detection/test_data_filtered.csv datasets/SARC/test_data_filtered.csv datasets/SemEval2022/test_data_filtered.csv datasets/iSarcasmEval/test_data_filtered.csv datasets/Riloff/test_data_filtered.csv datasets/MUStARD/test_data_filtered.csv" \
    --scores_output_dir "$PWD/scores_results_finetuned/affective_flip/predictions" \
    --scores_output_file "$PWD/scores_results_finetuned/affective_flip/scores_${m}.csv"
done
```

`testing` mode loads BERT checkpoints (`model_state_dict.pth` saved by Stage 2), so these three models must be trained with `bert-base-uncased`.

Then run the analysis and plots (they also need the Stage 4 affective labels in `scores_results_finetuned/aux_labels`):

```bash
python affective_flip_analysis.py        # group tests, KL profiles, logit model -> scores_results_finetuned/affective_flip/summary.md
python affective_flip_cluster_plots.py   # cluster-2 figures and LaTeX table -> scores_results_finetuned/affective_flip/figures
```
