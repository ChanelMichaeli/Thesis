# Thesis
Sarcasm detection poses a persistent challenge in natural language processing due to its inherent ambiguity and reliance on contextual and pragmatic cues. This study introduces an affective dimension-aware framework aimed at improving the generalizability of sarcasm detection models across diverse domains.
We propose a taxonomy of sarcasm affective dimensions—humor, irony, and toxicity and analyze their distribution across eight benchmark datasets. Our empirical analysis reveals significant affective dimensions distributional discrepancies that correlate with degradation in cross-domain performance.
To mitigate these challenges, we implement an affective dimensions-targeted upsampling strategy that selectively enriches underrepresented affective dimensions in the training data. Experimental results demonstrate that our approach improves cross-domain performance, and with a dramatic effect when discrepancies are large. 

Stages:
1. convert all sarcasm datasets to CVS files and ensure balanced datasets for unity. See Thesis.ipynb file.
2. Train each data set and test on all other datasets. The fine tuned bert model is in bertFineTuneKfolds.py and to use it run the script runBERTmodels.py , for examples:
   
    !python runBERTmodels.py  --datasets "Riloff/train_data.csv SemEval2022/test_data_filtered.csv MUStARD/test_data_filtered.csv SPIRS/test_data_filtered.csv iSarcasmEval/test_data_filtered.csv multimodal_sarcasm_detection/test_data_filtered.csv MSTI/test_data_filtered.csv SARC/test_data_filtered.csv Ptacek/test_data_filtered.csv Riloff/test_data.csv"
   
   The first data set is the training set and all other are the testing sets.
4. To get the affective dimensions of the sarcasm datasets , see evaluation_datasets_all_sublabels.ipynb.
5. For the enrichment stage see enrichment.ipynb
