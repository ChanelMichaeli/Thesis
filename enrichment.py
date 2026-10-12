import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

base_path = "C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/Datasets/"


def rand_sampling(data,number_of_samples):
    downsampling_data = data.sample(n=number_of_samples)
    return downsampling_data


def create_balanced_dataset_from_filtered(
    filtered_sarcastic_path,
    original_train_path,
    output_path=None,
    random_state=42,
):
    """
    Build a balanced binary dataset by pairing a filtered set of sarcastic
    samples with an equally-sized random sample of non-sarcastic samples drawn
    from the original training split.

    Parameters
    ----------
    filtered_sarcastic_path : str
        Path to the CSV that contains sarcastic texts (label=1) with affective
        probability columns but no 'label' column (e.g. Ptacek_train70_like_SPIRS_irony.csv).
    original_train_path : str
        Path to the original training CSV that has at least 'text' and 'label'
        columns. Non-sarcastic samples (label==0) are drawn from here.
    output_path : str or None
        If provided, the resulting dataset is saved to this path as a CSV.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Balanced DataFrame with columns ['text', 'label'] (label 1 = sarcastic,
        label 0 = non-sarcastic), shuffled.
    """
    sarcastic_df = pd.read_csv(filtered_sarcastic_path, encoding='utf-8')
    sarcastic_df = sarcastic_df[['text']].copy()
    sarcastic_df['label'] = 1

    original_df = pd.read_csv(original_train_path, encoding='utf-8')
    non_sarcastic = original_df[original_df['label'] == 0][['text', 'label']].copy()

    # exclude texts that already appear in the sarcastic filtered set
    sarcastic_texts = set(sarcastic_df['text'].tolist())
    non_sarcastic = non_sarcastic[~non_sarcastic['text'].isin(sarcastic_texts)]
    non_sarcastic = non_sarcastic.reset_index(drop=True)

    n_samples = len(sarcastic_df)
    if len(non_sarcastic) < n_samples:
        raise ValueError(
            f"Not enough non-sarcastic samples: need {n_samples}, "
            f"found {len(non_sarcastic)} after deduplication."
        )

    sampled_non_sarcastic = non_sarcastic.sample(n=n_samples, random_state=random_state)

    balanced_df = pd.concat([sarcastic_df, sampled_non_sarcastic], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Sarcastic samples  (label=1): {len(sarcastic_df)}")
    print(f"Non-sarcastic samples (label=0): {len(sampled_non_sarcastic)}")
    print(f"Total balanced dataset size: {len(balanced_df)}")

    if output_path:
        balanced_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Dataset saved to: {output_path}")

    return balanced_df


def enrichment_single_label(train_data_path,source_path,prob_source_path,enrichment_size,label,pos_samples,balance=True):
    train_data = pd.read_csv(train_data_path)
    source_data = pd.read_csv(source_path)
    prob_source = pd.read_csv(prob_source_path)
    
    # source_data = source_data.drop(columns=["id"])
    source_sarcasm = source_data[source_data["label"] == 1]
    prob_source = prob_source.drop(columns=["text"])
    source_sarcasm.reset_index(inplace = True , drop = True)
    
    positive_test_prob =pd.concat([source_sarcasm, prob_source], axis=1, join="inner")
    positive_test_prob = positive_test_prob[~positive_test_prob["text"].isin(train_data["text"])]
    positive_test_prob.reset_index(inplace = True , drop = True)

    if pos_samples:
        positive_highest_prob = positive_test_prob[(positive_test_prob[label]>0.75)]
    else:
        positive_highest_prob = positive_test_prob[(positive_test_prob[label]<0.25)]
    
    pos_samples_number_to_add = min(enrichment_size*len(train_data), len(positive_highest_prob))
    print("size to add to train : ", enrichment_size*len(train_data), ", maximum len to add: ",len(positive_highest_prob))

    data_to_add = rand_sampling(positive_highest_prob,int(pos_samples_number_to_add))
    
    enrichment_train = pd.concat([train_data,data_to_add])
    enrichment_train = enrichment_train.drop_duplicates(keep="first")
    enrichment_train.reset_index(inplace = True , drop = True)
    print("data size after enrichment: ",len(enrichment_train))
    enrichment_train = enrichment_train.sample(frac=1).reset_index(drop=True)
    
    if balance:
        source_negative = source_data[source_data['label'] == 0]
        negative_data_to_add = rand_sampling(source_negative,len(enrichment_train) -len(train_data))
        enrichment_train = pd.concat([enrichment_train,negative_data_to_add])
        enrichment_train = enrichment_train.drop_duplicates(keep="first")
        enrichment_train.reset_index(inplace = True , drop = True)
        enrichment_train = enrichment_train.sample(frac=1).reset_index(drop=True)
        
    return enrichment_train


def enrichment_two_label(train_data_path,source_path,prob_source_path,enrichment_size,label1,label2,pos_samples,balance= True):
    train_data = pd.read_csv(train_data_path)
    source_data = pd.read_csv(source_path)
    prob_source = pd.read_csv(prob_source_path)

    # source_data = source_data.drop(columns=["id"])
    source_sarcasm = source_data[source_data["label"] == 1]
    prob_source = prob_source.drop(columns=["text"])
    source_sarcasm.reset_index(inplace = True , drop = True)
    
    positive_test_prob =pd.concat([source_sarcasm, prob_source], axis=1, join="inner")
    positive_test_prob = positive_test_prob[~positive_test_prob["text"].isin(train_data["text"])]
    positive_test_prob.reset_index(inplace = True , drop = True)

    if pos_samples:
        positive_highest_prob = positive_test_prob[(positive_test_prob[label1]>0.75) & (positive_test_prob[label2]>0.75)]
    else:
        positive_highest_prob = positive_test_prob[(positive_test_prob[label1]<0.25) & (positive_test_prob[label2]<0.25)]

    pos_samples_number_to_add = min(enrichment_size*len(train_data), len(positive_highest_prob))
    data_to_add = rand_sampling(positive_highest_prob,int(pos_samples_number_to_add))
    print("size to add to train : ", enrichment_size*len(train_data), ", maximum len to add: ",len(positive_highest_prob))
    
    enrichment_train = pd.concat([train_data,data_to_add])
    enrichment_train = enrichment_train.drop_duplicates(keep="first")
    enrichment_train.reset_index(inplace = True , drop = True)
    enrichment_train = enrichment_train.sample(frac=1).reset_index(drop=True)
    
    if balance:
        source_negative = source_data[source_data['label'] == 0]
        negative_data_to_add = rand_sampling(source_negative,len(enrichment_train) -len(train_data))
        enrichment_train = pd.concat([enrichment_train,negative_data_to_add])
        enrichment_train = enrichment_train.drop_duplicates(keep="first")
        enrichment_train.reset_index(inplace = True , drop = True)
        enrichment_train = enrichment_train.sample(frac=1).reset_index(drop=True)
        
    return enrichment_train


def add_by_label(train_data_path,source_path,prob_source_path,enrichment_size,balance= True):
    train_data = pd.read_csv(train_data_path)
    source_data = pd.read_csv(source_path)
    prob_source = pd.read_csv(prob_source_path)

    # source_data = source_data.drop(columns=["id"])
    source_sarcasm = source_data[source_data["label"] == 1]
    prob_source = prob_source.drop(columns=["text"])
    source_sarcasm.reset_index(inplace = True , drop = True)
    
    positive_test_prob =pd.concat([source_sarcasm, prob_source], axis=1, join="inner")
    positive_highest_prob = positive_test_prob[(positive_test_prob["irony_probs"]<0.25) & (positive_test_prob["toxcity_probs"]<0.25)]
    positive_highest_prob.reset_index(inplace = True , drop = True)

    pos_samples_number_to_add = min(enrichment_size*len(train_data), len(positive_highest_prob))
    data_to_add = rand_sampling(positive_highest_prob,int(pos_samples_number_to_add))
    print("size to add to train : ", enrichment_size*len(train_data), ", maximum len to add: ",len(positive_highest_prob))
    
    humor_neg_prob = positive_test_prob[(positive_test_prob["humor_probs"]<0.25)]

    pos_samples_number_to_add = min(enrichment_size*len(train_data), len(humor_neg_prob))
    data_to_add = rand_sampling(positive_highest_prob,int(pos_samples_number_to_add))
    print("size to add to train : ", enrichment_size*len(train_data), ", maximum len to add: ",len(positive_highest_prob))
    
    enrichment_train = pd.concat([train_data,data_to_add])
    enrichment_train = enrichment_train.drop_duplicates(keep="first")
    enrichment_train.reset_index(inplace = True , drop = True)
    enrichment_train = enrichment_train.sample(frac=1).reset_index(drop=True)
    
    if balance:
        source_negative = source_data[source_data['label'] == 0]
        negative_data_to_add = rand_sampling(source_negative,len(enrichment_train) -len(train_data))
        enrichment_train = pd.concat([enrichment_train,negative_data_to_add])
        enrichment_train = enrichment_train.drop_duplicates(keep="first")
        enrichment_train.reset_index(inplace = True , drop = True)
        enrichment_train = enrichment_train.sample(frac=1).reset_index(drop=True)
        
    return enrichment_train


def enrichment_lexically_matched(train_data_path,source_path,target_domain_path,enrichment_size,balance=True,random_state=None):
    """
    Generic-shift-matched augmentation baseline: like enrichment_randomly,
    but instead of uniform-random sampling from the source pool, samples the
    additional examples whose lexical (TF-IDF unigram+bigram) profile is
    closest to the target test domain - blind to any affective score.

    This isolates lexical/topical similarity from affective similarity: if
    enrichment_single_label/enrichment_two_label (which select by affect
    score) still outperform this at equal enrichment_size, that is evidence
    the affective signal used for selection carries information beyond
    generic lexical/topical overlap with the target domain, addressing the
    concern that affect-selected examples might win purely by having better
    generic (non-affective) coverage of the target domain.

    Parameters mirror enrichment_randomly, with target_domain_path added:
    the CSV (e.g. the cross-domain test set) whose 'text' column defines the
    domain we're trying to match source examples to. `random_state` seeds the
    negative-side balancing sample and the final shuffle, for reproducibility
    across repeated runs (the positive selection itself is deterministic:
    always the top-N most lexically similar examples).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_data = pd.read_csv(train_data_path)
    source_data = pd.read_csv(source_path)
    target_data = pd.read_csv(target_domain_path)

    source_sarcasm = source_data[source_data["label"] == 1]
    source_sarcasm = source_sarcasm[~source_sarcasm["text"].isin(train_data["text"])]
    source_sarcasm.reset_index(inplace=True, drop=True)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorizer.fit(pd.concat([source_sarcasm["text"], target_data["text"]]).astype(str))
    source_vectors = vectorizer.transform(source_sarcasm["text"].astype(str))
    target_centroid = vectorizer.transform(target_data["text"].astype(str)).mean(axis=0)
    target_centroid = np.asarray(target_centroid)

    similarity = cosine_similarity(source_vectors, target_centroid).ravel()
    source_sarcasm = source_sarcasm.assign(_target_similarity=similarity)
    source_sarcasm = source_sarcasm.sort_values("_target_similarity", ascending=False)

    pos_samples_number_to_add = min(enrichment_size*len(train_data), len(source_sarcasm))
    print("size to add to train :", enrichment_size*len(train_data), ", maximum len to add: ",len(source_sarcasm))
    data_to_add = source_sarcasm.head(int(pos_samples_number_to_add)).drop(columns=["_target_similarity"])

    enrichment_train = pd.concat([train_data,data_to_add])
    enrichment_train.reset_index(inplace = True , drop = True)
    enrichment_train = enrichment_train.sample(frac=1, random_state=random_state).reset_index(drop=True)
    enrichment_train = enrichment_train.drop_duplicates(keep="first")

    if balance:
        source_negative = source_data[source_data['label'] == 0]
        negative_data_to_add = source_negative.sample(n=len(enrichment_train) -len(train_data), random_state=random_state)
        enrichment_train = pd.concat([enrichment_train,negative_data_to_add])
        enrichment_train = enrichment_train.drop_duplicates(keep="first")
        enrichment_train.reset_index(inplace = True , drop = True)
        enrichment_train = enrichment_train.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return enrichment_train


def enrichment_randomly(train_data_path,source_path,enrichment_size,balance= True):
    train_data = pd.read_csv(train_data_path)
    source_data = pd.read_csv(source_path)
    
    # source_data = source_data.drop(columns=["id"])
    source_sarcasm = source_data[source_data["label"] == 1]
    source_sarcasm = source_sarcasm[~source_sarcasm["text"].isin(train_data["text"])]
    source_sarcasm.reset_index(inplace = True , drop = True)
    
    pos_samples_number_to_add = min(enrichment_size*len(train_data), len(source_sarcasm))
    data_to_add = rand_sampling(source_sarcasm,int(pos_samples_number_to_add))
    print("size to add to train :", enrichment_size*len(train_data), ", maximum len to add: ",len(source_sarcasm))
    
    enrichment_train = pd.concat([train_data,data_to_add])
    enrichment_train.reset_index(inplace = True , drop = True)
    enrichment_train = enrichment_train.sample(frac=1).reset_index(drop=True)
    enrichment_train = enrichment_train.drop_duplicates(keep="first")

    if balance:
        source_negative = source_data[source_data['label'] == 0]
        negative_data_to_add = rand_sampling(source_negative,len(enrichment_train) -len(train_data))
        enrichment_train = pd.concat([enrichment_train,negative_data_to_add])
        enrichment_train = enrichment_train.drop_duplicates(keep="first")
        enrichment_train.reset_index(inplace = True , drop = True)
        enrichment_train = enrichment_train.sample(frac=1).reset_index(drop=True)
        
    return enrichment_train


def rand_sampling(data,number_of_samples):
    downsampling_data = data.sample(n=number_of_samples)
    return downsampling_data

def remove_single_label(train_data_path,probs_eval_path,main_label,percentage_to_remove,balance = True):
    train_data = pd.read_csv(train_data_path)
    probs_eval= pd.read_csv(probs_eval_path)
    train_sarcasm = train_data[train_data["label"] == 1]
    probs_eval = probs_eval.drop(columns=['text'])
    # train_sarcasm = train_sarcasm.drop(columns=["dataset","rag_in_domain","empathy_probs","humor_probs","toxcity_probs","irony_probs","politeness_probs","offensive_probs"])

    train_sarcasm.reset_index(inplace = True , drop = True)
    probs_eval.reset_index(inplace = True , drop = True)

    # add the model probability to the roastMe dataset
    positive_data_prob =pd.concat([train_sarcasm, probs_eval], axis=1, join="inner")
    #remove rows that the main model prob is lower than 0.75 
    positive_data_prob.reset_index(inplace = True , drop = True)
    positive_highest_prob = positive_data_prob[(positive_data_prob[main_label]>0.75)]

    print("size to remove from train: ", percentage_to_remove*len(train_data), ", maximum len to remove: ",len(positive_highest_prob))
    pos_samples_number_to_remove = min(percentage_to_remove*len(train_data), len(positive_highest_prob))
    data_to_remove = rand_sampling(positive_highest_prob,int(pos_samples_number_to_remove))

    unenrichment_data = train_data[~train_data['text'].isin(data_to_remove['text'])]
    unenrichment_data.reset_index(inplace = True , drop = True)
    
    if balance:
        # remove negeative sarcasm samples
        negative_data_downsampling = rand_sampling(unenrichment_data[unenrichment_data['label'] == 0],len(unenrichment_data[unenrichment_data['label'] == 0]) -len(unenrichment_data[unenrichment_data['label'] == 1]))
        unenrichment_data = unenrichment_data[~unenrichment_data['text'].isin(negative_data_downsampling['text'])]
        
    unenrichment_data.reset_index(inplace = True , drop = True)

    return unenrichment_data


def remove_two_labels(train_data_path,probs_eval_path,remove_label1,remove_label2,percentage_to_remove,balance = True):
    train_data = pd.read_csv(train_data_path)
    probs_eval= pd.read_csv(probs_eval_path)
    train_sarcasm = train_data[train_data["label"] == 1]
    probs_eval = probs_eval.drop(columns=['text'])
    # train_sarcasm = train_sarcasm.drop(columns=["dataset","rag_in_domain","empathy_probs","humor_probs","toxcity_probs","irony_probs","politeness_probs","offensive_probs"])
    train_sarcasm.reset_index(inplace = True , drop = True)
    probs_eval.reset_index(inplace = True , drop = True)

    # add the model probability to the roastMe dataset
    positive_data_prob =pd.concat([train_sarcasm, probs_eval], axis=1, join="inner")
    positive_data_prob.reset_index(inplace = True , drop = True)

        #remove rows that the main model prob is lower than 0.75 
    remove_positive_prob = positive_data_prob[(positive_data_prob[remove_label1]>0.75) & (positive_data_prob[remove_label2]>0.75)]
    
    print("size to remove from train: ", percentage_to_remove*len(train_data), ", maximum len to remove: ",len(remove_positive_prob))
    pos_samples_number_to_remove = min(percentage_to_remove*len(train_data), len(remove_positive_prob))
    
    downsampled_pos_data_to_remove = rand_sampling(remove_positive_prob,int(pos_samples_number_to_remove))
    unenrichment_data = train_data[~train_data['text'].isin(downsampled_pos_data_to_remove['text'])]
    unenrichment_data.reset_index(inplace = True , drop = True)
    
    if balance:
        # remove negeative sarcasm samples
        negative_data_downsampling = rand_sampling(unenrichment_data[unenrichment_data['label'] == 0],len(unenrichment_data[unenrichment_data['label'] == 0]) -len(unenrichment_data[unenrichment_data['label'] == 1]))
        unenrichment_data = unenrichment_data[~unenrichment_data['text'].isin(negative_data_downsampling['text'])]
    unenrichment_data.reset_index(inplace = True , drop = True)
        
    return unenrichment_data

def remove_randomly(train_data_path,percentage_to_remove,balance = True):
    train_data = pd.read_csv(train_data_path)
    train_sarcasm = train_data[train_data["label"] == 1]
    train_sarcasm.reset_index(inplace = True , drop = True)
    
    print("size to remove from train: ", percentage_to_remove*len(train_data), ", maximum len to remove: ",len(train_sarcasm))
    pos_samples_number_to_remove = min(percentage_to_remove*len(train_data), len(train_sarcasm))    
    pos_data_to_remove = rand_sampling(train_sarcasm,int(pos_samples_number_to_remove))
    unenrichment_data = train_data[~train_data['text'].isin(pos_data_to_remove['text'])]
    unenrichment_data.reset_index(inplace = True , drop = True)
    
    if balance:
        # remove negeative sarcasm samples
        negative_data_downsampling = rand_sampling(unenrichment_data[unenrichment_data['label'] == 0],len(unenrichment_data[unenrichment_data['label'] == 0]) -len(unenrichment_data[unenrichment_data['label'] == 1]))
        unenrichment_data = unenrichment_data[~unenrichment_data['text'].isin(negative_data_downsampling['text'])]
        unenrichment_data.reset_index(inplace = True , drop = True)   
    return unenrichment_data

def remove_samples_by_labels(train_data_path,probs_eval_path,percentage_to_remove,balance = True):
    train_data = pd.read_csv(train_data_path)
    probs_eval= pd.read_csv(probs_eval_path)
    train_sarcasm = train_data[train_data["label"] == 1]
    probs_eval = probs_eval.drop(columns=['text'])
    train_sarcasm.reset_index(inplace = True , drop = True)
    # add the model probability to the roastMe dataset
    positive_data_prob =pd.concat([train_sarcasm, probs_eval], axis=1, join="inner")
        #remove rows that the main model prob is lower than 0.75 
    remove_positive_prob = positive_data_prob[(positive_data_prob["irony_probs"]>0.75) & (positive_data_prob["toxcity_probs"]>0.75)  & 
                                              (positive_data_prob["humor_probs"]>0.75)]
    
    print("size to remove from train: ", percentage_to_remove*len(train_sarcasm), ", maximum len to remove: ",len(remove_positive_prob))
    pos_samples_number_to_remove = min(percentage_to_remove*len(train_sarcasm), len(remove_positive_prob))
    
    downsampled_pos_data_to_remove = rand_sampling(remove_positive_prob,int(pos_samples_number_to_remove))
    unenrichment_data = train_data[~train_data['text'].isin(downsampled_pos_data_to_remove['text'])]
    unenrichment_data.reset_index(inplace = True , drop = True)
    
    if balance:
        # remove negeative sarcasm samples
        negative_data_downsampling = rand_sampling(unenrichment_data[unenrichment_data['label'] == 0],len(unenrichment_data[unenrichment_data['label'] == 0]) -len(unenrichment_data[unenrichment_data['label'] == 1]))
        unenrichment_data = unenrichment_data[~unenrichment_data['text'].isin(negative_data_downsampling['text'])]
    unenrichment_data.reset_index(inplace = True , drop = True)
        
    return unenrichment_data

def save_results(downsample_type,options,results_file, results_dir,iter_size):
    results_path = "C:/Users/chanm/Desktop/studies/masters/Thesis/Thesis/BERTmodels/" + results_dir #scores_results_updownsampling/SARC/enrichment_imbalanced/"
    # options = ['18']
    # results_file = ["random", "sarc"]
    # downsample_type = ["random", "sarc"]
    for i in range(len(results_file)):
        for j in range(len(options)):
            dfs = []
            for num in range(iter_size):
                results_dir = results_path + results_file[i] + str(num)+".csv"
                df = pd.read_csv(results_dir)
                dfs.append(df)     
            combined_df = pd.concat(dfs)
            
            mean_f1 = combined_df.groupby('test data').agg(
                    mean_f1=('mean f1 binary', 'mean'),
                    std_f1=('mean f1 binary', 'std'),
                    mean_recall=('mean recall', 'mean'),
                    std_recall=('mean recall', 'std'),
                    mean_precision=('mean precision', 'mean'),
                    std_precision=('mean precision', 'std')
                    ).reset_index()
            mean_f1["downsample type"] = downsample_type[i]
            if j == 0:
                merged_df = mean_f1
            else:
                merged_df = pd.merge(merged_df, mean_f1, on=['downsample type', 'test data'], how='inner')
        if i == 0:
            merged_df_balanced = merged_df
        else:
            merged_df_balanced = pd.concat([merged_df_balanced, merged_df], ignore_index=True)
    merged_df_balanced
    merged_df_balanced.to_csv(f"{results_path}/mean.csv", index=False, header=True)

def f1_diff_kl_corr(kl,differences):
    # Filter the KL divergence DataFrame for "irony" and "toxicity" sublabels only
    filtered_kl = kl[kl["sublabel_index"].isin(["irony", "toxicity"])]
    filtered_kl = filtered_kl[filtered_kl["cross_domain"].isin(differences.columns)]

    downsampling_indices = {"random": 2, "irony": 0, "toxicity": 3, "irony and toxicity": 1}

    # Add the F1 score differences to the filtered DataFrame
    filtered_kl["f1_diff"] = filtered_kl.apply(
        lambda row: differences.loc[downsampling_indices[row["sublabel_index"]], row["cross_domain"]],
        axis=1
    )

    # Create the plot
    plt.figure(figsize=(10, 6))
    # Iterate over the downsampling types (rows in `differences`)
    downsampling_types = ["random", "irony", "toxicity", "irony and toxicity"]
    for i, ds_type in enumerate(downsampling_types):
        # Get the corresponding F1 differences for the downsampling type
        filtered_kl["f1_diff"] = filtered_kl.apply(
            lambda row: differences.loc[i, row["cross_domain"]],
            axis=1
        )
        
        added_datasets = set()  # To track which datasets have been labeled
        # Scatter plot for each sublabel type (irony/toxicity) with the current downsampling type
        for sublabel in ["irony", "toxicity"]:
            
            subset = filtered_kl[filtered_kl["sublabel_index"] == sublabel]
            
            print(sublabel,", ",ds_type," : ",pearsonr(subset["f1_diff"], subset["kl_divergence"]))
            plt.scatter(
                subset["kl_divergence"],
                subset["f1_diff"],
                label=f"{ds_type} - {sublabel}"
            )

            for idx, row in subset.iterrows():
                if row["cross_domain"] not in added_datasets:
                    plt.text(
                        row["kl_divergence"], 
                        row["f1_diff"], 
                        row["cross_domain"], 
                        fontsize=9,  # Adjust the font size as needed
                        ha='right',  # Align text to the right of the point
                        va='bottom',  # Place text slightly below the point
                        color='black'  # Set text color
                    )
                    added_datasets.add(row["cross_domain"])  # Mark this dataset as labeled


    # Add plot details
    plt.title("Correlation Between KL Divergence and F1 Score Differences")
    plt.xlabel("KL Divergence")
    plt.ylabel("F1 Score Difference")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))  # Place legend outside the plot
    plt.grid(True)
    plt.tight_layout()

def f1_diff(table,thresholds):
    # table = table.drop(columns=["SARC1"])
    data_names = table.columns
    
    for i, data in enumerate(data_names):
        table[data] = table[data] - thresholds[i]
    return table
    
    
def plot_f1_downsample(table,thresholds,title,score_type,show_values=True):

    # Common x labels
    # x_labels = ["irony negative","toxicity negative","toxicity and irony negative","random negative"]
    x_labels = ["irony","toxicity","random","toxicity and irony"]
    table = table[table['test data'] != "SARC1"]
    datasets = table['test data'].unique()
    datas = ["MUStARD","Ptacek", "Riloff","SARC","SPIRS","SemEval2022","iSarcasmEval","MSD"]
    # datas = [ "Ptacek", "Riloff"]
    # Initialize lists to store y values and error values for each dataset
    y_values = []
    errors = []
    for dataset in datasets:
        y_data = []
        error_data = []
        
        # Filter data for each downsample type within the dataset
        subset = table[table['test data'] == dataset]
        for dtype in x_labels:
            row = subset[subset['downsample type'] == dtype]
            y_data.append(row[f'mean_{score_type}'].values[0])
            error_data.append(row[f'std_{score_type}'].values[0])
        
        y_values.append(y_data)
        errors.append(error_data)

    # Parameters for the grouped bar plot
    n_labels = len(x_labels)
    bar_width = 0.15  # Width of each bar
    x = np.arange(len(datasets))  # X positions for the groups of bars

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each label as a separate set of bars within each group
    for j in range(n_labels):
        # Calculate the position of each bar
        bar_positions = x + j * bar_width
        bars = ax.bar(bar_positions, [y[j] * 100 for y in y_values], 
                    yerr=[e[j] * 100 for e in errors], 
                    capsize=5, width=bar_width, label=x_labels[j], alpha=0.7, edgecolor='black')

        if show_values:
            # Add the y value as a percentage at the bottom of each bar (on the bar itself)
            for i, bar in enumerate(bars):
                y_val = bar.get_height()
                error_val = errors[i][j] * 100
                
                # Add y value at the bottom of the bar (slightly above the base)
                ax.text(bar.get_x() + bar.get_width() / 2, 35, f'{y_val:.1f}%', 
                        ha='center', va='bottom', fontsize=6, color='black', rotation=0)

                # Add error value as a percentage at the top of each error bar
                ax.text(bar.get_x() + bar.get_width() / 2, y_val + error_val + 0.5, 
                        f'±{error_val:.1f}%', ha='center', va='bottom', fontsize=6, color='black', rotation=90)

    # Customize the plot
    ax.set_xlabel('Test DataSet',fontsize=16)
    ax.set_ylabel('Mean F1 Score (%)',fontsize=16)

    # ax.set_title(f'F1 score Mean and STD {title}')
    ax.set_xticks(x + (n_labels - 1) * bar_width / 2)
    ax.set_xticklabels([f'{data}' for data in datas])
    
        # Add a horizontal line for each threshold, limited to the corresponding dataset
    for i, threshold in enumerate(thresholds):
        # Calculate the xmin and xmax for the threshold line
        xmin = i / len(y_values)
        xmax = (i + 1) / len(y_values)
        ax.axhline(y=threshold * 100, color='red', linestyle='--', linewidth=1, alpha=0.7,
                xmin=xmin, xmax=xmax)
                # , label=f'original F1 {datas[i]} = {threshold * 100:.1f}%')
        
    ax.legend(fontsize=14,title='Enrichment type', title_fontsize=14)  # Add a legend to differentiate the labels

        
    # Adjust the y-axis limits to ensure the text fits within the plot
    ax.set_ylim(0, max(max(y) + max(e) for y, e in zip(y_values, errors)) * 100 + 10)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    # Show the plot
    plt.tight_layout()
    plt.show()
    
from scipy.special import rel_entr

def kl_div_by_in_domain(in_domain_data, data_prob):
    kl_results = []
    labels = ['irony','toxicity']
    for cross_domain_data in data_prob.columns:
        if cross_domain_data == "txc" or cross_domain_data == "irn" or cross_domain_data == "txc_irn" or cross_domain_data == "random":
            continue
        for idx ,label in enumerate(labels): 
            kl_value = np.sum(rel_entr(np.array(data_prob[in_domain_data][idx]),
                                       np.array(data_prob[cross_domain_data][idx])))
            kl_results.append({
                'cross_domain': cross_domain_data,
                'sublabel_index': label,
                'kl_divergence': kl_value
            })
    
    # Convert results to a DataFrame
    kl_df = pd.DataFrame(kl_results)
    return kl_df

def sublabels_riloff_prob_enr():
    
    MUSTARD_irony_p = [0.55,0,0.01,0.44]
    MUSTARD_toxicity_p = [0.73,0,0,0.27]
    
    SARC_irony_p = [0.56,0,0,0.43]
    SARC_toxicity_p = [0.73,0,0,0.27]
    
    SPIRS_irony_p = [0.95,0,0,0.05]
    SPIRS_toxicity_p = [0.72,0,0,0.28]
    
    iSarcasmEval_irony_p = [0.82,0,0,0.18]
    iSarcasmEval_toxicity_p = [0.83,0,0,0.17]
    
    pmsd_irony_p = [0.84,0,0,0.16]
    pmsd_toxicity_p = [0.68,0,0,0.32]
    
    SemEval2022_irony_p = [0.85,0,0,0.15]
    SemEval2022_toxicity_p = [0.69,0,0,0.31]

    sublabels_prob = pd.DataFrame({"txc": [[0.18,0,0,0.81],[0.66,0,0,0.34]],
                                   "irn": [[0.56,0,0,0.44],[0.37,0,0,0.62]],
                                   "txc_irn": [[0.56,0,0,0.44],[0.66,0,0,0.34]],
                                   "random": [[0.13,0,0,0.87],[0.31,0,0,0.69]],
                                    "SemEval2022" : [SemEval2022_irony_p,SemEval2022_toxicity_p],
                                    "multimodal_sarcasm_detection":[pmsd_irony_p,pmsd_toxicity_p],
                                    "iSarcasmEval":[iSarcasmEval_irony_p,iSarcasmEval_toxicity_p],
                                    "SPIRS":[SPIRS_irony_p,SPIRS_toxicity_p],
                                    "SARC":[SARC_irony_p,SARC_toxicity_p],
                                    "MUStARD":[MUSTARD_irony_p,MUSTARD_toxicity_p]})

    return sublabels_prob
    

def sublabels_prob():
    MUSTARD_humor_p = [0.11,0,0,0.89]
    MUSTARD_irony_p = [0.55,0,0.01,0.44]
    MUSTARD_politeness_p = [0.8,0.02,0.02,0.16]
    MUSTARD_toxicity_p = [0.73,0,0,0.27]
    MUSTARD_empathy_p = [0.75,0.01,0.01,0.23]
    MUSTARD_offensive_p = [0.95,0,0.01,0.04]

    SARC_humor_p = [0.24,0.01,0,0.75]
    SARC_irony_p = [0.56,0,0,0.43]
    SARC_politeness_p = [0.9,0.01,0.02,0.07]
    SARC_toxicity_p = [0.73,0,0,0.27]
    SARC_empathy_p = [0.84,0.01,0.01,0.14]
    SARC_offensive_p = [0.86,0,0,0.13]

    SPIRS_humor_p = [0.07,0,0,0.93]
    SPIRS_irony_p = [0.95,0,0,0.05]
    SPIRS_politeness_p = [0.89,0.01,0.01,0.08]
    SPIRS_toxicity_p = [0.72,0,0,0.28]
    SPIRS_empathy_p = [0.62,0.01,0.01,0.36]
    SPIRS_offensive_p = [0.86,0,0,0.14]

    iSarcasmEval_humor_p = [0.26,0,0,0.74]
    iSarcasmEval_irony_p = [0.82,0,0,0.18]
    iSarcasmEval_politeness_p = [0.83,0.02,0.02,0.14]
    iSarcasmEval_toxicity_p = [0.83,0,0,0.17]
    iSarcasmEval_empathy_p = [0.77,0.01,0.01,0.21]
    iSarcasmEval_offensive_p = [0.95,0,0,0.05]

    pmsd_humor_p = [0.1,0,0,0.9]
    pmsd_irony_p = [0.84,0,0,0.16]
    pmsd_politeness_p = [0.86,0.01,0.02,0.11]
    pmsd_toxicity_p = [0.68,0,0,0.32]
    pmsd_empathy_p = [0.62,0.01,0.01,0.36]
    pmsd_offensive_p = [0.92,0,0,0.08]

    SemEval2022_humor_p = [0.24,0,0,0.76]
    SemEval2022_irony_p = [0.85,0,0,0.15]
    SemEval2022_politeness_p = [0.86,0.01,0.02,0.11]
    SemEval2022_toxicity_p = [0.69,0,0,0.31]
    SemEval2022_empathy_p = [0.77,0.01,0.01,0.21]
    SemEval2022_offensive_p = [0.9,0,0,0.1]

    ptacek_humor_p = [0.02,0,0,0.98]
    ptacek_irony_p = [0.14,0,0,0.86]
    ptacek_politeness_p = [0.94,0.01,0.01,0.04]
    ptacek_toxicity_p = [0.32,0,0,0.68]
    ptacek_empathy_p = [0.75,0.01,0.01,0.23]
    ptacek_offensive_p = [0.9,0,0,0.1]

    riloff_humor_p = [0.02,0,0,0.98]
    riloff_irony_p = [0.11,0,0,0.89]
    riloff_politeness_p = [0.94,0.01,0.01,0.04]
    riloff_toxicity_p = [0.31,0,0,0.69]
    riloff_empathy_p = [0.79,0,0.01,0.2]
    riloff_offensive_p = [0.9,0,0,0.1]

    sublabels_prob = pd.DataFrame({"Riloff": [riloff_humor_p,riloff_irony_p,riloff_politeness_p,riloff_toxicity_p,riloff_empathy_p,riloff_offensive_p],
                                        "Ptacek" : [ptacek_humor_p,ptacek_irony_p,ptacek_politeness_p,ptacek_toxicity_p,ptacek_empathy_p,ptacek_offensive_p],
                                        "SemEval2022" : [SemEval2022_humor_p,SemEval2022_irony_p,SemEval2022_politeness_p,SemEval2022_toxicity_p,SemEval2022_empathy_p,SemEval2022_offensive_p],
                                        "multimodal_sarcasm_detection":[pmsd_humor_p,pmsd_irony_p,pmsd_politeness_p,pmsd_toxicity_p,pmsd_empathy_p,pmsd_offensive_p],
                                        "iSarcasmEval":[iSarcasmEval_humor_p,iSarcasmEval_irony_p,iSarcasmEval_politeness_p,iSarcasmEval_toxicity_p,iSarcasmEval_empathy_p,iSarcasmEval_offensive_p],
                                        "SPIRS":[SPIRS_humor_p,SPIRS_irony_p,SPIRS_politeness_p,SPIRS_toxicity_p,SPIRS_empathy_p,SPIRS_offensive_p],
                                        "SARC":[SARC_humor_p,SARC_irony_p,SARC_politeness_p,SARC_toxicity_p,SARC_empathy_p,SARC_offensive_p],
                                        "MUStARD":[MUSTARD_humor_p,MUSTARD_irony_p,MUSTARD_politeness_p,MUSTARD_toxicity_p,MUSTARD_empathy_p,MUSTARD_offensive_p]})
    
    return sublabels_prob


def corr_p_val_plot(data_prop):
    f1_scores =[]
    sub_labels_kl = []
    sub_labels_kl_irony = []
    sub_labels_kl_toxicity = []
    
    for data_pair in data_prop.columns:
        f1_scores.append(data_prop[data_pair][2])
        sub_labels_kl_irony.append(data_prop[data_pair][0])
        sub_labels_kl_toxicity.append(data_prop[data_pair][1])
        for kl in data_prop[data_pair][0:5]:
            sub_labels_kl.append(kl)
            
    print("irony : " ,pearsonr(f1_scores, sub_labels_kl_irony))
    print("toxicity : " ,pearsonr(f1_scores, sub_labels_kl_toxicity))
    
    colors = ['blue', 'red']

    # Plotting the data
    plt.figure(figsize=(10, 6))
    
    last_row = data_prop.iloc[-1]
    labels = [ 'irony', 'toxicity']
    for idx in range(len(data_prop) - 1):
        x_values = data_prop.iloc[idx]
        y_values = last_row
        plt.scatter(x_values, y_values, color=colors[idx], label=labels[idx])
            

    # Titles and labels
    plt.title("Sublabels' KL Divergence and F1 Score of In and Cross Domain")
    plt.xlabel('KL divergence')
    plt.ylabel('F1 score (the gap between in domain and cross domain)')
    plt.legend()

    plt.show()

def kl_div_f1(train_data_prob,test_data_prob,in_A_A,cross_A_B):
    epsilon = 1e-10
    sublabel_kl = []
    for idx in range(2):
        p = np.array(test_data_prob[idx])
        q = np.array(train_data_prob[idx])
        q = np.where(q == 0, epsilon, q) # avoid division with zero
        sublabel_kl.append(np.sum(rel_entr(p,q)))
    f1_score = f1_score_eval(in_A_A,cross_A_B)
    sublabel_kl.append(f1_score)
    return sublabel_kl

def f1_score_eval(in_A_A,cross_A_B):
    return in_A_A-cross_A_B