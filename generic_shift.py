"""
Generic (non-affective) domain-shift metrics, computed the same way the
affective KL divergence is computed in sublabels_analysis.py, so results can
be merged into the same train/test-pair analysis table.

These exist to test whether affective KL divergence retains explanatory
power for cross-domain F1 degradation once generic lexical/topical/embedding
shift is controlled for (see partial_corr_analysis/regression_analysis in
sublabels_analysis.py).
"""
import numpy as np
import pandas as pd
import torch
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import jensenshannon
from transformers import AutoTokenizer, AutoModel

EMBEDDING_MODEL_NAME = "bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def lexical_js_divergence(texts_a, texts_b, ngram_range=(1, 2)):
    """
    Jensen-Shannon divergence between two datasets' unigram+bigram term
    distributions. Both texts are vectorized against their shared (union)
    vocabulary so the resulting count vectors are directly comparable.
    """
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    vectorizer.fit(pd.concat([texts_a, texts_b]))

    counts_a = np.asarray(vectorizer.transform(texts_a).sum(axis=0)).ravel()
    counts_b = np.asarray(vectorizer.transform(texts_b).sum(axis=0)).ravel()

    dist_a = counts_a / counts_a.sum()
    dist_b = counts_b / counts_b.sum()

    return jensenshannon(dist_a, dist_b) ** 2  # squared: divergence, not distance


def get_bert_embeddings(text_list, tokenizer, model, batch_size=32, max_length=128):
    """
    Mean-pooled last_hidden_state embedding per text (excluding [CLS]/[SEP]),
    following the pooling convention already used in pca.ipynb, but batched
    and run on DEVICE for speed.
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(text_list), batch_size):
            batch = list(text_list[start:start + batch_size])
            inputs = tokenizer(
                batch, return_tensors="pt", truncation=True,
                padding=True, max_length=max_length,
            ).to(DEVICE)
            outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            summed = (outputs.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            batch_embeddings = (summed / counts).cpu().numpy()
            embeddings.append(batch_embeddings)
    return np.concatenate(embeddings, axis=0)


def frechet_distance(embeddings_a, embeddings_b):
    """
    Frechet-style distance between two embedding distributions, assuming each
    is approximately Gaussian: ||mu_a - mu_b||^2 + tr(Sa + Sb - 2*sqrt(Sa*Sb)).
    Same construction as FID, applied to domain embedding shift instead of
    generated-image quality.
    """
    mu_a, mu_b = embeddings_a.mean(axis=0), embeddings_b.mean(axis=0)
    sigma_a = np.cov(embeddings_a, rowvar=False)
    sigma_b = np.cov(embeddings_b, rowvar=False)

    diff = mu_a - mu_b
    covmean = _sqrtm_psd(sigma_a @ sigma_b)
    return float(diff @ diff + np.trace(sigma_a + sigma_b - 2 * covmean))


def _sqrtm_psd(mat):
    """Matrix square root for a PSD matrix via eigendecomposition (real, symmetric-safe)."""
    mat = (mat + mat.T) / 2
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, 0, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def compute_generic_shift_table(datasets, embedding_dim_reduction=50, random_state=42):
    """
    Compute lexical JS divergence and embedding Frechet distance for every
    ordered pair of datasets.

    `datasets` is a dict of {dataset_name: dataframe}, each with a 'text'
    column (as loaded from train_data_filtered.csv / test_data_filtered.csv).

    Returns a DataFrame with columns:
        train_dataset, test_dataset, lexical_js_divergence, embedding_distance
    matching the (train, test) pair keying used by the affective KL and F1
    gap tables, so it can be merged with them directly.
    """
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(DEVICE)

    raw_embeddings = {
        name: get_bert_embeddings(df["text"].astype(str).tolist(), tokenizer, model)
        for name, df in datasets.items()
    }

    if embedding_dim_reduction:
        from sklearn.decomposition import PCA
        stacked = np.concatenate(list(raw_embeddings.values()), axis=0)
        n_components = min(embedding_dim_reduction, stacked.shape[0], stacked.shape[1])
        pca = PCA(n_components=n_components, random_state=random_state).fit(stacked)
        reduced_embeddings = {name: pca.transform(emb) for name, emb in raw_embeddings.items()}
    else:
        reduced_embeddings = raw_embeddings

    rows = []
    for name_a, name_b in combinations(datasets, 2):
        texts_a = datasets[name_a]["text"].astype(str)
        texts_b = datasets[name_b]["text"].astype(str)

        js_div = lexical_js_divergence(texts_a, texts_b)
        embed_dist = frechet_distance(reduced_embeddings[name_a], reduced_embeddings[name_b])

        for train_name, test_name in [(name_a, name_b), (name_b, name_a)]:
            rows.append({
                "train_dataset": train_name,
                "test_dataset": test_name,
                "lexical_js_divergence": js_div,
                "embedding_distance": embed_dist,
            })

    for name in datasets:
        rows.append({
            "train_dataset": name,
            "test_dataset": name,
            "lexical_js_divergence": 0.0,
            "embedding_distance": 0.0,
        })

    return pd.DataFrame(rows)
