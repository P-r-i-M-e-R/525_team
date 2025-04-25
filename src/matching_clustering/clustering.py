import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize



def determine_k(embeddings: np.ndarray) -> int:
    k_min = 4
    clusters = [x for x in range(2, k_min * 9)]
    metrics = []
    for i in clusters:
        metrics.append((KMeans(n_clusters=i).fit(embeddings)).inertia_)
    k = elbow_calculation(k_min, clusters, metrics)
    return k


def elbow_calculation(k_min: int, clusters: list[int], metrics: list[float]) -> int:
    score = []

    for i in tqdm(range(k_min, clusters[-3])):
        y1 = np.array(metrics)[:i + 1]
        y2 = np.array(metrics)[i:]
    
        df1 = pd.DataFrame({'x': clusters[:i + 1], 'y': y1})
        df2 = pd.DataFrame({'x': clusters[i:], 'y': y2})
    
        reg1 = LinearRegression().fit(np.asarray(df1.x).reshape(-1, 1), df1.y)
        reg2 = LinearRegression().fit(np.asarray(df2.x).reshape(-1, 1), df2.y)

        y1_pred = reg1.predict(np.asarray(df1.x).reshape(-1, 1))
        y2_pred = reg2.predict(np.asarray(df2.x).reshape(-1, 1))    
        
        score.append(mean_squared_error(y1, y1_pred) + mean_squared_error(y2, y2_pred))

    return np.argmin(score) + k_min


def extract_top_5_texts_per_cluster(df: pd.DataFrame, emb_norm: np.ndarray) -> pd.DataFrame:

    top5_per_cluster = []
    for lbl in tqdm(sorted(df['label'].unique())):
        idxs = df.index[df['label'] == lbl].tolist()
        centroid = emb_norm[idxs].mean(axis=0, keepdims=True)
        sims = cosine_similarity(emb_norm[idxs], centroid).flatten()
        top5 = np.array(idxs)[np.argsort(sims)[-5:][::-1]]

        for rank, i in enumerate(top5, start=1):
            top5_per_cluster.append({
                'cluster': lbl,
                'rank': rank,
                'article_id': df.at[i, 'idx'],
                'text': df.at[i, 'article']
            })
    
    return pd.DataFrame(top5_per_cluster)


def best_clustering(
        result_texts: list[str], n_clusters: int,
        embeddings_result: list[np.ndarray]
    ) -> list[dict[str, Any]]:
    articles_dct = [
        {'idx': i, 'article': result_texts[i]} for i in range(len(result_texts))
    ]
    labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(embeddings_result)
    for meta, label in zip(articles_dct, labels):
        meta["label"] = int(label)
    return articles_dct


def visualize_clusters(articles_dct: dict[str, Any], embeddings: list[list[int]], perplexity: int = 10) -> pd.DataFrame:
    df = pd.DataFrame(articles_dct)
    emb_norm = normalize(embeddings, norm='l2')
    tsne = TSNE(n_components=2, metric='cosine', random_state=42, perplexity=perplexity, n_iter=1000)
    emb_tsne2 = tsne.fit_transform(emb_norm)

    df['tsne1'], df['tsne2'] = emb_tsne2[:, 0], emb_tsne2[:, 1]
    fig, axes = plt.subplots(1, 1, figsize=(14, 6), dpi=500)

    for lbl in sorted(df['label'].unique()):
        subset = df[df['label'] == lbl]
        axes.scatter(subset['tsne1'], subset['tsne2'], label=f'Cluster {lbl}', alpha=0.7)

    axes.set_title('t-SNE кластеров')
    axes.set_xlabel('t-SNE')
    axes.set_ylabel('t-SNE')
    axes.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.tight_layout()
    plt.show()
    return df

