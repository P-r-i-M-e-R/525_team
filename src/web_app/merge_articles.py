from typing import Any

import numpy as np
import pandas as pd
from PIL.Image import Image
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
from tqdm import tqdm

BERT_MODEL = 'ai-forever/sbert_large_nlu_ru'
PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#3ae6ca', '#aedbd8', '#f7b6d2', '#c49c94']

MIN_ARTICLE_LENGTH = 3
SIMILARITY_THRESHOLD = 0.78  


def calculate_normalized_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Convert bounding boxes to normalized coordinates and center points."""
    # Extract coordinates
    coords = np.array(df['rects'].tolist())

    # Calculate page dimensions
    x_min, y_min = coords[:, 0, :].min(axis=0)
    x_max, y_max = coords[:, 1, :].max(axis=0)
    width = x_max - x_min
    height = y_max - y_min

    # Normalize coordinates
    normalized_coords = []
    for rect in df['rects']:
        norm_rect = [
            [(rect[0][0] - x_min) / width, (rect[0][1] - y_min) / height],
            [(rect[1][0] - x_min) / width, (rect[1][1] - y_min) / height]
        ]
        normalized_coords.append(norm_rect)

    df['norm_rects'] = normalized_coords

    # Calculate center points
    centers = []
    for rect in normalized_coords:
        center_x = (rect[0][0] + rect[1][0]) / 2
        center_y = (rect[0][1] + rect[1][1]) / 2
        centers.append([center_x, center_y])

    df['center'] = centers
    return df


def generate_bert_embeddings(texts: list[str]) -> np.ndarray:
    """Generate BERT embeddings for texts."""
    model = SentenceTransformer(BERT_MODEL)
    return model.encode(texts, convert_to_tensor=False)

def build_hybrid_similarity_matrix(df: pd.DataFrame, embeddings: np.ndarray, w_content=0.7, w_spatial=0.3) -> np.ndarray:
    """Build similarity matrix combining content and spatial proximity."""
    # Content similarity
    content_sim = cosine_similarity(embeddings)

    # Spatial similarity (inverse of distance)
    centers = np.array(df['center'].tolist())
    spatial_dist = cdist(centers, centers, 'euclidean')
    spatial_sim = 1 / (1 + spatial_dist)  # Convert distance to similarity

    # Combine with preference for content
    hybrid_sim = w_content * content_sim + w_spatial * spatial_sim

    # Set diagonal to 0 to avoid self-matches
    np.fill_diagonal(hybrid_sim, 0)
    return hybrid_sim


def cluster_blocks(df: pd.DataFrame, similarity_matrix: np.ndarray) -> list[list[int]]:
    """Cluster blocks into articles using agglomerative clustering."""
    clusters = []
    unassigned = set(range(len(df)))

    while unassigned:
        # Start with the highest similarity pair
        idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        max_sim = similarity_matrix[idx]

        if max_sim < SIMILARITY_THRESHOLD:
            # Remaining items go to their own clusters
            clusters.extend([[i] for i in unassigned])
            break

        # Find all blocks that should be merged
        new_cluster = set(idx)
        added = True

        while added:
            added = False
            for i in list(new_cluster):
                # Find neighbors above threshold
                neighbors = set(np.where(similarity_matrix[i] >= SIMILARITY_THRESHOLD)[0])
                new_neighbors = neighbors - new_cluster

                if new_neighbors:
                    new_cluster.update(new_neighbors)
                    added = True

        # Add to clusters
        clusters.append(list(new_cluster))
        unassigned -= new_cluster

        # Prevent these from being merged again
        for i in new_cluster:
            similarity_matrix[i, :] = 0
            similarity_matrix[:, i] = 0

    # Filter clusters to only include article blocks (titles will be assigned later)
    article_clusters = []
    for cluster in clusters:
        article_blocks = [i for i in cluster if df.iloc[i]['entity_types'] == 'article']
        if article_blocks:
            article_clusters.append(article_blocks)

    return article_clusters


def assign_titles_to_articles(df: pd.DataFrame, article_clusters: list[list[int]]) -> list[dict]:
    """Assign title blocks to each article cluster."""
    results = []
    title_indices = df[df['entity_types'] == 'title'].index.tolist()
    assigned_titles = set()

    for cluster in article_clusters:
        # Get article bounds
        cluster_rects = df.iloc[cluster]['rects'].tolist()
        min_x = min(r[0][0] for r in cluster_rects)
        max_x = max(r[1][0] for r in cluster_rects)
        min_y = min(r[0][1] for r in cluster_rects)
        max_y = max(r[1][1] for r in cluster_rects)

        # Find titles above the article and within x-bounds
        candidate_titles = []
        for title_idx in title_indices:
            if title_idx in assigned_titles:
                continue

            title_rect = df.loc[title_idx]['rects']

            # Check if title is above article and horizontally aligned
            if (title_rect[0][1] < max_y and
                    title_rect[0][0] >= min_x - (max_x - min_x) * 0.1 and
                    title_rect[1][0] <= max_x + (max_x - min_x) * 0.1):
                candidate_titles.append(title_idx)

        # If multiple candidates, choose the closest one
        if candidate_titles:
            cluster_center = np.mean([df.iloc[i]['center'] for i in cluster], axis=0)
            title_distances = []

            for title_idx in candidate_titles:
                title_center = df.loc[title_idx]['center']
                distance = np.linalg.norm(np.array(cluster_center) - np.array(title_center))
                title_distances.append(distance)

            best_title = candidate_titles[np.argmin(title_distances)]
        else:
            best_title = None

        # Prepare result
        article_text = " ".join(df.iloc[i]['text'] for i in cluster)
        title_text = df.loc[best_title]['text'] if best_title is not None else ""

        assigned_titles.add(best_title)

        results.append({
            'article_blocks': cluster,
            'title_block': best_title,
            'article_text': article_text,
            'title_text': title_text,
            'rects': [df.iloc[i]['rects'] for i in cluster],
            'title_rect': df.loc[best_title]['rects'] if best_title else None
        })

    return results

def visualize_page_layout(df: pd.DataFrame, clusters: list[dict[str, Any]], page_num: int, image_background: Image | None = None):
    """Visualize the page layout with article groupings, image background, and reading order."""
    fig = plt.figure(figsize=(12, 15))
    plt.title(f"Page {page_num} Article Grouping")

    # Add page image background if available
    if image_background is not None:
        plt.imshow(image_background, alpha=0.5)

    # Draw all rectangles
    for i, row in df.iterrows():
        rect = row['rects']
        color = '#cccccc'  # Default color for ungrouped
        label = None

        # Check if this is a title
        if row['entity_types'] == 'title':
            for cluster_idx, cluster in enumerate(clusters):
                if i == cluster["title_block"]:
                    color = PLOT_COLORS[cluster_idx % len(PLOT_COLORS)]
                    label = f'Article {cluster_idx + 1}'
                    break
        else:
            # Find if this block is in any cluster
            for cluster_idx, cluster in enumerate(clusters):
                if i in cluster["article_blocks"]:
                    color = PLOT_COLORS[cluster_idx % len(PLOT_COLORS)]
                    label = f'Article {cluster_idx + 1}'
                    break

        # Draw rectangle
        width = rect[1][0] - rect[0][0]
        height = rect[1][1] - rect[0][1]
        patch = plt.Rectangle(
            (rect[0][0], rect[0][1]), width, height,
            fill=False, edgecolor=color, linewidth=2, label=label
        )
        plt.gca().add_patch(patch)

        # Add text label
        plt.text(
            rect[0][0], rect[0][1], f"{i}: {row['entity_types']}",
            fontsize=8, ha='left', va='bottom', color=color,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

    # Draw reading order arrows for each cluster
    for cluster_idx, cluster in enumerate(clusters):
        if len(cluster) < 2:
            continue

        # Get centers in reading order (top-to-bottom, left-to-right)
        centers = []
        title_block = [cluster["title_block"]] if cluster["title_block"] is not None else []
        for i in title_block + cluster["article_blocks"]:
            is_title = int(df.loc[i]['entity_types'] == 'title')
            rect = df.loc[i]['rects']
            center_x = (rect[0][0] + rect[1][0]) / 2
            center_y = (rect[0][1] + rect[1][1]) / 2
            centers.append((is_title, center_x, center_y, i))

        # Sort by y (top to bottom) then by x (left to right)
        centers.sort(key=lambda c: (-c[0], c[1], -c[2]))

        # Draw arrows between consecutive blocks
        color = PLOT_COLORS[cluster_idx % len(PLOT_COLORS)]
        for j in range(len(centers) - 1):
            _, x1, y1, i1 = centers[j]
            _, x2, y2, i2 = centers[j + 1]

            plt.annotate("",
                         xy=(x2, y2), xycoords='data',
                         xytext=(x1, y1), textcoords='data',
                         arrowprops=dict(arrowstyle="->", color=color,
                                         connectionstyle="arc3,rad=-0.2",
                                         linestyle='dashed', alpha=0.7, linewidth=2),
                         zorder=10
                         )

    # Set limits and legend
    all_rects = np.array(df['rects'].tolist())
    x_min, y_min = all_rects[:, 0, :].min(axis=0)
    x_max, y_max = all_rects[:, 1, :].max(axis=0)

    plt.xlim(x_min - 50, x_max + 50)
    plt.ylim(y_max + 50, y_min - 50)  # Invert y-axis
    plt.gca().set_aspect('equal')

    # Create custom legend entries
    legend_elements = []
    for cluster_idx in range(len(clusters)):
        legend_elements.append(
            plt.Line2D([0], [0], color=PLOT_COLORS[cluster_idx % len(PLOT_COLORS)],
                       lw=2, label=f'Article {cluster_idx + 1}')
        )

    plt.legend(handles=legend_elements)
    plt.tight_layout()
    return fig


def convert_class(orig_type: str) -> str:
    match orig_type:
        case "Text":
            return "article"
        case "Section-header":
            return "title"
        case _:
            return orig_type


def main_merge(page_df: pd.DataFrame, image: Image) -> tuple[str, Any]:
    page_df["entity_types"] = page_df["class"].apply(convert_class)
    page_df['rects'] = page_df.apply(lambda row: [[row['x1'], row['y1']], [row['x2'], row['y2']]], axis=1)
    page_df['text'] = page_df['corrected_text'].str.strip()

    page_df = calculate_normalized_coords(page_df)

    # Generate embeddings
    embeddings = generate_bert_embeddings(page_df['text'].tolist())

    # Build similarity matrix combining content and spatial proximity
    similarity_matrix = build_hybrid_similarity_matrix(page_df, embeddings)

    # Cluster blocks into articles
    article_clusters = cluster_blocks(page_df, similarity_matrix)

    # Find titles for each article
    page_results = assign_titles_to_articles(page_df, article_clusters)
    
    fig = visualize_page_layout(page_df, page_results, 1, image)
    
    result = pd.DataFrame(page_results)
    
    articles = []
    for _, entity in tqdm(result.iterrows()):
        title = "# " + " ".join(entity["title_text"].strip().split("\n"))
        content = entity["article_text"]
        articles.append(title + "\n\n" + content + "\n")

    return "\n\n".join(articles), fig
    
    
    