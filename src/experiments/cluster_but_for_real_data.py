from functools import cache

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
from ast import literal_eval
from typing import List, Dict, Tuple, Any
import warnings

from metrics.calculate_wer_cer import ALERT_TEXT
from src.config import PROJECT_ROOT

# =============== CONSTANTS TO CONFIGURE ===============
# Data processing
MIN_ARTICLE_LENGTH = 3  # Minimum characters to consider as valid text
SIMILARITY_THRESHOLD = 0.78  # Minimum similarity score to merge blocks
# SIMILARITY_THRESHOLD = 0.32

# Visualization
PLOT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#3ae6ca', '#aedbd8', '#f7b6d2', '#c49c94']
SHOW_GRAPH = False  # Whether to display the connection graph
SHOW_PAGE_LAYOUT = False  # Whether to display page layout with groupings

# Embedding configuration
USE_BERT = True  # Whether to use BERT embeddings (False = use TF-IDF)
BERT_MODEL = 'ai-forever/sbert_large_nlu_ru'  # BERT model to use


@cache
def get_model():
    return SentenceTransformer(BERT_MODEL)

# =============== MAIN SCRIPT ===============
def merge_vertical_columns(df: pd.DataFrame, y_threshold: int = 30) -> pd.DataFrame:
    """
    Merge bboxes that are in the same column (same x bounds) and vertically close.

    Args:
        df: DataFrame containing 'rects' column with bbox coordinates
        y_threshold: Maximum normalized vertical gap between boxes to merge (0-1)

    Returns:
        DataFrame with merged boxes
    """
    # Calculate normalized coordinates if not already present
    if 'norm_rects' not in df.columns:
        df = calculate_normalized_coords(df)

    # Group by page first
    merged_data = []
    for page_num, page_df in df.groupby('Page'):
        # Create a list of all boxes with their original indices
        boxes = [(i, rect) for i, rect in enumerate(page_df['rects'])]

        # Sort boxes by x1 (left) position
        boxes.sort(key=lambda x: (x[1][0][0], x[1][0][1]))

        merged_indices = set()
        new_boxes = []

        # Group boxes into columns
        columns = []
        current_column = []

        blacklist = set()

        for i, box in boxes:
            box_width = box[1][0] - box[0][0]
            box_height = box[1][1] - box[0][1]
            if box_width / box_height > 10:
                blacklist.add(i)
                continue
            if not current_column:
                current_column.append((i, box))
            else:
                # Check if box belongs to current column (similar x bounds)
                last_box = current_column[-1][1]
                x_overlap = min(box[1][0], last_box[1][0]) - max(box[0][0], last_box[0][0])

                curr_width = (box[1][0] - box[0][0])
                prev_width = (last_box[1][0] - last_box[0][0])

                width_similarity = abs(curr_width - prev_width) / max(curr_width, prev_width)

                if x_overlap > 0 and width_similarity < 0.2:  # 20% width difference allowed
                    current_column.append((i, box))
                else:
                    columns.append(current_column)
                    current_column = [(i, box)]

        if current_column:
            columns.append(current_column)

        # Process each column
        for column in columns:
            # Sort column by y position (top to bottom)
            column.sort(key=lambda x: x[1][0][1])

            i = 0
            while i < len(column):
                current_idx, current_box = column[i]
                if current_idx in merged_indices:
                    i += 1
                    continue

                merged_box = current_box
                text_parts = [page_df.iloc[current_idx]['text']]
                entity_types = [page_df.iloc[current_idx]['entity_types']]
                merged_indices.add(current_idx)

                # Look ahead for boxes to merge
                j = i + 1
                while j < len(column):
                    next_idx, next_box = column[j]
                    y_gap = next_box[0][1] - merged_box[1][1]  # Distance between bottom of current and top of next
                    # norm_y_gap = y_gap / (page_df['rects'].apply(lambda x: x[1][1]).max() -
                    #                       page_df['rects'].apply(lambda x: x[0][1]).min())

                    if y_gap < y_threshold:
                        # Merge boxes
                        merged_box = [
                            [min(merged_box[0][0], next_box[0][0]), min(merged_box[0][1], next_box[0][1])],
                            [max(merged_box[1][0], next_box[1][0]), max(merged_box[1][1], next_box[1][1])]
                        ]
                        text_parts.append(page_df.iloc[next_idx]['text'])
                        entity_types.append(page_df.iloc[next_idx]['entity_types'])
                        merged_indices.add(next_idx)
                        j += 1
                    else:
                        break

                # Create new merged row
                new_row = page_df.iloc[current_idx].copy()
                new_row['rects'] = merged_box
                new_row['text'] = ' '.join(text_parts)

                # For entity type, prefer 'article' if any of the merged boxes were articles
                new_row['entity_types'] = 'article' if 'article' in entity_types else entity_types[0]

                new_boxes.append(new_row)
                i = j

        # Add unmerged boxes
        for i, row in page_df.iterrows():
            if i not in merged_indices and i not in blacklist:
                new_boxes.append(row)

        merged_data.extend(new_boxes)

    return pd.DataFrame(merged_data).reset_index(drop=True)


# Modify the main function to include preprocessing
def main():
    # Load and preprocess data
    df = pd.read_csv(PROJECT_ROOT / 'final_unrolled_corrected.csv', converters={'rects': literal_eval})

    # Filter only article and title blocks
    df = df[df['entity_types'].isin(['article', 'title'])].copy()
    df['text'] = df['Texts_after_LLM'].where(df['Texts_after_LLM'] != ALERT_TEXT,
                                           df['After Sage Correction'])
    df['text'] = df['text'].str.strip()
    df = df[df['text'].str.len() >= MIN_ARTICLE_LENGTH].reset_index(drop=True)

    # Change type to "title" for small "article" blocks
    # df.loc[(df['entity_types'] == 'article') & (df['text'].str.len() < 20), 'entity_types'] = 'title'

    # Add preprocessing step
    print("Merging vertical columns...")
    df = merge_vertical_columns(df)

    # Process each page separately
    results = []
    for page_num, page_df in df.groupby('Page'):
        print(f"\nProcessing page {page_num} with {len(page_df)} blocks...")

        page_df = page_df.reset_index(drop=True)

        # Calculate normalized coordinates
        page_df = calculate_normalized_coords(page_df)

        # Generate embeddings
        if USE_BERT:
            print("Using BERT embeddings...")
            embeddings = generate_bert_embeddings(page_df['text'].tolist())
        else:
            print("Using TF-IDF embeddings...")
            embeddings = generate_tfidf_embeddings(page_df['text'].tolist())

        # Build similarity matrix combining content and spatial proximity
        similarity_matrix = build_hybrid_similarity_matrix(page_df, embeddings)

        # Cluster blocks into articles
        article_clusters = cluster_blocks(page_df, similarity_matrix)

        # Find titles for each article
        page_results = assign_titles_to_articles(page_df, article_clusters)
        results.extend(page_results)

        # Visualize results
        if SHOW_PAGE_LAYOUT:
            visualize_page_layout(page_df, page_results, page_num)

        # break

    # Save results
    result_df = pd.DataFrame(results)
    result_df.to_csv(PROJECT_ROOT / 'grouped_articles.csv', index=False)
    print("\nProcessing complete. Results saved to grouped_articles.csv")


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


def generate_tfidf_embeddings(texts: List[str]) -> np.ndarray:
    """Generate TF-IDF embeddings for texts."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix.toarray()


def generate_bert_embeddings(texts: List[str]) -> np.ndarray:
    """Generate BERT embeddings for texts."""
    return get_model().encode(texts, convert_to_tensor=False)


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


def cluster_blocks(df: pd.DataFrame, similarity_matrix: np.ndarray) -> List[List[int]]:
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


def assign_titles_to_articles(df: pd.DataFrame, article_clusters: List[List[int]]) -> List[Dict]:
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
            'newspaper_date': df.iloc[cluster[0]]['newspaper_date'],
            'newspaper_title': df.iloc[cluster[0]]['newspaper_title'],
            'page': df.iloc[cluster[0]]['Page'],
            'article_blocks': cluster,
            'title_block': best_title,
            'article_text': article_text,
            'title_text': title_text,
            'rects': [df.iloc[i]['rects'] for i in cluster],
            'title_rect': df.loc[best_title]['rects'] if best_title else None
        })

    return results


def visualize_page_layout(df: pd.DataFrame, clusters: list[dict[str, Any]], page_num: int):
    """Visualize the page layout with article groupings, image background, and reading order."""
    plt.figure(figsize=(12, 15))
    plt.title(f"Page {page_num} Article Grouping")

    # Add page image background if available
    if 'newspaper_title' in df.columns:
        img_path = df['newspaper_title'].iloc[0]
        img_path = img_path.replace("№ 19 (3348)", "№_19_(3348)") # FIXME
        try:
            img = plt.imread((PROJECT_ROOT / "val_data" / img_path))

            plt.imshow(img, alpha=0.5)
        except Exception as e:
            print(f"Could not load page image: {e}")

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
    plt.savefig(PROJECT_ROOT / f"page_layout_{page_num}.png")
    # plt.show()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()