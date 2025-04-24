from ast import literal_eval

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

MIN_ARTICLE_LENGTH = 3


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

def merge_vertical_columns(df: pd.DataFrame, y_threshold: int = 50) -> pd.DataFrame:
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


if __name__ == '__main__':
    df = pd.read_csv(PROJECT_ROOT / 'ocr_val_unrolled_corrected.csv', converters={'rects': literal_eval})

    # Filter only article and title blocks
    df = df[df['entity_types'].isin(['article', 'title'])].copy()
    df['text'] = df['After Sage Correction'].str.strip()
    df = df[df['text'].str.len() >= MIN_ARTICLE_LENGTH].reset_index(drop=True)

    # Add preprocessing step
    print("Merging vertical columns...")
    df = merge_vertical_columns(df)

    df.to_csv(PROJECT_ROOT / 'ocr_val_unrolled_corrected_merged.csv', index=False)