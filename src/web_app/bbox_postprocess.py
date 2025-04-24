import pandas as pd


def merge_vertical_columns(df: pd.DataFrame, y_threshold: int = 50) -> pd.DataFrame:
    """
    Merge bboxes that are in the same column (same x bounds) and vertically close.

    Args:
        df: DataFrame containing bbox coordinates in columns x1,y1,x2,y2
        y_threshold: Maximum vertical gap between boxes to merge in pixels

    Returns:
        DataFrame with merged boxes
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    print("Boxes before merge:", len(df))

    # Group by page if 'Page' column exists, otherwise process whole dataframe as one group
    group_key = 'Page' if 'Page' in df.columns else None
    grouped = df.groupby(group_key) if group_key else [(None, df)]

    merged_data = []

    for page_num, page_df in grouped:
        # Create a list of all boxes with their original indices and coordinates
        boxes = [(i, (row['x1'], row['y1'], row['x2'], row['y2']))
                 for i, row in page_df.iterrows()]

        # Sort boxes by x1 (left) position
        boxes.sort(key=lambda x: (x[1][0], x[1][1]))

        merged_indices = set()
        new_boxes = []

        # Group boxes into columns
        columns = []
        current_column = []
        blacklist = set()

        for i, box in boxes:
            box_width = box[2] - box[0]  # x2 - x1
            box_height = box[3] - box[1]  # y2 - y1

            # Skip very wide boxes (likely horizontal lines or full-width elements)
            if box_width / box_height > 10:
                blacklist.add(i)
                continue

            if not current_column:
                current_column.append((i, box))
            else:
                # Check if box belongs to current column (similar x bounds)
                last_box = current_column[-1][1]
                x_overlap = min(box[2], last_box[2]) - max(box[0], last_box[0])

                curr_width = box[2] - box[0]
                prev_width = last_box[2] - last_box[0]
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
            column.sort(key=lambda x: x[1][1])  # Sort by y1

            i = 0
            while i < len(column):
                current_idx, current_box = column[i]
                if current_idx in merged_indices:
                    i += 1
                    continue

                merged_box = current_box
                class_parts = [page_df.iloc[current_idx]['class']]
                confidence_values = [page_df.iloc[current_idx]['confidence']]
                merged_indices.add(current_idx)

                # Look ahead for boxes to merge
                j = i + 1
                while j < len(column):
                    next_idx, next_box = column[j]
                    y_gap = next_box[1] - merged_box[3]  # Distance between bottom of current and top of next

                    if y_gap < y_threshold:
                        # Merge boxes
                        merged_box = (
                            min(merged_box[0], next_box[0]),  # new x1
                            min(merged_box[1], next_box[1]),  # new y1
                            max(merged_box[2], next_box[2]),  # new x2
                            max(merged_box[3], next_box[3])  # new y2
                        )
                        class_parts.append(page_df.iloc[next_idx]['class'])
                        confidence_values.append(page_df.iloc[next_idx]['confidence'])
                        merged_indices.add(next_idx)
                        j += 1
                    else:
                        break

                # Create new merged row
                new_row = page_df.iloc[current_idx].copy()
                new_row['x1'] = merged_box[0]
                new_row['y1'] = merged_box[1]
                new_row['x2'] = merged_box[2]
                new_row['y2'] = merged_box[3]
                new_row['width'] = merged_box[2] - merged_box[0]
                new_row['height'] = merged_box[3] - merged_box[1]

                # For class, join with semicolon if different, otherwise keep single value
                if len(set(class_parts)) > 1:
                    new_row['class'] = ';'.join(sorted(set(class_parts)))
                else:
                    new_row['class'] = class_parts[0]

                # For confidence, take the average
                new_row['confidence'] = sum(confidence_values) / len(confidence_values)

                new_boxes.append(new_row)
                i = j

        # Add unmerged boxes
        for i, row in page_df.iterrows():
            if i not in merged_indices and i not in blacklist:
                new_boxes.append(row)

        merged_data.extend(new_boxes)

    # Create new dataframe and reset index
    result_df = pd.DataFrame(merged_data).reset_index(drop=True)
    print("Boxes after post-processing:", len(result_df))
    return result_df