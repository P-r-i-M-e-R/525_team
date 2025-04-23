from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import PROJECT_ROOT
from src.dataset.annotations_to_df import NO_ID

ROOT = Path("/home/jupyter/datasphere/project/Data/cropped_blocks_no_matching/")
ORIG_PATH_PREFIX = Path("Вечерняя Москва/1935/№_19_(3348)/")


def get_bbox_by_path(path: str):
    rel_path = Path(path).relative_to(ROOT)
    page = rel_path.parts[0]
    file_name = rel_path.with_suffix("").name
    components = file_name.split("_")[1:]
    if components[0].isnumeric() and not components[1].startswith("newspaper"):
        entity_idx = int(components[0])
    else:
        entity_idx = NO_ID
    entity_type = components[1:-1]
    entity_type = "_".join(entity_type)
    block_idx = int(components[-1])

    filter_by_page = df_annot["relpath"].str.endswith(f"{page}.json")
    filter_by_entity_idx = df_annot["entity_id"] == entity_idx
    filter_by_entity_type = df_annot["entity_label"] == entity_type

    filtered_entities = df_annot[filter_by_page & filter_by_entity_idx & filter_by_entity_type]
    assert len(filtered_entities) > 0, (path, entity_idx, entity_type, block_idx)
    entity = filtered_entities.iloc[block_idx, :]
    return entity["rect"]


def get_original_path(path: str):
    rel_path = Path(path).relative_to(ROOT)
    page = rel_path.parts[0]
    file_name = rel_path.with_suffix("").name
    components = file_name.split("_")[1:]
    if components[0].isnumeric() and not components[1].startswith("newspaper"):
        entity_idx = int(components[0])
    else:
        entity_idx = None
    entity_type = components[1:-1]
    entity_type = "_".join(entity_type)
    block_idx = int(components[-1])

    if entity_idx is None:
        original_rel_path = Path(page) / entity_type / f"block_{block_idx}.jpg"
    else:
        original_rel_path = Path(page) / str(entity_idx) / entity_type / f"block_{block_idx}.jpg"

    return str(ORIG_PATH_PREFIX / original_rel_path)


if __name__ == '__main__':
    tqdm.pandas()

    df_ocr = pd.read_csv(PROJECT_ROOT / "ocr.csv")

    # Convert str columns to list
    df_ocr["Path"] = df_ocr["Path"].apply(lambda x: eval(x))
    df_ocr["After OCR"] = df_ocr["After OCR"].apply(lambda x: eval(x))
    df_ocr["After LLM"] = df_ocr["After LLM"].apply(lambda x: eval(x))

    df_annot = pd.read_json(PROJECT_ROOT / "annotations.json", orient="records")

    df_ocr["rects"] = df_ocr["Path"].progress_apply(lambda paths: [get_bbox_by_path(path) for path in paths])
    df_ocr["original_paths"] = df_ocr["Path"].progress_apply(lambda paths: [get_original_path(path) for path in paths])

    df_ocr.to_csv(PROJECT_ROOT / "ocr_with_boxes.csv", index=False)


