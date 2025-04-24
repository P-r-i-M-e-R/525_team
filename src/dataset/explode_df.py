from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT


def unroll_list_columns(df):
    # Reset index to keep original index as a column
    df = df.reset_index()

    # Create a MultiIndex Series by exploding all columns
    list_cols = ["bboxes", "Texts_after_OCR", "Texts_after_LLM"]
    df_list = df.copy()
    df_list[list_cols] = df[list_cols].apply(lambda x: x.apply(eval))

    s = df_list.apply(lambda x: x.explode())

    return s.reset_index(drop=True)

def extract_entity_type(bbox: list) -> str:
    orig_type = bbox[-1]
    match orig_type:
        case "Text":
            return "article"
        case "Section-header":
            return "title"
        case _:
            return orig_type

def extract_box(bbox: list) -> list:
    x1, y1, x2, y2, _ = bbox
    return [[x1, y1], [x2, y2]]


if __name__ == '__main__':
    df_ocr = pd.read_csv(PROJECT_ROOT / "FINAL.csv")
    # df_ocr["Page"] = df_ocr["newspaper_title"].apply(lambda x: Path(x).with_suffix("").name)
    df_ocr = df_ocr.set_index("Page").drop([4, 13])

    df_unrolled = unroll_list_columns(df_ocr)
    df_unrolled["entity_types"] = df_unrolled["bboxes"].apply(extract_entity_type)
    df_unrolled["rects"] = df_unrolled["bboxes"].apply(extract_box)
    df_unrolled["After OCR"] = df_unrolled["Texts_after_OCR"]
    df_unrolled.to_csv(PROJECT_ROOT / "final_unrolled.csv", index=False)

