from enum import StrEnum

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.cer import CharErrorRate
from tqdm import tqdm

import re
import string

from src.config import PROJECT_ROOT


def preprocess_text(text: str) -> str:
    # Define a pattern that includes all punctuation and various quotation marks
    punctuation_pattern = f"[{re.escape(string.punctuation)}«»„“”‘’‹›〝〞〟＂]"

    # Remove all punctuation and quotation marks
    text = re.sub(punctuation_pattern, "", text.lower())

    # Remove any remaining whitespace and return
    return text.strip()

VAL_DATA_DIR = PROJECT_ROOT / "ocr_data_validated"
ALERT_TEXT = "В интернете есть много сайтов с информацией на эту тему. [Посмотрите, что нашлось в поиске](https://ya.ru)"


class PlotTypes(StrEnum):
    only_ocr = "Only OCR"
    yandex_gpt = "YandexGPT"
    sage_t5 = "Sage T5"

class SkipTypes(StrEnum):
    no_skipping = "without skipping"
    skipping = "with skipping"


def unroll_list_columns(df):
    # Reset index to keep original index as a column
    df = df.reset_index()

    # Create a MultiIndex Series by exploding all columns
    list_cols = ["Path", "After OCR", "After LLM", "rects", "original_paths"]
    df_list = df.copy()
    df_list[list_cols] = df[list_cols].apply(lambda x: x.apply(eval))
    # df_list = df
    s = df_list.apply(lambda x: x.explode())

    # The indices will align properly if all lists have same length
    return s.reset_index(drop=True)

def for_file(name: str) -> str:
    return name.lower().replace(" ", "_")

if __name__ == '__main__':
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # df_ocr = pd.read_csv("../ocr_with_boxes.csv").set_index("Page")
    #
    # df_unrolled = unroll_list_columns(df_ocr)
    # df_unrolled.to_csv("../ocr_with_boxes_unrolled.csv", index=False)

    df_unrolled = pd.read_csv(PROJECT_ROOT / "ocr_with_boxes_unrolled_corrected.csv")

    print(df_unrolled.head())

    # for SELECTED_PLOT in [(PlotTypes.only_ocr, SkipTypes.no_skipping), (PlotTypes.only_ocr, SkipTypes.skipping), (PlotTypes.yandex_gpt, SkipTypes.no_skipping), (PlotTypes.yandex_gpt, SkipTypes.skipping)]:
    for SELECTED_PLOT in [(PlotTypes.sage_t5, SkipTypes.no_skipping), (PlotTypes.sage_t5, SkipTypes.skipping)]:
        wer = WordErrorRate()
        cer = CharErrorRate()

        skipped_blocks = 0

        wer_values = []
        cer_values = []

        for md_file in tqdm(list(VAL_DATA_DIR.rglob("*.md"))):
            with open(md_file, "r") as f:
                true_content = preprocess_text(f.read())

            path_to_search = md_file.relative_to(VAL_DATA_DIR).with_suffix(".jpg")
            matched_df = df_unrolled[df_unrolled["original_paths"] == str(path_to_search)]
            assert len(matched_df) == 1, (matched_df, path_to_search)
            predicted_content = preprocess_text(matched_df["After LLM"].iloc[0])
            if SELECTED_PLOT[1] is SkipTypes.skipping and predicted_content == preprocess_text(ALERT_TEXT):
                skipped_blocks += 1
                continue

            if SELECTED_PLOT[0] is PlotTypes.only_ocr:
                predicted_content = preprocess_text(matched_df["After OCR"].iloc[0])

            if SELECTED_PLOT[0] is PlotTypes.sage_t5:
                predicted_content = preprocess_text(matched_df["After Sage Correction"].iloc[0])

            wer.update(predicted_content, true_content)
            cer.update(predicted_content, true_content)

            wer_values.append(wer(predicted_content, true_content))
            cer_values.append(cer(predicted_content, true_content))

            if wer_values[-1] > 1 or cer_values[-1] > 1:
                print("!!!!!!!!!! SUS !!!!!!!!!!!")
                print(true_content)
                print()
                print(predicted_content)
                print()
                print(wer_values[-1], cer_values[-1])
                wer_values[-1] = torch.tensor(min(1, wer_values[-1]))
                cer_values[-1] = torch.tensor(min(1, cer_values[-1]))
                # break

        wer_value = wer.compute()
        cer_value = cer.compute()
        print("Metrics for", SELECTED_PLOT[0], SELECTED_PLOT[1])
        print(f"WER: {wer_value:.1%}")
        print(f"CER: {cer_value:.1%}")
        if skipped_blocks:
            print(f"Skipped {skipped_blocks} blocks")

        fig, ax = plt.subplots(1, 2, figsize=(20, 5), tight_layout=True)
        fig.suptitle(f"WER and CER for {SELECTED_PLOT[0]} {SELECTED_PLOT[1]}")
        wer.plot(wer_values, ax=ax[0])
        ax[0].set_title(f"Total WER = {wer_value:.1%}")
        cer.plot(cer_values, ax=ax[1])
        ax[1].set_title(f"Total CER = {cer_value:.1%}")
        plt.savefig(PROJECT_ROOT / f"metrics_report/ocr/{for_file(SELECTED_PLOT[0])}_metrics_{for_file(SELECTED_PLOT[1])}.png")