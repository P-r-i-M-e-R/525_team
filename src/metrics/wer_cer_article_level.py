# FIXME: THESE METRICS ARE STUPID AND SAYS THE ERROR IS 200%

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.cer import CharErrorRate
from tqdm import tqdm, trange

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


VAL_DATA_DIR = PROJECT_ROOT / "val_articles"


def unroll_list_columns(df):
    # Reset index to keep original index as a column
    df = df.reset_index()

    # Create a MultiIndex Series by exploding all columns
    list_cols = ["Path", "After OCR", "After LLM", "rects", "original_paths", "entity_types"]
    df_list = df.copy()
    df_list[list_cols] = df[list_cols].apply(lambda x: x.apply(eval))
    # df_list = df
    s = df_list.apply(lambda x: x.explode())

    # The indices will align properly if all lists have same length
    return s.reset_index(drop=True)


def for_file(name: str) -> str:
    return name.lower().replace(" ", "_")


if __name__ == '__main__':
    val_items = [val_file.read_text() for val_file in VAL_DATA_DIR.rglob("*.md")]

    df = pd.read_csv(PROJECT_ROOT / "grouped_articles.csv")

    df_only_complete = df[~df["title_block"].isna()]

    wer = WordErrorRate()
    cer = CharErrorRate()

    wer_values = []
    cer_values = []

    for _, entity in tqdm(df_only_complete.iterrows(), total=len(df_only_complete), desc="Calculating metrics"):
        predicted_content = preprocess_text(entity["title_text"] + "\n" + entity["article_text"])
        min_err = float("inf")
        for i in trange(len(val_items), desc="Searching for best match"):
            true_content = preprocess_text(val_items[i])
            metric_value = cer(predicted_content, true_content)
            if metric_value < min_err:
                min_err = metric_value
                best_i = i

        best_match = val_items.pop(best_i)
        true_content = preprocess_text(best_match)

        wer.update(predicted_content, true_content)
        cer.update(predicted_content, true_content)

        wer_values.append(wer(predicted_content, true_content))
        cer_values.append(cer(predicted_content, true_content))

    wer_value = wer.compute()
    cer_value = cer.compute()
    print("Metrics for article matching")
    print(f"WER: {wer_value:.1%}")
    print(f"CER: {cer_value:.1%}")

    fig, ax = plt.subplots(1, 2, figsize=(20, 5), tight_layout=True)
    fig.suptitle(f"WER and CER for article matching")
    wer.plot(wer_values, ax=ax[0])
    ax[0].set_title(f"Total WER = {wer_value:.1%}")
    cer.plot(cer_values, ax=ax[1])
    ax[1].set_title(f"Total CER = {cer_value:.1%}")
    plt.savefig(
        PROJECT_ROOT / f"metrics_report/article_matching/wer_cer.png")
