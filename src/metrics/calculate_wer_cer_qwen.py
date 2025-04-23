from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.cer import CharErrorRate
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT
from src.metrics.calculate_wer_cer import preprocess_text

VAL_DATA_DIR = PROJECT_ROOT / "ocr_data_validated"
PRED_DATA_DIR = PROJECT_ROOT / "ocr_data"

if __name__ == '__main__':
    wer = WordErrorRate()
    cer = CharErrorRate()

    wer_values = []
    cer_values = []

    skipped_blocks = 0

    for md_file in tqdm(list(VAL_DATA_DIR.rglob("*.md"))):
        with open(md_file, "r") as f:
            true_content = preprocess_text(f.read())

        path_to_search = PRED_DATA_DIR / md_file.relative_to(VAL_DATA_DIR)
        assert path_to_search.exists(), path_to_search
        predicted_content = preprocess_text(path_to_search.read_text())

        wer.update(predicted_content, true_content)
        cer.update(predicted_content, true_content)

        wer_values.append(wer(predicted_content, true_content))
        cer_values.append(cer(predicted_content, true_content))

    print(f"WER: {wer.compute():.2%}")
    print(f"CER: {cer.compute():.2%}")
    if skipped_blocks:
        print(f"Skipped {skipped_blocks} blocks")

    fig, ax = plt.subplots(1, 2, figsize=(20, 5), tight_layout=True)
    fig.suptitle("WER and CER for Qwen 2 VL")
    wer.plot(wer_values, ax=ax[0])
    ax[0].set_title("WER")
    cer.plot(cer_values, ax=ax[1])
    ax[1].set_title("CER")
    plt.savefig(PROJECT_ROOT / "metrics_report/ocr/qwen_metrics.png")