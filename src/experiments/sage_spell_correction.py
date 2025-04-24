import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch

from src.config import PROJECT_ROOT

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sage-fredt5-distilled-95m")
model = AutoModelForSeq2SeqLM.from_pretrained("ai-forever/sage-fredt5-distilled-95m")
model.to("cuda")

def preprocess_text(text: str):
    if len(text) > 100:
        return "".join(text.split("\n"))
    else:
        return " ".join(text.split("\n"))

# Function to process batch of texts
def correct_spelling_batch(texts, batch_size=32):
    corrected_texts = []

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch = texts[i:i + batch_size]

        # Tokenize the batch
        inputs = tokenizer(
            batch,
            max_length=None,
            padding="longest",
            truncation=False,
            return_tensors="pt"
        )

        # Generate corrections
        with torch.no_grad():
            outputs = model.generate(
                **inputs.to(model.device),
                max_length=inputs["input_ids"].size(1) * 1.5
            )

        # Decode the outputs
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        corrected_texts.extend(decoded)

    return corrected_texts


# Load the unrolled DataFrame
df_unrolled = pd.read_csv(PROJECT_ROOT / "ocr_with_boxes_unrolled.csv")

# Apply spelling correction to the "After OCR" column
print("Starting spelling correction...")
corrected_texts = correct_spelling_batch([preprocess_text(text) if isinstance(text, str) else "" for text in df_unrolled["After OCR"].tolist()])

# Add corrected texts as a new column
df_unrolled["After Sage Correction"] = corrected_texts

# Save the results
output_path = PROJECT_ROOT / "ocr_with_boxes_unrolled_corrected.csv"
df_unrolled.to_csv(output_path, index=False)
print(f"Spelling correction complete. Results saved to {output_path}")