import torch
from tqdm import trange


def preprocess_text(text: str):
    if len(text) > 100:
        return "".join(text.split("\n"))
    else:
        return " ".join(text.split("\n"))

def correct_spelling_batch(model, tokenizer, texts, batch_size=32):
    texts = [preprocess_text(text) if isinstance(text, str) else "" for text in texts]
    corrected_texts = []

    # Process in batches
    for i in trange(0, len(texts), batch_size, desc="Spelling correction batches"):
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