from PIL import Image
from tqdm import tqdm
from unsloth import FastVisionModel
from transformers import TextStreamer

from src.config import PROJECT_ROOT

DATA_ROOT = PROJECT_ROOT / "val_data"
OUTPUT_ROOT = PROJECT_ROOT / "cropped_blocks"
OCR_RESULTS_ROOT = PROJECT_ROOT / "ocr_results"


# Initialize model and tokenizer
def initialize_model():
    print("Loading Qwen VL model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen2-VL-7B-Instruct",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer


# Process a single image with the model
def process_image(model, tokenizer, img_path):
    try:
        img = Image.open(img_path)

        instruction = """Your goal is to restore the text in Russian from the newspaper scan as close to original as possible. 
        Pay attention to:
        1. Preserve all characters even if partially erased
        2. Maintain original structure (headings, paragraphs)
        3. Keep original formatting (line breaks, spacing)
        4. Output in markdown format
        """

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]

        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            img,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        text_streamer = TextStreamer(tokenizer, skip_prompt=True)

        outputs = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=1024,
            use_cache=True,
            temperature=0.7,
            top_p=0.9,
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None


# Process all blocks and save OCR results
def process_all_blocks(model, tokenizer):
    # Walk through the directory structure
    for block_dir in tqdm(list(OUTPUT_ROOT.rglob("block_*.png")), desc="Processing blocks"):
        # Get the relative path structure
        rel_path = block_dir.relative_to(OUTPUT_ROOT)

        # Create corresponding output directory
        output_dir = OCR_RESULTS_ROOT / rel_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Output file path (replace .png with .txt)
        output_file = output_dir / f"{block_dir.stem}.md"

        # Skip if already processed
        if output_file.exists():
            continue

        # Process the image
        result = process_image(model, tokenizer, block_dir)

        # Save the result
        if result:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)


# Main function
def main():
    # Initialize model
    model, tokenizer = initialize_model()

    # Create output directory if it doesn't exist
    OCR_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Process all blocks
    process_all_blocks(model, tokenizer)

    print(f"Finished OCR processing. Results saved to {OCR_RESULTS_ROOT}")


if __name__ == '__main__':
    main()