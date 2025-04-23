# FIXME: usually makes everything worse

import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

from src.config import PROJECT_ROOT

DATA_ROOT = PROJECT_ROOT / 'downscaled_data'
OUTPUT_DIR = DATA_ROOT.parent / 'preprocessed_data'
CONTRAST_FACTOR = 1.5  # Adjust this value to control the contrast enhancement
THRESHOLD = 150  # Threshold for R, G, B
DIFF_THRESHOLD = 100  # Maximum allowed difference between R, G, B


def preprocess_image(input_path: Path, output_path: Path, contrast_factor: float) -> None:
    """Preprocess an image to improve OCR accuracy and save it to the output path."""
    with Image.open(input_path) as img:
        # Step 2: Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

        # Step 3: Remove noise using a median filter
        img = img.filter(ImageFilter.MedianFilter(size=3))

        # Step 4: Apply binarization with the given formula
        img_array = np.array(img)
        if img_array.ndim == 3:  # Ensure the image has multiple channels (e.g., RGB)
            r, g, b = img_array[..., 0], img_array[..., 1], img_array[..., 2]
            condition = (
                (r < THRESHOLD) & (g < THRESHOLD) & (b < THRESHOLD) &
                (np.max(np.stack([np.abs(r - g), np.abs(r - b), np.abs(g - b)], axis=0), axis=0) < DIFF_THRESHOLD)
            )
            binary_array = (1 - condition.astype(np.uint8)) * 255  # Convert to binary (0 or 255)
        else:
            raise ValueError("Input image must have 3 channels (RGB).")

        binary_img = Image.fromarray(binary_array, mode="L")  # Convert back to grayscale image

        # Save the preprocessed image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        binary_img.save(output_path)


if __name__ == '__main__':
    for image_path in DATA_ROOT.rglob('*.jpg'):  # Adjust extension if needed
        relative_path = image_path.relative_to(DATA_ROOT)
        output_path = OUTPUT_DIR / relative_path
        preprocess_image(image_path, output_path, CONTRAST_FACTOR)