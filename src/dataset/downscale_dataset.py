from pathlib import Path
from PIL import Image

from src.config import PROJECT_ROOT

DATA_ROOT = PROJECT_ROOT / 'test-data'
OUTPUT_DIR = DATA_ROOT.parent / 'downscaled_data'
DOWNSCALE_FACTOR = 1.5

def downscale_image(input_path: Path, output_path: Path, factor: float | int) -> None:
    """Downscale an image by a given factor and save it to the output path."""
    with Image.open(input_path) as img:
        new_size = (int(img.width // factor), int(img.height // factor))
        downscaled_img = img.resize(new_size, Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        downscaled_img.save(output_path)

if __name__ == '__main__':
    for image_path in DATA_ROOT.rglob('*.jpg'):
        relative_path = image_path.relative_to(DATA_ROOT)
        output_path = OUTPUT_DIR / relative_path
        downscale_image(image_path, output_path, DOWNSCALE_FACTOR)