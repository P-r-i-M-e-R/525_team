from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm

from src.config import PROJECT_ROOT
from src.dataset.annotations_to_df import NO_ID

DATA_ROOT = PROJECT_ROOT / "val_data"
OUTPUT_ROOT = PROJECT_ROOT / "cropped_blocks"


def crop_and_save_blocks(df: pd.DataFrame):
    # Group by relative path (document)
    for rel_path, doc_group in df.groupby('relpath'):
        # Get the image path by replacing .json with .png (or other image extensions)
        image_path = DATA_ROOT / rel_path
        ext = ".jpg"

        test_path = image_path.with_suffix(ext)
        image_path = test_path

        if not image_path.exists():
            print(f"Warning: Could not find image file for {image_path}")
            continue

        # Open the image once for all crops
        try:
            img = Image.open(image_path)
        except Exception as e:
            print(f"Warning: Could not open image {image_path}: {e}")
            continue

        # Create path components
        rel_path_without_filename = str(Path(rel_path).parent)
        page = Path(rel_path).stem  # filename without extension

        grouped = doc_group.groupby(['entity_id', 'entity_label'])

        # Group by entity_id and entity_label
        for (entity_id, entity_label), entity_group in tqdm(grouped, total=len(grouped), desc=f"Blocks for {rel_path}"):
            # Create output directory for this entity
            if entity_id != NO_ID:
                entity_dir = OUTPUT_ROOT / rel_path_without_filename / page / str(entity_id) / entity_label
            else:
                entity_dir = OUTPUT_ROOT / rel_path_without_filename / page / entity_label

            entity_dir.mkdir(parents=True, exist_ok=True)

            # Crop each block
            for i, (_, row) in enumerate(entity_group.iterrows()):
                (x1, y1), (x2, y2) = row['rect']
                try:
                    cropped_img = img.crop((x1, y1, x2, y2))
                    output_path = entity_dir / f"block_{i}.jpg"
                    cropped_img.save(output_path)
                except Exception as e:
                    print(f"Warning: Could not crop image {image_path} at {row['rect']}: {e}")


if __name__ == '__main__':
    # Load annotations
    df = pd.read_json(PROJECT_ROOT / 'annotations.json')

    # Create output directory if it doesn't exist
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Process all annotations
    crop_and_save_blocks(df)

    print(f"Finished cropping images. Results saved to {OUTPUT_ROOT}")