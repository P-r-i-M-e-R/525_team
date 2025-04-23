import shutil
from pathlib import Path

from datasets import tqdm

from src.config import PROJECT_ROOT

DATA_ROOT = PROJECT_ROOT / "cropped_blocks"
OUTPUT_ROOT = PROJECT_ROOT / "cropped_blocks_no_matching"

if __name__ == '__main__':
    block_file: list[Path] = list(DATA_ROOT.rglob('*.jpg'))
    for block_path in tqdm(block_file):
        rel_path = block_path.relative_to(DATA_ROOT)
        file_name = rel_path.with_suffix('').name
        block_idx = file_name.split('_')[1]
        page_number = rel_path.parts[3]
        entity_idx: str = rel_path.parent.parent.name
        if entity_idx.isnumeric():
            entity_idx = int(entity_idx)
        else:
            entity_idx = None
        entity_type = rel_path.parent.name
        if entity_idx is not None:
            out_name = f"block_{entity_idx}_{entity_type}_{block_idx}.jpg"
        else:
            out_name = f"block_{entity_type}_{block_idx}.jpg"
        output_path = OUTPUT_ROOT / page_number / out_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(block_path, output_path)