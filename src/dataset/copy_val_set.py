import shutil

import pandas as pd
from pathlib import Path

from tqdm import tqdm

from src.config import PROJECT_ROOT

DATA_ROOT = PROJECT_ROOT / "data"

if __name__ == '__main__':
    df = pd.read_csv(PROJECT_ROOT / "newspapers_val_sample.csv")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        rel_path = Path(row['newspaper_name']) / str(row['newspaper_year']) / row['newspaper_index'].replace(" ", "_") / f"{row['newspaper_page']}.jpg"
        origin_path = DATA_ROOT / rel_path
        target_path = DATA_ROOT / ".." / 'val_data' / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(origin_path, target_path)
