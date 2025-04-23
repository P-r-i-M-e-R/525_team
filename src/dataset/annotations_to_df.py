import json

from tqdm import tqdm
import pandas as pd

from src.config import PROJECT_ROOT

DATA_ROOT = PROJECT_ROOT / "val_data"
NO_ID = -999
FIGURE_OFFSET = 1000

def fix_rectangle(rect: list[list[int, int], list[int, int]]) -> list[list[int]]:
    (x1, y1), (x2, y2) = rect
    upper_left = [min(x1, x2), min(y1, y2)]
    lower_right = [max(x1, x2), max(y1, y2)]
    return [upper_left, lower_right]

if __name__ == '__main__':
    annotations = []
    annotation_files = list(DATA_ROOT.rglob('*.json'))
    for annotation_path in tqdm(annotation_files):
        rel_path = annotation_path.relative_to(DATA_ROOT)
        with open(annotation_path) as f:
            annotation = json.load(f)
            for shape in annotation['shapes']:
                rect = fix_rectangle(shape['points'])
                *entity_label, entity_id = shape['label'].split('_')
                if entity_id.isdigit():
                    entity_id = int(entity_id)
                    entity_label = '_'.join(entity_label)
                    if entity_label.startswith("figure"):
                        entity_id += FIGURE_OFFSET
                else:
                    entity_id = NO_ID
                    entity_label = shape['label']
                annotations.append([str(rel_path), rect, entity_label, entity_id])


    df = pd.DataFrame(annotations, columns=['relpath', 'rect', 'entity_label', 'entity_id'])
    df.to_json(PROJECT_ROOT / 'annotations.json', orient='records', force_ascii=False)


