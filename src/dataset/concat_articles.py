from tqdm import tqdm

from src.config import PROJECT_ROOT

BLOCKS_DIR = PROJECT_ROOT / "ocr_data_validated" / "Вечерняя Москва" / "1935" / "№_19_(3348)" / "1"
ARTICLE_DIR = PROJECT_ROOT / "val_articles"
ARTICLE_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    for entity in tqdm(sorted(BLOCKS_DIR.iterdir())):
        if not entity.name.isnumeric():
            continue
        title = "# " + " ".join((entity / "title" / "block_0.md").read_text().strip().split("\n"))
        article_blocks = []
        for block in sorted((entity / "article").iterdir()):
            article_blocks.append(block.read_text())
        content = "\n".join(article_blocks)
        with open(ARTICLE_DIR / f"{entity.name}.md", "w") as f:
            f.write(title + "\n")
            f.write(content + "\n\n")