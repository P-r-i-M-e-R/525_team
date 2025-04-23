from typing import NamedTuple

from src.config import PROJECT_ROOT

PAGE_ROOT = PROJECT_ROOT / "ocr_data_validated" / "Вечерняя Москва" / "1935" / "№_19_(3348)" / "1"


class Article(NamedTuple):
    title: str
    content: str

if __name__ == '__main__':
    articles = []
    for entity in sorted(PAGE_ROOT.iterdir()):
        if not entity.name.isnumeric():
            continue
        title = "# " + " ".join((entity / "title" / "block_0.md").read_text().strip().split("\n"))
        article_blocks = []
        for block in sorted((entity / "article").iterdir()):
            article_blocks.append(block.read_text())
        content = "\n".join(article_blocks)
        articles.append(Article(title=title, content=content))

    with open(PROJECT_ROOT / "page.md", "w") as f:
        for article in articles:
            f.write(article.title + "\n")
            f.write(article.content + "\n\n")

