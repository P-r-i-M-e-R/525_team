from datasets import tqdm

from src.config import PROJECT_ROOT

response_prefix = """system
You are a helpful assistant.
user
Your goal is to restore the text in Russian from the newspaper scan as close to original as possible. 
        Pay attention to:
        1. Preserve all characters even if partially erased
        2. Maintain original structure (headings, paragraphs)
        3. Keep original formatting (line breaks, spacing)
        4. Output in markdown format
        
assistant"""

DATA_ROOT = PROJECT_ROOT / "ocr_data"

if __name__ == '__main__':
    for md_file in tqdm(list(DATA_ROOT.rglob("*.md"))):
        with open(md_file, "r") as f:
            content = f.read()
        content = content.replace(response_prefix, "").replace("```markdown", "").replace("```", "").strip()
        with open(md_file, "w") as f:
            f.write(content)