import aiohttp
import asyncio
import pandas as pd
from pathlib import Path
import logging
import aiofiles
from tqdm import tqdm
from asyncio import Queue

from src.config import PROJECT_ROOT

DATA_ROOT = PROJECT_ROOT / 'data'
NUM_CONCURRENT = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def download_file(session: aiohttp.ClientSession, url: str, filepath: Path, progress_bar: tqdm) -> None:
    """Download a single file asynchronously"""
    if filepath.is_file() and filepath.stat().st_size > 0:
        await asyncio.sleep(0.01)
        progress_bar.update(1)
        return

    try:
        async with session.get(url) as response:
            if response.status == 200:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(filepath, mode='wb') as f:
                    await f.write(await response.read())
                progress_bar.update(1)
            else:
                logging.error(f"Failed to download {url}, status: {response.status}")
    except Exception as e:
        logging.error(f"Error downloading {url}: {str(e)}")
    finally:
        progress_bar.update(0)


async def worker(queue: Queue, session: aiohttp.ClientSession,
                 semaphore: asyncio.Semaphore, progress_bar: tqdm) -> None:
    """Worker that processes download tasks from the queue"""
    while True:
        try:
            item = await queue.get()
            if item is None:
                break

            async with semaphore:
                await download_file(session, item['url'], Path(item['path']), progress_bar)

            queue.task_done()
        except Exception as e:
            logging.error(f"Worker error: {str(e)}")
            queue.task_done()


def sample_dataset(df: pd.DataFrame, fraction: float = 0.1) -> pd.DataFrame:
    """Sample equal fraction of issues (keeping all pages) from each newspaper and year combination"""
    if not 0 < fraction <= 1:
        raise ValueError("Fraction must be between 0 and 1")

    sampled_data = []

    for (newspaper, year), year_group in df.groupby(['newspaper_name', 'newspaper_year']):
        unique_issues = year_group['newspaper_index'].unique()

        sample_size = max(1, int(len(unique_issues) * fraction))

        sampled_issues = pd.Series(unique_issues).sample(n=sample_size, random_state=42)

        sampled_group = year_group[year_group['newspaper_index'].isin(sampled_issues)]
        sampled_data.append(sampled_group)

        logging.info(f"Sampled {sample_size} issues ({len(sampled_group)} pages) from {newspaper} {year}")

    return pd.concat(sampled_data, ignore_index=True)


async def main(fraction: float = 0.1):
    df = pd.read_csv(PROJECT_ROOT / 'newspapers_no_ocr_splitted_title.csv').drop(columns=['Unnamed: 0'])
    # df = pd.read_csv('newspapers_sample.csv')

    df = sample_dataset(df, fraction)
    logging.info(f"Total selected files: {len(df)}")
    df.to_csv("newspapers_sample.csv", index=False)
    # df.to_csv("newspapers_val_sample.csv", index=False)
    # exit()

    downloads = []
    for _, row in tqdm(df.iterrows(), "Preparing tasks", total=len(df)):
        url = row['origin_img']
        newspaper = row['newspaper_name']
        year = str(row['newspaper_year'])
        issue = row['newspaper_index'].replace(' ', '_')
        page = f"{row['newspaper_page']}.jpg"

        filepath = DATA_ROOT / newspaper / year / issue / page
        downloads.append({
            'url': url,
            'path': filepath
        })

    queue = Queue()
    semaphore = asyncio.Semaphore(NUM_CONCURRENT)

    progress_bar = tqdm(total=len(downloads), desc="Downloading files")

    async with aiohttp.ClientSession() as session:

        workers = [
            asyncio.create_task(
                worker(queue, session, semaphore, progress_bar)
            )
            for _ in range(NUM_CONCURRENT)
        ]

        for download in downloads:
            await queue.put(download)

        for _ in range(NUM_CONCURRENT):
            await queue.put(None)

        await asyncio.gather(*workers)

    progress_bar.close()


if __name__ == "__main__":
    DATA_ROOT.mkdir(exist_ok=True)
    asyncio.run(main(0.04))
