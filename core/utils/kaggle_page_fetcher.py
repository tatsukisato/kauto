# kaggle_page_fetcher.py
"""Utility to fetch full text of Kaggle competition Overview and Data pages.

This script uses plain HTTP requests (with a realistic User-Agent) and BeautifulSoup
to retrieve the visible text of the pages. It works without the browser subagent,
so it can be executed directly in the environment.

Usage:
    uv run python core/utils/kaggle_page_fetcher.py <competition_slug>

Example:
    uv run python core/utils/kaggle_page_fetcher.py titanic
"""

import argparse
import requests
from bs4 import BeautifulSoup
from pathlib import Path

def _fetch_page_text(url: str) -> str:
    """Fetch the visible text of a Kaggle page.
    Returns a plain‑text representation of the main article or body.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return f"Failed to fetch (status {resp.status_code})"
        soup = BeautifulSoup(resp.text, "html.parser")
        # Prefer the <article> element if Kaggle provides one
        article = soup.find("article")
        if article:
            return article.get_text("\n", strip=True)
        # Fallback to the whole body text
        return soup.body.get_text("\n", strip=True) if soup.body else "No content"
    except Exception as e:
        return f"Error fetching page: {e}"

def fetch_competition_pages(competition_name: str, base_dir: str = "./competitions"):
    """Fetch Overview and Data pages for a competition and write markdown files.

    The files are saved as:
        <base_dir>/<competition>/OVERVIEW.md
        <base_dir>/<competition>/DATA.md
    """
    comp_dir = Path(base_dir) / competition_name
    comp_dir.mkdir(parents=True, exist_ok=True)

    overview_url = f"https://www.kaggle.com/competitions/{competition_name}/overview"
    data_url = f"https://www.kaggle.com/competitions/{competition_name}/data"

    overview_text = _fetch_page_text(overview_url)
    data_text = _fetch_page_text(data_url)

    docs_dir = comp_dir / "docs"
    docs_dir.mkdir(exist_ok=True)

    (docs_dir / "OVERVIEW.md").write_text(f"# {competition_name} Overview\n\n{overview_text}\n", encoding="utf-8")
    (docs_dir / "DATA.md").write_text(f"# {competition_name} Data Page\n\n{data_text}\n", encoding="utf-8")
    print(f"Saved OVERVIEW.md and DATA.md to {docs_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Kaggle competition pages.")
    parser.add_argument("competition", help="Kaggle competition slug (e.g., titanic)")
    args = parser.parse_args()
    fetch_competition_pages(args.competition)
