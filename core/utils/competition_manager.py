import os
import json
import requests
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from bs4 import BeautifulSoup

class CompetitionManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.api = KaggleApi()
        self.api.authenticate()

    def setup_competition(self, competition_name: str):
        """
        Sets up the directory structure and documentation for a competition.
        """
        print(f"Setting up competition: {competition_name}")
        
        # 1. Create Directories
        comp_dir = self.base_dir / "competitions" / competition_name
        for subdir in ["src", "experiments", "notebooks"]:
            (comp_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Data directories
        (comp_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
        (comp_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)
        
        # 2. Download Data (if not exists)
        if not list((comp_dir / "data" / "raw").glob("*")):
            print("Downloading data...")
            try:
                self.api.competition_download_files(competition_name, path=comp_dir / "data" / "raw", unzip=True)
                # Cleanup zip files
                for zip_file in (comp_dir / "data" / "raw").glob("*.zip"):
                    zip_file.unlink()
            except Exception as e:
                print(f"Error downloading data: {e}")
        else:
            print("Data directory not empty, skipping download.")

        # 3. Fetch Metadata & Generate Docs
        self._generate_docs(competition_name, comp_dir)

    def _generate_docs(self, competition_name: str, comp_dir: Path):
        print("Generating documentation...")
        
        # Basic Info from API
        comps = self.api.competitions_list(search=competition_name)
        # Filter by exact match on ref (which might be a URL or slug depending on version)
        # We'll try to find the one that ends with the slug
        target_comp = None
        for c in comps:
            if c.ref.endswith(competition_name) or c.ref == competition_name:
                target_comp = c
                break
        
        if not target_comp:
            print(f"Could not find competition metadata for {competition_name}")
            return

        # Scrape Description from Web (overview page)
        description_text = "Could not fetch description."
        try:
            overview_url = f"https://www.kaggle.com/competitions/{competition_name}/overview"
            resp = requests.get(overview_url)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    description_text = meta_desc['content']
        except Exception as e:
            print(f"Error scraping overview description: {e}")

        # Scrape Overview page (full content)
        overview_content = "Could not fetch overview content."
        try:
            overview_url = f"https://www.kaggle.com/competitions/{competition_name}/overview"
            resp_ov = requests.get(overview_url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"})
            if resp_ov.status_code == 200:
                soup_ov = BeautifulSoup(resp_ov.text, 'html.parser')
                # Attempt to extract main article content
                article = soup_ov.find('article')
                if article:
                    overview_content = article.get_text('\n', strip=True)
                else:
                    overview_content = soup_ov.get_text('\n', strip=True)
        except Exception as e:
            print(f"Error scraping overview page: {e}")

        # Scrape Data page (full content)
        data_content = "Could not fetch data page content."
        try:
            data_url = f"https://www.kaggle.com/competitions/{competition_name}/data"
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"}
            resp_dp = requests.get(data_url, headers=headers)
            if resp_dp.status_code == 200:
                soup_dp = BeautifulSoup(resp_dp.text, 'html.parser')
                article_dp = soup_dp.find('article')
                if article_dp:
                    data_content = article_dp.get_text('\n', strip=True)
                else:
                    data_content = soup_dp.get_text('\n', strip=True)
        except Exception as e:
            print(f"Error scraping data page: {e}")

        # Write separate markdown files
        docs_dir = comp_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        overview_path = docs_dir / "OVERVIEW.md"
        with open(overview_path, "w", encoding="utf-8") as f_ov:
            f_ov.write(f"# {competition_name} Overview\n\n{overview_content}\n")
        data_path = docs_dir / "DATA.md"
        with open(data_path, "w", encoding="utf-8") as f_dp:
            f_dp.write(f"# {competition_name} Data Page\n\n{data_content}\n")

        # File Info
        files_info = []
        for f in (comp_dir / "data" / "raw").iterdir():
            files_info.append(f"- `{f.name}`: {f.stat().st_size / 1024 / 1024:.2f} MB")

        # Create Markdown Content for README (brief)
        content = f"""# {target_comp.title}

## Overview
- **Competition Name**: {competition_name}
- **Category**: {getattr(target_comp, 'category', 'N/A')}
- **Metric**: {getattr(target_comp, 'evaluationMetric', 'N/A')}
- **Deadline**: {getattr(target_comp, 'deadline', 'N/A')}
- **URL**: https://www.kaggle.com/competitions/{competition_name}
- **Full Overview**: [OVERVIEW.md](OVERVIEW.md)
- **Full Data Page**: [DATA.md](DATA.md)

## Description
{description_text}

## Data Structure
### Files (Raw Data)
{chr(10).join(files_info)}

### Column Descriptions
{self._analyze_data(comp_dir)}
"""
        
        # Save to README.md
        with open(docs_dir / "README.md", "w") as f:
            f.write(content)
        print(f"Documentation saved to {docs_dir / 'README.md'}")

    def _analyze_data(self, comp_dir: Path) -> str:
        import pandas as pd
        import io
        
        report = ""
        data_dir = comp_dir / "data" / "raw"
        
        for file_name in ["train.csv", "test.csv"]:
            file_path = data_dir / file_name
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    report += f"#### {file_name} ({df.shape[0]} rows, {df.shape[1]} columns)\n"
                    
                    # Create a summary table
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    info_str = buffer.getvalue()
                    
                    # Custom markdown table
                    report += "| Column | Type | Missing | Unique | Example |\n"
                    report += "|---|---|---|---|---|\n"
                    for col in df.columns:
                        dtype = str(df[col].dtype)
                        missing = df[col].isnull().sum()
                        unique = df[col].nunique()
                        example = str(df[col].iloc[0])[:50] if not df.empty else ""
                        report += f"| {col} | {dtype} | {missing} | {unique} | {example} |\n"
                    report += "\n"
                except Exception as e:
                    report += f"Error analyzing {file_name}: {e}\n"
        
        return report if report else "(No CSV data found or analysis failed)"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("competition", help="Competition slug (e.g., titanic)")
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parents[2]
    manager = CompetitionManager(base_dir)
    manager.setup_competition(args.competition)
