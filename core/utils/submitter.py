import argparse
from pathlib import Path
import subprocess

def submit_to_kaggle(competition, file_path, message):
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", competition,
        "-f", str(file_path),
        "-m", message
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error submitting to Kaggle: {e}")
        print(e.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit to Kaggle")
    parser.add_argument("--competition", "-c", type=str, required=True, help="Competition name")
    parser.add_argument("--file", "-f", type=str, required=True, help="Path to submission file")
    parser.add_argument("--message", "-m", type=str, default="Automated submission", help="Submission message")
    
    args = parser.parse_args()
    submit_to_kaggle(args.competition, args.file, args.message)
