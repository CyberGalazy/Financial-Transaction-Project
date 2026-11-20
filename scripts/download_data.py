"""Download the Kaggle dataset `computingvictor/transactions-fraud-datasets` into data/raw/.

The dataset is public, so `kagglehub` downloads it anonymously; if Kaggle ever requires auth,
place your API token at ~/.kaggle/kaggle.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud import data  # noqa: E402

if __name__ == "__main__":
    print("Saved to", data.download_dataset())
