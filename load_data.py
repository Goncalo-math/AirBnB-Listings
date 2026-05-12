import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw")


def load_all():
    airbnb_listings = pd.read_csv(DATA_PATH / "Listings.csv",encoding="latin-1", low_memory=False)

    return {
        "airbnb_listings": airbnb_listings
    }


if __name__ == "__main__":
    dfs = load_all()
    for name, df in dfs.items():
        print(f"{name}: {df.shape[0]} rows × {df.shape[1]} cols")
