"""Load one Football-Data.co.uk season CSV and inspect its columns."""

from pathlib import Path

import pandas as pd


DATA_FILE = Path(__file__).parent / "data" / "epl_2025_2026.csv"


def main() -> None:
    if not DATA_FILE.exists():
        print(f"Can't find the CSV yet: {DATA_FILE}")
        print("Download a Premier League season CSV and save it at that path.")
        return

    matches = pd.read_csv(DATA_FILE)

    key_columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    missing_columns = [column for column in key_columns if column not in matches.columns]
    if missing_columns:
        print(f"This CSV is missing expected columns: {missing_columns}")
        print("Available columns:")
        print(matches.columns.tolist())
        return

    print("First five matches (showing only the columns we need first):")
    print(matches[key_columns].head().to_string(index=False))
    print(f"\nNumber of matches: {len(matches)}")
    print("\nResult codes (FTR): H = home win, D = draw, A = away win")
    print(matches["FTR"].value_counts().sort_index().to_string())
    print("\nMissing values in these columns:")
    print(matches[key_columns].isna().sum().to_string())


if __name__ == "__main__":
    main()
