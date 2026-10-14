"""Download or refresh the four retained Premier League seasons."""

from pathlib import Path
from urllib.request import urlopen


DATA_DIR = Path(__file__).parent / "data"
FIRST_SEASON_START = 2022
LAST_SEASON_START = 2025


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    for start_year in range(FIRST_SEASON_START, LAST_SEASON_START + 1):
        end_year = start_year + 1
        season_code = f"{start_year % 100:02d}{end_year % 100:02d}"
        file_path = DATA_DIR / f"epl_{start_year}_{end_year}.csv"

        url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
        try:
            with urlopen(url, timeout=30) as response:
                current_bytes = response.read()
            if file_path.exists() and file_path.read_bytes() == current_bytes:
                print(f"Already current: {file_path.name}")
            else:
                file_path.write_bytes(current_bytes)
                print(f"Downloaded or refreshed: {file_path.name}")
        except Exception as error:
            print(f"Could not download {start_year}/{end_year}: {error}")


if __name__ == "__main__":
    main()
