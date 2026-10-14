"""Build pre-match team-form features from historical season CSV files."""

from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_FILE = DATA_DIR / "match_features.csv"
HISTORY_LENGTH = 5
INITIAL_ELO = 1500.0
ELO_K = 20.0
HOME_ADVANTAGE_ELO = 65.0
SEASON_ELO_RETENTION = 0.75
REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
ODDS_COLUMNS = ["AvgCH", "AvgCD", "AvgCA"]


def recent_averages(history: deque[dict[str, float]]) -> tuple[float, float, float]:
    """Return average points, goals for, and goals against from prior matches."""
    if not history:
        return float("nan"), float("nan"), float("nan")

    recent = list(history)[-HISTORY_LENGTH:]
    count = len(recent)
    return (
        sum(match["points"] for match in recent) / count,
        sum(match["goals_for"] for match in recent) / count,
        sum(match["goals_against"] for match in recent) / count,
    )


def parse_match_date(value: object) -> pd.Timestamp:
    """Parse Football-Data dates, which use both two- and four-digit years."""
    if pd.isna(value):
        return pd.NaT

    for date_format in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return pd.Timestamp(datetime.strptime(str(value).strip(), date_format))
        except ValueError:
            continue
    return pd.NaT


def main() -> None:
    files = sorted(DATA_DIR.glob("epl_20*.csv"))
    if not files:
        print("No season CSV files found. Run download_data.py first.")
        return

    seasons = []
    for file_path in files:
        wanted_columns = REQUIRED_COLUMNS + ODDS_COLUMNS
        season = pd.read_csv(file_path, usecols=lambda column: column in wanted_columns)
        missing_columns = set(REQUIRED_COLUMNS) - set(season.columns)
        if missing_columns:
            print(f"Skipping {file_path.name}; missing columns: {sorted(missing_columns)}")
            continue
        for column in ODDS_COLUMNS:
            if column not in season.columns:
                season[column] = float("nan")
        season_start = int(file_path.stem.split("_")[1])
        season["Season"] = f"{season_start}/{str(season_start + 1)[-2:]}"
        season["Date"] = season["Date"].apply(parse_match_date)
        season = season.dropna(subset=REQUIRED_COLUMNS)
        seasons.append(season)

    matches = pd.concat(seasons, ignore_index=True)
    matches = matches.sort_values("Date", kind="stable").reset_index(drop=True)

    history: dict[str, deque[dict[str, float]]] = defaultdict(
        lambda: deque(maxlen=HISTORY_LENGTH)
    )
    home_history: dict[str, deque[dict[str, float]]] = defaultdict(
        lambda: deque(maxlen=HISTORY_LENGTH)
    )
    away_history: dict[str, deque[dict[str, float]]] = defaultdict(
        lambda: deque(maxlen=HISTORY_LENGTH)
    )
    games_played: dict[str, int] = defaultdict(int)
    home_games_played: dict[str, int] = defaultdict(int)
    away_games_played: dict[str, int] = defaultdict(int)
    elo_ratings: dict[str, float] = defaultdict(lambda: INITIAL_ELO)
    current_season = None
    feature_rows = []

    # Build all features for a calendar day before adding that day's results.
    # This prevents one same-day match from leaking into another's features.
    for match_date, day_matches in matches.groupby("Date", sort=True):
        day_season = day_matches.iloc[0]["Season"]
        if current_season is not None and day_season != current_season:
            # Retain most of each team's rating across the summer, but allow change.
            for team in list(elo_ratings):
                elo_ratings[team] = INITIAL_ELO + SEASON_ELO_RETENTION * (
                    elo_ratings[team] - INITIAL_ELO
                )
        current_season = day_season

        for row in day_matches.itertuples(index=False):
            home_points, home_gf, home_ga = recent_averages(history[row.HomeTeam])
            away_points, away_gf, away_ga = recent_averages(history[row.AwayTeam])
            home_home_points, home_home_gf, home_home_ga = recent_averages(
                home_history[row.HomeTeam]
            )
            away_away_points, away_away_gf, away_away_ga = recent_averages(
                away_history[row.AwayTeam]
            )
            home_elo = elo_ratings[row.HomeTeam]
            away_elo = elo_ratings[row.AwayTeam]
            home_expected = 1 / (
                1 + 10 ** ((away_elo - home_elo - HOME_ADVANTAGE_ELO) / 400)
            )
            feature_rows.append(
                {
                    "Date": match_date,
                    "Season": row.Season,
                    "HomeTeam": row.HomeTeam,
                    "AwayTeam": row.AwayTeam,
                    "HomePointsLast5": home_points,
                    "HomeGoalsForLast5": home_gf,
                    "HomeGoalsAgainstLast5": home_ga,
                    "HomeGamesBefore": games_played[row.HomeTeam],
                    "AwayPointsLast5": away_points,
                    "AwayGoalsForLast5": away_gf,
                    "AwayGoalsAgainstLast5": away_ga,
                    "AwayGamesBefore": games_played[row.AwayTeam],
                    "HomeHomePointsLast5": home_home_points,
                    "HomeHomeGoalsForLast5": home_home_gf,
                    "HomeHomeGoalsAgainstLast5": home_home_ga,
                    "HomeHomeGamesBefore": home_games_played[row.HomeTeam],
                    "AwayAwayPointsLast5": away_away_points,
                    "AwayAwayGoalsForLast5": away_away_gf,
                    "AwayAwayGoalsAgainstLast5": away_away_ga,
                    "AwayAwayGamesBefore": away_games_played[row.AwayTeam],
                    "HomeEloBefore": home_elo,
                    "AwayEloBefore": away_elo,
                    "EloDifference": home_elo - away_elo,
                    "HomeEloExpected": home_expected,
                    "ClosingOddsHome": row.AvgCH,
                    "ClosingOddsDraw": row.AvgCD,
                    "ClosingOddsAway": row.AvgCA,
                    "FTR": row.FTR,
                }
            )

        for row in day_matches.itertuples(index=False):
            if row.FTR == "H":
                home_points, away_points = 3, 0
            elif row.FTR == "A":
                home_points, away_points = 0, 3
            else:
                home_points = away_points = 1

            history[row.HomeTeam].append(
                {"points": home_points, "goals_for": row.FTHG, "goals_against": row.FTAG}
            )
            history[row.AwayTeam].append(
                {"points": away_points, "goals_for": row.FTAG, "goals_against": row.FTHG}
            )
            home_history[row.HomeTeam].append(
                {"points": home_points, "goals_for": row.FTHG, "goals_against": row.FTAG}
            )
            away_history[row.AwayTeam].append(
                {"points": away_points, "goals_for": row.FTAG, "goals_against": row.FTHG}
            )
            games_played[row.HomeTeam] += 1
            games_played[row.AwayTeam] += 1
            home_games_played[row.HomeTeam] += 1
            away_games_played[row.AwayTeam] += 1

            home_expected = 1 / (
                1
                + 10
                ** (
                    (
                        elo_ratings[row.AwayTeam]
                        - elo_ratings[row.HomeTeam]
                        - HOME_ADVANTAGE_ELO
                    )
                    / 400
                )
            )
            actual_home_score = 1.0 if row.FTR == "H" else 0.5 if row.FTR == "D" else 0.0
            rating_change = ELO_K * (actual_home_score - home_expected)
            elo_ratings[row.HomeTeam] += rating_change
            elo_ratings[row.AwayTeam] -= rating_change

    features = pd.DataFrame(feature_rows)
    features.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(features)} matches with pre-match features to {OUTPUT_FILE}")
    print("First five rows:")
    print(features.head().to_string(index=False))
    print("\nMatches without five prior games for both teams:")
    cold_start = (features["HomeGamesBefore"] < HISTORY_LENGTH) | (
        features["AwayGamesBefore"] < HISTORY_LENGTH
    )
    print(int(cold_start.sum()))


if __name__ == "__main__":
    main()
