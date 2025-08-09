import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

def fetch_snap_counts(season=2023, weeks=range(1, 19)):
    all_weeks_data = []

    for week in weeks:
        print(f"\n Fetching Week {week} snap counts for {season}...")
        url = f"https://www.fantasypros.com/nfl/reports/snap-counts/?year={season}&week={week}&position=ALL"

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find("table")
        if table is None:
            print(f" No table found for Week {week}, skipping.")
            continue

        df = pd.read_html(str(table))[0]

        print(" Raw Columns:", df.columns.tolist())

        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        possible_snap_cols = [col for col in df.columns if 'snap' in col]
        if not possible_snap_cols:
            print("No snap count column found. Skipping this week.")
            continue

        snap_col = possible_snap_cols[0]  

        df['snap_count'] = df[snap_col]
        df['week'] = week
        df['season'] = season

        df.rename(columns={
            'player': 'player_name',
            'team': 'team',
            'pos': 'position'
        }, inplace=True)

        final_cols = ['season', 'week', 'player_name', 'team', 'position', 'snap_count']
        df = df[[col for col in final_cols if col in df.columns]]

        all_weeks_data.append(df)
        time.sleep(1.5)

    if not all_weeks_data:
        print("No valid data collected.")
        return pd.DataFrame()

    final_df = pd.concat(all_weeks_data, ignore_index=True)
    return final_df


snap_data_2023 = fetch_snap_counts(2023)
snap_data_2024 = fetch_snap_counts(2024, weeks=range(1, 23))


combined = pd.concat([snap_data_2023, snap_data_2024], ignore_index=True)
combined.to_csv("nfl_snap_counts_2023_2024.csv", index=False)

print("\nSaved: nfl_snap_counts_2023_2024.csv")
