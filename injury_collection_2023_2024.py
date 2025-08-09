import pandas as pd

# ✅ Updated URL from nflverse GitHub
url = "https://raw.githubusercontent.com/nflverse/nflverse-data/master/releases/injuries/injuries.csv"

try:
    # Load the data
    injuries = pd.read_csv(url)

    # Filter for 2023 and 2024
    injuries = injuries[injuries["season"].isin([2023, 2024])]

    # Keep relevant columns
    cols_to_keep = [
        "season", "week", "player_name", "position", "team", 
        "opponent", "status", "practice_status", "game_status"
    ]
    injuries = injuries[cols_to_keep]

    # Fill missing values for practice/game status
    injuries["practice_status"] = injuries["practice_status"].fillna("Unknown")
    injuries["game_status"] = injuries["game_status"].fillna("Questionable")

    # Sort and save
    injuries.sort_values(by=["season", "week", "player_name"], inplace=True)
    injuries.to_csv("injury_report_2023_2024.csv", index=False)

    print("✅ Saved injury_report_2023_2024.csv with", len(injuries), "rows")

except Exception as e:
    print("❌ Failed to download or process injury data:", e)

