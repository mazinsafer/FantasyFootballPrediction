import pandas as pd
import nfl_data_py as nfl

print("Loading 2023 play-by-play and schedule data...")
pbp = nfl.import_pbp_data([2023])
games = nfl.import_schedules([2023]) 
print(" 2023 data loaded.")


print(" Downcasting floats...")
pbp = pbp.copy()
for col in pbp.select_dtypes(include="float"):
    pbp[col] = pd.to_numeric(pbp[col], downcast="float")
games = games.copy()
for col in games.select_dtypes(include="float"):
    games[col] = pd.to_numeric(games[col], downcast="float")

pbp["game_id"] = pbp["game_id"].astype(str)
games["game_id"] = games["game_id"].astype(str)

print(" Aggregating weather data...")
weather_features = pbp.groupby("game_id").agg({
    "temp": "first",
    "wind": "first",
    "roof": "first",
    "surface": "first"
}).reset_index()

print(" Adding spread and final score info...")
games["spread_line"] = pd.to_numeric(games["spread_line"], errors="coerce")
games["score_diff_final"] = games["home_score"] - games["away_score"]
script_features = games[["game_id", "spread_line", "score_diff_final"]]

print("Merging game context features...")
context = pd.merge(weather_features, script_features, on="game_id", how="left")

context.to_csv("game_context_2023_weather.csv", index=False)
print(" Saved: game_context_2023.csv")

