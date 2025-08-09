import pandas as pd
import numpy as np
import nfl_data_py as nfl

print(" Loading 2024 play-by-play and schedule data...")
pbp = nfl.import_pbp_data([2024])
games = nfl.import_schedules([2024])
print(" 2024 data loaded.")

print(" Downcasting floats...")
for col in pbp.select_dtypes(include="float"):
    pbp[col] = pd.to_numeric(pbp[col], downcast="float")
for col in games.select_dtypes(include="float"):
    games[col] = pd.to_numeric(games[col], downcast="float")


pbp["game_id"] = pbp["game_id"].astype(str)
games["game_id"] = games["game_id"].astype(str)


pbp["receiving_touchdown"] = ((pbp["pass_touchdown"] == 1) & (pbp["receiver_player_id"].notnull())).astype(int)


print(" Aggregating opponent defensive stats...")
def_stats = (
    pbp.query("season_type == 'REG'")
    .groupby(["week", "defteam"])
    .agg(
        pass_yards_allowed=("passing_yards", "sum"),
        rush_yards_allowed=("rushing_yards", "sum"),
        pass_td_allowed=("pass_touchdown", "sum"),
        rush_td_allowed=("rush_touchdown", "sum"),
        rec_td_allowed=("receiving_touchdown", "sum"),
        total_yards_allowed=("yards_gained", "sum"),
        fantasy_points_allowed=("fantasy", "sum"),
        plays_against=("play_id", "count")
    )
    .reset_index()
)

print(" Creating Vegas context features...")
vegas = games[["game_id", "home_team", "away_team", "spread_line", "total_line", "week"]].copy()
vegas["home_favorite"] = vegas["spread_line"] < 0
vegas["implied_home_score"] = (vegas["total_line"] + vegas["spread_line"]) / 2
vegas["implied_away_score"] = vegas["total_line"] - vegas["implied_home_score"]


print(" Aggregating weather data...")
weather = pbp.groupby("game_id").agg({
    "temp": "first",
    "wind": "first",
    "roof": "first",
    "surface": "first"
}).reset_index()


print(" Adding game script features...")
games["score_diff_final"] = games["home_score"] - games["away_score"]
script = games[["game_id", "spread_line", "score_diff_final"]]


print(" Merging all game context features...")
context = pd.merge(weather, script, on="game_id", how="left")
context = pd.merge(context, vegas, on="game_id", how="left")


print(" Saving features...")
def_stats.to_csv("defensive_opponent_stats_2024.csv", index=False)
vegas.to_csv("game_context_2024_def.csv", index=False)
context.to_csv("game_context_2024_weather.csv", index=False)

print(" All 2024 features generated and saved.")
