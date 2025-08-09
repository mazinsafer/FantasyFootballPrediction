import pandas as pd
import numpy as np
from nfl_data_py import import_pbp_data

print("Loading data...")
pbp = import_pbp_data([2023])
print("2023 done.")

for col in pbp.select_dtypes(include=["float"]):
    pbp[col] = pd.to_numeric(pbp[col], downcast="float")
print("Downcasting floats.")

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

print("Adding game context features...")
games = pbp.drop_duplicates("game_id")[
    ["game_id", "home_team", "away_team", "spread_line", "total_line", "week"]
].copy()

games["home_favorite"] = games["spread_line"] < 0
games["implied_home_score"] = (games["total_line"] + games["spread_line"]) / 2
games["implied_away_score"] = games["total_line"] - games["implied_home_score"]

print("Saving features...")
def_stats.to_csv("defensive_opponent_stats_2023.csv", index=False)
games.to_csv("game_context_2023_def.csv", index=False)

print("All features generated and saved.")
