import pandas as pd

df = pd.read_csv("nfl_weekly_player_stats_2023_2024.csv")

df.sort_values(by=['player_id', 'season', 'week'], inplace=True)

games_played = df.groupby('player_id')['week'].nunique().reset_index()
games_played.columns = ['player_id', 'weeks_played']

games_played['injury_risk'] = 1 - (games_played['weeks_played'] / 18)

df = df.merge(games_played[['player_id', 'injury_risk']], on='player_id', how='left')

df.to_csv("injury_risk_added.csv", index=False)
print(" Injury risk added based on games played.")
