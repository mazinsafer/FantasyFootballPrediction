import pandas as pd

df = pd.read_csv("nfl_weekly_player_stats_2023_2024.csv")

df['volume'] = df['carries'].fillna(0) + df['targets'].fillna(0)
seasonal_usage = df.groupby(['player_id', 'season'])['volume'].sum().reset_index()

def assign_role_tier(vol):
    if vol >= 120:
        return "starter"
    elif vol >= 60:
        return "committee"
    else:
        return "backup"

seasonal_usage['role_tier'] = seasonal_usage['volume'].apply(assign_role_tier)

df = df.merge(seasonal_usage[['player_id', 'season', 'role_tier']], on=['player_id', 'season'], how='left')

df.to_csv("role_tier_added.csv", index=False)
print(" Role tiers added based on carries + targets.")
