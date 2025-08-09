import pandas as pd

# Load your dataset
df = pd.read_csv("nfl_weekly_player_stats_2023_2024.csv")

# Keep only necessary columns
relevant_cols = [
    'season', 'week', 'recent_team', 'attempts', 'carries',
    'passing_yards', 'rushing_yards', 'receiving_yards'
]
df = df[relevant_cols].copy()

# Replace NaNs with 0 (some players may have missing stats)
df.fillna(0, inplace=True)

# Group by team-week to aggregate total stats
team_week_df = df.groupby(['season', 'week', 'recent_team'], as_index=False).agg({
    'attempts': 'sum',
    'carries': 'sum',
    'passing_yards': 'sum',
    'rushing_yards': 'sum',
    'receiving_yards': 'sum'
})

# Calculate derived features
team_week_df['total_plays'] = team_week_df['attempts'] + team_week_df['carries']
team_week_df['pass_rate'] = team_week_df['attempts'] / team_week_df['total_plays'].replace(0, 1)
team_week_df['total_yards'] = (
    team_week_df['passing_yards'] +
    team_week_df['rushing_yards'] +
    team_week_df['receiving_yards']
)
team_week_df['yards_per_play'] = team_week_df['total_yards'] / team_week_df['total_plays'].replace(0, 1)
team_week_df['pass_yards_per_play'] = team_week_df['passing_yards'] / team_week_df['attempts'].replace(0, 1)
team_week_df['rush_yards_per_play'] = team_week_df['rushing_yards'] / team_week_df['carries'].replace(0, 1)

# Rename for clarity
team_week_df.rename(columns={'recent_team': 'team'}, inplace=True)

# Export to CSV
team_week_df.to_csv("team_context_features.csv", index=False)
print("✅ Team context features saved to 'team_context_features.csv'")





