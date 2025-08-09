import pandas as pd

print(" Loading datasets...")

weekly_2023 = pd.read_csv("weekly_2023_player_stats.csv")
weekly_2024 = pd.read_csv("weekly_2024_player_stats.csv")

context_2023 = pd.read_csv("weekly_game_context_2023.csv")
context_2024 = pd.read_csv("weekly_game_context_2024.csv")

defense_stats = pd.read_csv("weekly_defense_stats_allowed.csv")


print(" Merging weekly context...")

def merge_context(weekly_df, context_df):
    return pd.merge(
        weekly_df,
        context_df,
        how='left',
        left_on=['season', 'week', 'game_id'],
        right_on=['season', 'week', 'game_id']
    )

merged_2023 = merge_context(weekly_2023, context_2023)
merged_2024 = merge_context(weekly_2024, context_2024)


print(" Merging opponent defense stats...")

def merge_defense(merged_df):
    return pd.merge(
        merged_df,
        defense_stats,
        how='left',
        left_on=['season', 'week', 'opponent_team'],
        right_on=['season', 'week', 'def_team'],
        suffixes=('', '_def')
    ).drop(columns=['def_team'])

merged_2023 = merge_defense(merged_2023)
merged_2024 = merge_defense(merged_2024)


print(" Saving merged datasets...")

merged_2023.to_csv("merged_weekly_2023_features.csv", index=False)
merged_2024.to_csv("merged_weekly_2024_features.csv", index=False)

print("Merged weekly datasets saved!")
