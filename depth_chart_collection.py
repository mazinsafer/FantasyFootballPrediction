from nfl_api_client.endpoints.team_depth_chart import TeamDepthChart
from nfl_api_client.lib.parameters import TeamID
import pandas as pd

def fetch_depth_chart(teams, year):
    all_rows = []
    for team in teams:
        try:
            chart = TeamDepthChart(team_id=team, year=year)
            df = chart.get_dataset("OFFENSE").get_dataframe()
            df['team'] = team.name
            all_rows.append(df[['player_name', 'position_abbreviation', 'rank', 'team']])
        except Exception as e:
            print(f"Error fetching depth chart for {team.name}: {e}")
    return pd.concat(all_rows, ignore_index=True)

teams = list(TeamID)
depth_df = fetch_depth_chart(teams, year=2024)
depth_df.to_csv("auto_role_label.csv", index=False)
print("Depth chart saved to auto_role_label.csv")

