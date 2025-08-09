import pandas as pd
import numpy as np

def create_elite_defense_features():
    
    print(" CREATING ELITE DEFENSE & OPPONENT FEATURES")
    print("=" * 55)
    print("Target: Push RMSE from 4.73 → 4.0-4.5 range")
    
    df = pd.read_csv('opportunity_share_enhanced.csv')
    print(f"Loaded base data: {len(df)} records")
    
    try:
        def_2023 = pd.read_csv('defensive_opponent_stats_2023.csv')
        def_2024 = pd.read_csv('defensive_opponent_stats_2024.csv')
        defense_df = pd.concat([def_2023, def_2024], ignore_index=True)
        print(f" Loaded defensive data: {len(defense_df)} records")
    except:
        print("No defensive data available - creating defaults")
        defense_df = pd.DataFrame()
    
    try:
        team_context = pd.read_csv('team_context_features.csv')
        print(f" Loaded team context: {len(team_context)} records")
    except:
        print("No team context - using simplified approach")
        team_context = pd.DataFrame()
    
    print("\n Creating Position-Specific Defensive Matchups")
    print("=" * 50)
    
    df['def_fp_allowed_to_QB'] = 18.5
    df['def_fp_allowed_to_RB'] = 12.8
    df['def_fp_allowed_to_WR'] = 10.2
    df['def_fp_allowed_to_TE'] = 7.4
    df['def_rank_vs_position'] = 16  
    df['def_EPA_allowed_vs_pos'] = 0.0
    
    if not defense_df.empty:
        defense_df = defense_df.sort_values(['defteam', 'week'])
        defense_df['est_fp_vs_QB'] = (
            defense_df['pass_yards_allowed'] * 0.04 +  # 1 pt per 25 yards
            defense_df['pass_td_allowed'] * 4 +        # 4 pts per TD
            15  # Base QB rushing/bonus
        ).clip(lower=8, upper=35)
        
        defense_df['est_fp_vs_RB'] = (
            defense_df['rush_yards_allowed'] * 0.1 +   # 1 pt per 10 yards  
            defense_df['rush_td_allowed'] * 6 +        # 6 pts per TD
            2.5 # Base receiving
        ).clip(lower=4, upper=25)
        
        defense_df['est_fp_vs_WR'] = (
            defense_df['pass_yards_allowed'] * 0.05 +  # Share of passing yards
            defense_df['pass_td_allowed'] * 2.2        # Share of pass TDs
        ).clip(lower=3, upper=20)
        
        defense_df['est_fp_vs_TE'] = (
            defense_df['pass_yards_allowed'] * 0.025 + # Smaller share
            defense_df['pass_td_allowed'] * 1.4        # Fewer TDs
        ).clip(lower=2, upper=15)
        
        print("Calculating 3-week rolling defensive averages...")
        
        rolling_def_stats = []
        
        for team in defense_df['defteam'].unique():
            team_def = defense_df[defense_df['defteam'] == team].copy()
            
            for col in ['est_fp_vs_QB', 'est_fp_vs_RB', 'est_fp_vs_WR', 'est_fp_vs_TE']:
                team_def[f'{col}_L3'] = team_def[col].rolling(window=3, min_periods=1).mean()
            
            rolling_def_stats.append(team_def)
        
        defense_rolling = pd.concat(rolling_def_stats, ignore_index=True)
        
        
        print(" Mapping defensive matchups to player games...")
        
        for _, player_game in df.iterrows():
            opponent = player_game['opponent_team']
            season = player_game['season']  
            week = player_game['week']
            position = player_game['position']
            idx = player_game.name
            
          
            opp_def = defense_rolling[
                (defense_rolling['defteam'] == opponent) &
                (defense_rolling['week'] < week) &
                (defense_rolling['week'] >= max(1, week - 3))
            ]
            
            if len(opp_def) > 0:
                recent_def = opp_def.iloc[-1]  
                
                if position == 'QB':
                    df.loc[idx, 'def_fp_allowed_to_QB'] = recent_def['est_fp_vs_QB_L3']
                elif position == 'RB':
                    df.loc[idx, 'def_fp_allowed_to_RB'] = recent_def['est_fp_vs_RB_L3']
                elif position == 'WR':
                    df.loc[idx, 'def_fp_allowed_to_WR'] = recent_def['est_fp_vs_WR_L3']
                elif position == 'TE':
                    df.loc[idx, 'def_fp_allowed_to_TE'] = recent_def['est_fp_vs_TE_L3']
        
        print(" Mapped position-specific defensive matchups")
    
    print("\n Creating Defensive Rankings by Position")
    print("=" * 45)
    
    df['def_rank_vs_position'] = 16 
    
    for season in [2023, 2024]:
        for week in range(1, 19):
            week_games = df[(df['season'] == season) & (df['week'] == week)]
            if len(week_games) > 0:
                for pos in ['QB', 'RB', 'WR', 'TE']:
                    pos_col = f'def_fp_allowed_to_{pos}'
                    pos_games = week_games[week_games['position'] == pos]
                    if len(pos_games) > 0:
                        pos_games_sorted = pos_games.sort_values(pos_col)
                        for rank, (idx, _) in enumerate(pos_games_sorted.iterrows(), 1):
                            df.loc[idx, 'def_rank_vs_position'] = rank
    
    print("Created defensive rankings by position")
    
    print("\n Creating Opponent-Aware Game Environment")
    print("=" * 45)
    
    df['opponent_pace'] = df['team_total_plays']
    df['combined_pace'] = df['team_total_plays']
    df['pace_differential'] = 0.0
    df['game_environment_score'] = 1.0
    
    if not team_context.empty:
        team_pace_lookup = team_context.groupby(['team', 'season', 'week'])['total_plays'].first().to_dict()
        print(" Calculating opponent-aware pace metrics...")
        for idx, game in df.iterrows():
            team = game['recent_team']
            opponent = game['opponent_team']
            season = game['season']
            week = game['week']
            team_pace_key = (team, season, week)
            team_pace = team_pace_lookup.get(team_pace_key, 65.0)
            opp_pace = 65.0
            for prev_week in range(max(1, week-3), week):
                key = (opponent, season, prev_week)
                if key in team_pace_lookup:
                    opp_pace = team_pace_lookup[key]
                    break
            combined_pace = (team_pace + opp_pace) / 2
            pace_diff = team_pace - opp_pace
            env_score = (combined_pace / 65.0) * 1.2
            if abs(pace_diff) > 10:
                env_score *= 1.1
            df.loc[idx, 'opponent_pace'] = opp_pace
            df.loc[idx, 'combined_pace'] = combined_pace
            df.loc[idx, 'pace_differential'] = pace_diff
            df.loc[idx, 'game_environment_score'] = min(env_score, 1.5)
        print(" Created opponent-aware pace metrics")
    
    print("\n Creating Elite Matchup Advantage Features")  
    print("=" * 45)
    df['matchup_advantage'] = np.where(
        df['def_rank_vs_position'] >= 24,
        2.0,
        np.where(
            df['def_rank_vs_position'] >= 17,
            1.2,
            np.where(
                df['def_rank_vs_position'] <= 8,
                0.6,
                1.0
            )
        )
    )
    df['matchup_intensity'] = 1.0
    qb_mask = df['position'] == 'QB'
    df.loc[qb_mask, 'matchup_intensity'] = np.where(
        df.loc[qb_mask, 'def_rank_vs_position'] <= 10, 0.7, 1.2
    )
    rb_mask = df['position'] == 'RB'
    df.loc[rb_mask, 'matchup_intensity'] = df.loc[rb_mask, 'matchup_advantage']
    wr_mask = df['position'] == 'WR'
    df.loc[wr_mask, 'matchup_intensity'] = (df.loc[wr_mask, 'matchup_advantage'] + 1.0) / 2
    te_mask = df['position'] == 'TE'
    df.loc[te_mask, 'matchup_intensity'] = (df.loc[te_mask, 'matchup_advantage'] * 0.5) + 0.75
    print(" Created elite matchup advantage features")
    
    print("\n Creating Elite Composite Features")
    print("=" * 35)
    df['opportunity_x_matchup'] = df['opportunity_share'] * df['matchup_advantage']
    df['opportunity_x_pace'] = df['opportunity_share'] * df['game_environment_score']

    if 'volume' not in df.columns:
        df['volume'] = df['opportunity_share']

    df['volume_x_matchup_intensity'] = df['volume'] * df['matchup_intensity']
    print("Created elite composite features")
    
   
    print(f"\n ELITE DEFENSE FEATURE SUMMARY:")
    avg_def_rank = df['def_rank_vs_position'].mean()
    great_matchups = (df['matchup_advantage'] >= 1.5).sum()
    tough_matchups = (df['matchup_advantage'] <= 0.8).sum()
    high_pace_games = (df['combined_pace'] >= 70).sum()
    print(f"   Average defensive rank: {avg_def_rank:.1f}")
    print(f"   Great matchups (≥1.5x): {great_matchups}")
    print(f"   Tough matchups (≤0.8x): {tough_matchups}")
    print(f"   High-pace games (≥70): {high_pace_games}")
    
    print(f"\n Saving elite defense-enhanced dataset...")
    df.to_csv('elite_defense_enhanced.csv', index=False)
    print(f" Saved {len(df)} records with elite defense features")
    
    new_features = [
        'def_fp_allowed_to_QB', 'def_fp_allowed_to_RB', 'def_fp_allowed_to_WR', 'def_fp_allowed_to_TE',
        'def_rank_vs_position', 'opponent_pace', 'combined_pace', 'pace_differential',
        'game_environment_score', 'matchup_advantage', 'matchup_intensity',
        'opportunity_x_matchup', 'opportunity_x_pace', 'volume_x_matchup_intensity'
    ]
    
    print(f"\n ADDED {len(new_features)} ELITE DEFENSE FEATURES:")
    for i, feature in enumerate(new_features, 1):
        if 'def_fp_allowed' in feature or 'def_rank' in feature:
            category = " Defense"
        elif 'pace' in feature:
            category = " Pace"
        elif 'matchup' in feature:
            category = " Matchup"
        elif 'opportunity_x' in feature or 'volume_x' in feature:
            category = " Composite"
        else:
            category = " Context"
        print(f"   {i:2}. {feature:<30} {category}")
    
    return df

if __name__ == "__main__":
    print(" ELITE DEFENSE & OPPONENT-AWARE FEATURES")
    try:
        enhanced_df = create_elite_defense_features()
        print("\n ELITE DEFENSE FEATURES COMPLETE!")
    except Exception as e:
        print(f" Error creating elite defense features: {e}")
        import traceback
        traceback.print_exc()
