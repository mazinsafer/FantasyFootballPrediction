import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from nfl_data_py import import_pbp_data, import_weekly_data
    print(" nfl_data_py imported successfully")
except ImportError:
    print(" nfl_data_py not found. Install with: pip install nfl_data_py")
    exit()

def extract_weekly_player_stats(years=[2023, 2024]):
    print(f" Loading play-by-play data for {years}...")
    
    try:
        pbp = import_pbp_data(years, downcast=True, cache=False)
        print(f" Loaded {len(pbp)} plays")
        
        pbp = pbp[pbp['season_type'] == 'REG'].copy()
        print(f" Filtered to {len(pbp)} regular season plays")
        

        fantasy_cols = [col for col in pbp.columns if 'fantasy' in col.lower()]
        print(f" Available fantasy columns: {fantasy_cols}")
        
        if 'fantasy_player_name' in pbp.columns:
            player_name_col = 'fantasy_player_name'
        elif 'player_name' in pbp.columns:
            player_name_col = 'player_name'
        else:
            player_name_col = 'name'  
            
        if 'fantasy_player_id' in pbp.columns:
            player_id_col = 'fantasy_player_id'
        elif 'player_id' in pbp.columns:
            player_id_col = 'player_id'
        else:
            player_id_col = 'id' 
        
        print(f" Using player columns: {player_name_col}, {player_id_col}")
      
        valid_mask = (
            pbp['week'].notnull() & 
            pbp[player_name_col].notnull() & 
            pbp[player_name_col] != ''
        )
        pbp = pbp[valid_mask].copy()
        print(f" Filtered to {len(pbp)} plays with valid player-week data")
        
        group_cols = [player_name_col, 'season', 'week', 'posteam']
        if 'position' in pbp.columns:
            group_cols.append('position')
        
        print(" Aggregating weekly player stats...")
        
        agg_dict = {
            'game_id': pd.Series.nunique, 
        }
        
        if 'fantasy_points' in pbp.columns:
            agg_dict['fantasy_points'] = 'sum'
        elif 'fantasy' in pbp.columns:
            agg_dict['fantasy_points'] = ('fantasy', 'sum')
        
        if 'passing_yards' in pbp.columns:
            agg_dict.update({
                'passing_yards': 'sum',
                'passing_tds': ('pass_touchdown', 'sum'),
                'passing_ints': ('interception', 'sum'),
                'pass_attempts': ('pass_attempt', 'sum'),
                'completions': ('complete_pass', 'sum')
            })
          
        if 'rushing_yards' in pbp.columns:
            agg_dict.update({
                'rushing_yards': 'sum',
                'rushing_tds': ('rush_touchdown', 'sum'),
                'rush_attempts': ('rush_attempt', 'sum')
            })
        
        if 'receiving_yards' in pbp.columns:
            agg_dict.update({
                'receiving_yards': 'sum',
                'receiving_tds': ('pass_touchdown', lambda x: x[x.index.isin(pbp[pbp['reception']==1].index)].sum()),
                'receptions': ('reception', 'sum'),
                'targets': ('pass_attempt', lambda x: x[x.index.isin(pbp[pbp['receiver_player_name'].notnull()].index)].sum())
            })
        
        if 'fumble_lost' in pbp.columns:
            agg_dict['fumbles_lost'] = ('fumble_lost', 'sum')
        
    
        weekly_stats = pbp.groupby(group_cols).agg(agg_dict).reset_index()
        
        
        if isinstance(weekly_stats.columns, pd.MultiIndex):
            weekly_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in weekly_stats.columns]
        

        column_renames = {
            'game_id': 'games_played',
            f'{player_name_col}': 'player_name'
        }
        weekly_stats.rename(columns=column_renames, inplace=True)
        
        stat_columns = [col for col in weekly_stats.columns if col not in ['player_name', 'season', 'week', 'posteam', 'position']]
        weekly_stats[stat_columns] = weekly_stats[stat_columns].fillna(0)
        
        if 'receptions' in weekly_stats.columns and 'fantasy_points' in weekly_stats.columns:
            weekly_stats['ppr_points'] = weekly_stats['fantasy_points'] + weekly_stats['receptions']
        
        print(f"Aggregated to {len(weekly_stats)} player-week records")
        print(f"Columns: {list(weekly_stats.columns)}")
        
        return weekly_stats
        
    except Exception as e:
        print(f" Error extracting weekly stats: {e}")
        return None

def try_alternative_method(years=[2023, 2024]):
  
    print(f" Trying alternative method with weekly data...")
    
    try:
        weekly_data = import_weekly_data(years, downcast=True)
        print(f" Loaded weekly data: {len(weekly_data)} records")
        print(f" Columns: {list(weekly_data.columns)}")

        if 'position' in weekly_data.columns:
            main_positions = ['QB', 'RB', 'WR', 'TE']
            weekly_data = weekly_data[weekly_data['position'].isin(main_positions)]
        
        return weekly_data
        
    except Exception as e:
        print(f" Alternative method failed: {e}")
        return None

def main():
    print(" NFL Weekly Player Stats Extractor")
    print("=" * 50)
    
    weekly_stats = extract_weekly_player_stats([2023, 2024])
    
    if weekly_stats is None or len(weekly_stats) == 0:
        print("\n Main method failed, trying alternative...")
        weekly_stats = try_alternative_method([2023, 2024])
    
    if weekly_stats is not None and len(weekly_stats) > 0:

        filename = "nfl_weekly_player_stats_2023_2024.csv"
        weekly_stats.to_csv(filename, index=False)
        print(f"\nSaved {len(weekly_stats)} records to: {filename}")
        
        print(f"\n DATA SUMMARY:")
        print(f"  Total player-week records: {len(weekly_stats)}")
        
        if 'season' in weekly_stats.columns:
            print(f"  Records by season:")
            print(weekly_stats['season'].value_counts().sort_index())
        
        if 'position' in weekly_stats.columns:
            print(f"  Records by position:")
            print(weekly_stats['position'].value_counts())
        
        if 'week' in weekly_stats.columns:
            print(f"  Weeks covered: {weekly_stats['week'].min()} - {weekly_stats['week'].max()}")
        
        
        print(f"\n SAMPLE DATA:")
        print(weekly_stats.head(3).to_string())
        
        print(f"\n SUCCESS!")
        
    else:
        print("\n Both methods failed.")

if __name__ == "__main__":
    main()


