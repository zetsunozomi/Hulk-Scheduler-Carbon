import pandas as pd
import os
import glob

def generate_lookup_table(processed_folder='processed_data'):
    """
    Reads processed CSV files and creates an average carbon intensity lookup table.
    Groups by Month, Day, and Hour to average across 2023 and 2024.
    """
    # Look for files in the specific input path
    search_pattern = os.path.join(processed_folder, "co2_*.csv")
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        print(f"No files found in {processed_folder}. Check the path.")
        return

    data_list = []

    for file in all_files:
        print(f"reading file: {file}")
        try:
            df = pd.read_csv(file)
            
            # Ensure required column exists
            if 'Carbon_Intensity_per_MW' not in df.columns:
                continue

            # Parse hour from 'Time' column (e.g., "13:45" -> 13)
            df['hour_only'] = df['Time'].apply(lambda x: int(str(x).split(':')[0]))
            
            # Extract date parts from filename: co2_YYYYMMDD.csv
            filename = os.path.basename(file)
            date_str = filename.split('_')[1].split('.')[0]
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            
            df['month'] = month
            df['day'] = day
            
            data_list.append(df[['month', 'day', 'hour_only', 'Carbon_Intensity_per_MW']])
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Combine all data frames
    big_df = pd.concat(data_list)
    
    # Calculate the mean intensity for every specific hour of the calendar year
    # This effectively averages 2023-01-01 12:00 and 2024-01-01 12:00
    lookup = big_df.groupby(['month', 'day', 'hour_only'])['Carbon_Intensity_per_MW'].mean().reset_index()
    
    # Export the lookup table
    lookup.to_csv('hourly_carbon_lookup.csv', index=False)
    print(f"Successfully created 'hourly_carbon_lookup.csv' from {len(all_files)} files.")

if __name__ == "__main__":
    generate_lookup_table()