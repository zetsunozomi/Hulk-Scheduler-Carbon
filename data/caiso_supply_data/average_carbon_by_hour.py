import pandas as pd

input_path = "/pscratch/sd/s/syfan/carbon/data/caiso_supply_data/hourly_carbon_lookup.csv"
output_path = "/pscratch/sd/s/syfan/carbon/data/caiso_supply_data/hourly_avg_carbon.csv"

try:
    # Read the existing lookup table
    df = pd.read_csv(input_path)
    
    # Group by 'hour_only' and calculate the mean of 'Carbon_Intensity_per_MW'
    # logical verification: The csv has columns [month, day, hour_only, Carbon_Intensity_per_MW]
    hourly_avg = df.groupby('hour_only')['Carbon_Intensity_per_MW'].mean().reset_index()
    
    # Rename columns for clarity if needed, though requests said "keep hour as entry"
    # hourly_avg.columns = ['hour', 'avg_carbon_intensity'] 
    
    # Save the result
    hourly_avg.to_csv(output_path, index=False)
    
    print(f"Successfully calculated average carbon intensity by hour.")
    print(f"Saved to: {output_path}")
    print("\nResult Preview:")
    print(hourly_avg)

except FileNotFoundError:
    print(f"Error: Input file not found at {input_path}")
except KeyError as e:
    print(f"Error: Column not found in CSV - {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
