import pandas as pd
import os
from datetime import datetime, timedelta

# IPCC Fifth Assessment Report (AR5), Working Group III
# Values represent median lifecycle GHG emissions
EMISSION_FACTORS = {
    'Solar': 48,          # Utility-scale Solar PV
    'Wind': 12,           # Onshore Wind (Offshore is ~12-15)
    'Geothermal': 38,
    'Biomass': 230,       # Dedicated biomass (range varies widely)
    'Biogas': 230,        # Often grouped with biomass
    'Small Hydro': 24,    # Usually similar to large hydro
    'Coal': 820,          # Supercritical coal (Global average is ~1000)
    'Nuclear': 12,
    'Natural Gas': 490,   # Combined Cycle (NGCC)
    'Large Hydro': 24,
    'Batteries': 0,       # Lifecycle usually ignored in grid intensity to avoid double counting
    'Imports': 450,       # Global average grid intensity (approximate)
    'Other': 450          # Placeholder for unspecified grid mix
}

input_folder = 'caiso_supply_data' # Current directory
output_folder = 'processed_data'

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def calculate_intensity(row):
    total_emissions = 0.0
    total_output = 0.0
    
    for source, factor in EMISSION_FACTORS.items():
        if source in row:
            val = row[source]
            # Only consider positive output (Generation)
            # Negative values (Charging/Consumption) are treated as 0
            pos_val = max(0, val)
            
            total_emissions += pos_val * factor
            total_output += pos_val
            
    # Calculate Intensity: Total Emissions / Total Output
    # Handle division by zero if the grid has no output at that moment
    if total_output > 0:
        return total_emissions / total_output
    else:
        return 0.0

def process_co2_files(start_date, end_date):
    current_date = start_date
    
    while current_date <= end_date:
        file_name = f"co2_{current_date.strftime('%Y%m%d')}.csv"
        file_path = os.path.join(input_folder, file_name)
        
        if os.path.exists(file_path):
            print(f"Processing: {file_name}")
            df = pd.read_csv(file_path)
            
            # Add the new column: Carbon Intensity (Emission per unit MW)
            df['Carbon_Intensity_per_MW'] = df.apply(calculate_intensity, axis=1)
            
            # Save to the output folder
            save_path = os.path.join(output_folder, file_name)
            df.to_csv(save_path, index=False) 
        else:
            print(f"Skipping: {file_name} (File not found)")
            
        current_date += timedelta(days=1)

if __name__ == "__main__":
    # Define date range
    start = datetime(2023, 1, 1)
    end = datetime(2024, 12, 31)
    
    process_co2_files(start, end)
    print("Task completed successfully.")