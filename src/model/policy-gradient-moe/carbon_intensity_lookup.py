import pandas as pd

# Load the lookup table into memory once
table_path="/pscratch/sd/s/syfan/carbon/data/caiso_supply_data/hourly_carbon_lookup.csv"
try:
    LOOKUP_TABLE = pd.read_csv(table_path).set_index(['month', 'day', 'hour_only'])['Carbon_Intensity_per_MW'].to_dict()
except FileNotFoundError:
    LOOKUP_TABLE = {}
    print("Warning: hourly_carbon_lookup.csv not found. Please run pre-processing first.")

def get_unit_carbon_intensity(date_input, hour_input):
    """
    Args:
        date_input (str or datetime): Date in 'YYYY-MM-DD' or datetime object
        hour_input (int): Hour of the day (0-23)
        
    Returns:
        float: Average unit carbon intensity (MW) for that time across 2023-2024
    """
    if isinstance(date_input, str):
        dt = pd.to_datetime(date_input)
    else:
        dt = date_input
        
    month = dt.month
    day = dt.day
    
    # Retrieve from the pre-calculated averages
    intensity = LOOKUP_TABLE.get((month, day, hour_input))
    
    if intensity is None:
        return "Data not available for this date/hour."
        
    return intensity

# --- Example Usage ---
# print(get_unit_carbon_intensity("2025-05-20", 14))

# Load the rough lookup table into memory once
rough_table_path = "/pscratch/sd/s/syfan/carbon/data/caiso_supply_data/hourly_avg_carbon.csv"
try:
    ROUGH_LOOKUP_TABLE = pd.read_csv(rough_table_path).set_index('hour_only')['Carbon_Intensity_per_MW'].to_dict()
except FileNotFoundError:
    ROUGH_LOOKUP_TABLE = {}
    print("Warning: hourly_avg_carbon.csv not found.")

def get_carbon_in_rough_table(date_input, hour_input):
    """
    Args:
        date_input (str or datetime): Date in 'YYYY-MM-DD' or datetime object (Ignored for this table)
        hour_input (int): Hour of the day (0-23)
        
    Returns:
        float: Average unit carbon intensity (MW) for that hour, ignoring specific date
    """
    # Retrieve from the pre-calculated hourly averages (date is ignored)
    intensity = ROUGH_LOOKUP_TABLE.get(hour_input)
    
    if intensity is None:
        return "Data not available for this hour."
        
    return intensity