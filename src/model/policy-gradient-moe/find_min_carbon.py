import pandas as pd

file_path = "/pscratch/sd/s/syfan/carbon/data/caiso_supply_data/hourly_carbon_lookup.csv"

try:
    df = pd.read_csv(file_path)
    min_value = df['Carbon_Intensity_per_MW'].min()
    print(f"Minimum Carbon Intensity: {min_value}")
    
    # 顺便打印一下最小值对应的行，方便查看是哪个月份/小时
    min_row = df.loc[df['Carbon_Intensity_per_MW'].idxmin()]
    print("\nCorresponding Row:")
    print(min_row)
    
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
