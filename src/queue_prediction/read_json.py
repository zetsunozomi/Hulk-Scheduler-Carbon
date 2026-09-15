import json
import os
import argparse

def print_structure(d, indent=0):
    """Recursively prints the structure of a dictionary or list."""
    spacing = "  " * indent
    if isinstance(d, dict):
        print(f"{spacing}Type: dict, Keys: {list(d.keys())}")
        for key, value in list(d.items())[:3]: # Show first 3 keys for structure
            print(f"{spacing}  Key: '{key}'")
            print_structure(value, indent + 2)
        if len(d) > 3:
            print(f"{spacing}  ... (total {len(d)} keys)")
            
    elif isinstance(d, list):
        print(f"{spacing}Type: list, Length: {len(d)}")
        if len(d) > 0:
            print(f"{spacing}  Sample item 0:")
            print_structure(d[0], indent + 2)
            if len(d) > 1:
                print(f"{spacing}  ... (middle items omitted) ...")
                print(f"{spacing}  Sample item {len(d)-1} (Last Item):")
                print_structure(d[-1], indent + 2)
    else:
        print(f"{spacing}Type: {type(d).__name__}, Value (preview): {str(d)[:100]}")

def main():
    parser = argparse.ArgumentParser(description="Read and analyze JSON data file.")
    parser.add_argument("filepath", nargs='?', 
                        default="/pscratch/sd/s/syfan/carbon-minimum-scheduling/src/queue_prediction/json_data/validation_data_new_4/all_data.json",
                        help="Path to the JSON file to read")
    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        print(f"Error: File not found at {args.filepath}")
        return

    print(f"Reading file: {args.filepath}...")
    try:
        with open(args.filepath, 'r') as f:
            data = json.load(f)
        
        print("\n--- Data Summary ---")
        if isinstance(data, list):
            print(f"Total Samples: {len(data)}")
        
        print("\n--- Data Structure ---")
        print_structure(data)

    except json.JSONDecodeError:
        print("Error: Failed to decode JSON. The file might be corrupted or not a valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()