import sys
import re
from datetime import datetime
from collections import Counter


def parse_log(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Patterns based on the file content:
    # submit2023-11-26 03:04:00
    # start2023-11-26 03:04:00
    # [result of step=0], ActionIdx=3, tat=6.0, nhc = 192, Done=False
    # tat of this episode: 23.06022270126516
    # nhc of this episode: 438.19379310715186
    # carbon of this episode: 6184.022660883346

    submit_pattern = re.compile(r"submit(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    start_pattern = re.compile(r"start(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    action_pattern = re.compile(r"ActionIdx=(\d+)")
    
    tat_pattern = re.compile(r"tat of this episode:\s*(\S+)")
    nhc_pattern = re.compile(r"nhc of this episode:\s*(\S+)")
    carbon_pattern = re.compile(r"carbon of this episode:\s*(\S+)")

    wait_times = []
    action_indices = []
    
    episode_tat = []
    episode_nhc = []
    episode_carbon = []

    current_submit = None

    for line in lines:
        submit_match = submit_pattern.search(line)
        if submit_match:
            try:
                current_submit = datetime.strptime(submit_match.group(1), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
            continue

        start_match = start_pattern.search(line)
        if start_match and current_submit:
            try:
                start_time = datetime.strptime(start_match.group(1), "%Y-%m-%d %H:%M:%S")
                # Calculate difference in seconds
                diff = (start_time - current_submit).total_seconds()
                wait_times.append(diff)
            except ValueError:
                pass
            current_submit = None # Reset pair
            continue

        action_match = action_pattern.search(line)
        if action_match:
            action_indices.append(int(action_match.group(1)))
            
        tat_match = tat_pattern.search(line)
        if tat_match:
            try:
                episode_tat.append(float(tat_match.group(1)))
            except ValueError:
                pass
            
        nhc_match = nhc_pattern.search(line)
        if nhc_match:
            try:
                episode_nhc.append(float(nhc_match.group(1)))
            except ValueError:
                pass
            
        carbon_match = carbon_pattern.search(line)
        if carbon_match:
            try:
                episode_carbon.append(float(carbon_match.group(1)))
            except ValueError:
                pass

    return wait_times, action_indices, episode_tat, episode_nhc, episode_carbon

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_validation_log.py <log_file>")
        sys.exit(1)

    log_file = sys.argv[1]
    print(f"Reading log file: {log_file}")
    
    try:
        wait_times, action_indices, episode_tat, episode_nhc, episode_carbon = parse_log(log_file)
    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found.")
        sys.exit(1)
    
    print("\n1. Time from Submit to Start (Wait Time):")
    print(f"Total jobs parsed: {len(wait_times)}")
    if wait_times:
        print(f"Average wait time: {sum(wait_times)/len(wait_times):.4f} seconds")
        print(f"Max wait time:     {max(wait_times):.4f} seconds")
        print(f"Min wait time:     {min(wait_times):.4f} seconds")
        
        # Check if all are zero
        if all(w == 0 for w in wait_times):
            print("Note: All wait times are 0.0 seconds.")
        else:
            print(f"Non-zero wait times count: {sum(1 for w in wait_times if w > 0)}")
            
        # Optional: Print raw values if user wants to see them (excerpt)
        print(f"\nFirst 10 wait time values: {wait_times[:10]}")
        print(f"Last 10 wait time values:  {wait_times[-10:]}")
    else:
        print("No wait time data found.")

    print("\n2. ActionIdx Distribution:")
    if action_indices:
        counter = Counter(action_indices)
        total = len(action_indices)
        sorted_indices = sorted(counter.keys())
        for idx in sorted_indices:
            count = counter[idx]
            percent = (count / total) * 100
            print(f"ActionIdx {idx}: {count:5d} ({percent:6.2f}%)")
    else:
        print("No ActionIdx data found.")
        
    print("\n3. Episode Statistics (Means):")
    if episode_tat:
        print(f"Mean TAT (Turnaround Time):    {sum(episode_tat)/len(episode_tat):.4f}")
    else:
        print("Mean TAT: No data found.")
        
    if episode_nhc:
        print(f"Mean NHC (Node Hour Cost):     {sum(episode_nhc)/len(episode_nhc):.4f}")
    else:
        print("Mean NHC: No data found.")
        
    if episode_carbon:
        print(f"Mean Carbon Emission:          {sum(episode_carbon)/len(episode_carbon):.4f}")
    else:
        print("Mean Carbon: No data found.")

if __name__ == "__main__":
    main()