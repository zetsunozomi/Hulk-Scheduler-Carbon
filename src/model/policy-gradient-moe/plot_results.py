import sys
import matplotlib.pyplot as plt
import re

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <log_file_path>")
        sys.exit(1)

    log_path = sys.argv[1]
    
    mean_tat = []
    mean_carbon = []
    weights = []

    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                if "Analyzing carbon weight" in line:
                    match = re.search(r"Analyzing carbon weight\s+([0-9\.]+)", line)
                    if match:
                        weights.append(match.group(1))

                if "Mean TAT (Turnaround Time):" in line:
                    # Extract the number
                    # assuming format: "Mean TAT (Turnaround Time):    26.5469"
                    parts = line.split(':')
                    if len(parts) > 1:
                        try:
                            value = float(parts[1].strip())
                            mean_tat.append(value)
                        except ValueError:
                            print(f"Could not parse TAT value in line: {line}")
                
                elif "Mean Carbon Emission:" in line:
                    # assuming format: "Mean Carbon Emission:          83567.3750"
                    parts = line.split(':')
                    if len(parts) > 1:
                        try:
                            value = float(parts[1].strip())
                            mean_carbon.append(value)
                        except ValueError:
                            print(f"Could not parse Carbon value in line: {line}")
    except FileNotFoundError:
        print(f"Error: File not found at {log_path}")
        sys.exit(1)

    if not mean_tat or not mean_carbon:
        print("No data found to plot.")
        sys.exit(0)

    # Synchronize lengths
    min_len = min(len(mean_tat), len(mean_carbon))
    if len(weights) > min_len:
        weights = weights[:min_len]
    elif len(weights) < min_len:
        # If weights are missing, pad with empty strings or just slice data
        min_len = len(weights) 
    
    mean_tat = mean_tat[:min_len]
    mean_carbon = mean_carbon[:min_len]
    weights = weights[:min_len]

    # Sort data by TAT for the line plot to make sense (optional, but usually good for "connecting lines", 
    # unless the order represents something else like time or increasing carbon weight. 
    # The user asked to "connect lines", usually implying a trajectory or relationship. 
    # Given the file structure implies increasing carbon weight, the order in file is 0.0 -> 1.0.
    # Connecting them in order of appearance makes sense to show the trend as weight increases.
    
    plt.figure(figsize=(10, 6))
    plt.plot(mean_carbon, mean_tat, marker='o', linestyle='-', color='b')
    
    # Annotate weights
    for i, txt in enumerate(weights):
        plt.annotate(txt, (mean_carbon[i], mean_tat[i]), 
                     xytext=(5, 5), textcoords='offset points')

    plt.xlabel('Mean Carbon Emission')
    plt.ylabel('Mean TAT (Turnaround Time)')
    plt.title('Mean Carbon Emission vs Mean TAT')
    plt.grid(True)
    
    output_image = "results.png"
    plt.savefig(output_image)
    print(f"Plot saved to {output_image}")

if __name__ == "__main__":
    main()
