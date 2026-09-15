import re
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os

def parse_log_and_plot(file_path, metric, output_file):
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return

    # Initialize storage per metric to ensure alignment
    # We store x (episode) and y (value)
    data = {
        'tat': {'x': [], 'y': []},
        'nhc': {'x': [], 'y': []},
        'carbon': {'x': [], 'y': []},
        'reward': {'x': [], 'y': []},
        'baseline': {'x': [], 'y': []},
        'advantage': {'x': [], 'y': []}
    }

    # Regex patterns - improved to handle scientific notation and negative numbers
    # matches numbers like 123, -123.45, 1.23e-4
    number_pattern = r"(-?[\d\.]+(?:[eE][-+]?\d+)?)"
    
    patterns = {
        'episode': re.compile(r"Starting episode (\d+)"),
        'tat': re.compile(f"tat of this episode: {number_pattern}"),
        'nhc': re.compile(f"nhc of this episode: {number_pattern}"),
        'carbon': re.compile(f"carbon of this episode: {number_pattern}"),
        'reward': re.compile(f"reward of this episode: {number_pattern}"),
        'baseline': re.compile(f"self\.baseline: {number_pattern}"),
        'advantage': re.compile(f"advantage: {number_pattern}"),
    }

    current_ep = None
    
    print(f"Parsing log file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Check for episode start
            ep_match = patterns['episode'].search(line)
            if ep_match:
                current_ep = int(ep_match.group(1))
                continue

            if current_ep is not None:
                # Check for metrics
                for key, pattern in patterns.items():
                    if key == 'episode': continue
                    
                    match = pattern.search(line)
                    if match:
                        try:
                            val = float(match.group(1))
                            data[key]['x'].append(current_ep)
                            data[key]['y'].append(val)
                        except ValueError:
                            pass

    # Plotting
    if metric not in data:
        print(f"Error: Metric '{metric}' not supported. Available: {list(data.keys())}")
        return

    xs = data[metric]['x']
    ys = data[metric]['y']

    if not xs:
        print(f"No data found for metric '{metric}' in the log file.")
        return

    print(f"Found {len(xs)} data points for {metric}.")

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker='o', linestyle='-', color='b', label=f'{metric}')
    
    # Smoothing if enough data
    if len(ys) > 10:
        # Dynamic window size
        window_size = min(50, max(2, len(ys)//10))
        smoothed = pd.Series(ys).rolling(window=window_size).mean()
        plt.plot(xs, smoothed, color='red', linewidth=2, label=f'Trend (SMA {window_size})')

    plt.title(f'{metric} over Episodes')
    plt.xlabel('Episode')
    plt.ylabel(metric)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    save_path = output_file if output_file else f'{metric}_plot.png'
    plt.savefig(save_path)
    print(f"Plot saved to '{save_path}'")

def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from log file.")
    parser.add_argument('--log_path', type=str, required=True, help='Path to the log file')
    parser.add_argument('--metric', type=str, default='reward', 
                        choices=['reward', 'tat', 'nhc', 'carbon', 'baseline', 'advantage'],
                        help='Metric to visualize')
    parser.add_argument('--output', type=str, default=None, help='Output image file path')

    args = parser.parse_args()
    
    parse_log_and_plot(args.log_path, args.metric, args.output)

if __name__ == "__main__":
    main()