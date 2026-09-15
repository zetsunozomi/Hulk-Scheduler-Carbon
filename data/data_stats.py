import sys
import datetime
import numpy as np
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt

def parse_datetime(dt_str):
    return datetime.datetime.fromisoformat(dt_str)

def read_wait_times(job_trace_file):
    from collections import defaultdict
    overall_wait_times = []
    monthly_wait_times_dict = defaultdict(list)

    with open(job_trace_file, 'r') as f:
        header_line = f.readline()  

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                start_str = parts[3]
                submit_str = parts[4]
                start_dt = parse_datetime(start_str)
                submit_dt = parse_datetime(submit_str)
            except (ValueError, IndexError):
                continue

            wait_sec = (start_dt - submit_dt).total_seconds()
            if wait_sec < 0:
                continue

            overall_wait_times.append(wait_sec)

            month_key = start_dt.strftime("%Y-%m")
            monthly_wait_times_dict[month_key].append(wait_sec)

    return overall_wait_times, monthly_wait_times_dict

def compute_distribution_stats(values):
    arr = np.array(values)
    stats = {}
    stats['count'] = len(values)
    stats['mean'] = np.mean(arr)
    stats['median'] = np.median(arr)
    stats['max'] = np.max(arr)
    stats['p10'] = np.percentile(arr, 10)
    stats['p25'] = np.percentile(arr, 25)
    stats['p50'] = np.percentile(arr, 50)
    stats['p75'] = np.percentile(arr, 75)
    stats['p90'] = np.percentile(arr, 90)
    return stats

def main():
    parser = argparse.ArgumentParser(description="Queue Wait Time Distribution with monthly breakdown")
    parser.add_argument("job_trace_file", help="Path to the job trace file")
    parser.add_argument("--log_file", default="distribution_stats.txt", 
                        help="Path to output log file (default: distribution_stats.txt)")
    parser.add_argument("--hist_file", default="queue_wait_hist.png",
                        help="Path to output histogram image (default: queue_wait_hist.png)")
    parser.add_argument("--bins", type=int, default=50, help="Number of bins for histogram (default=50)")
    args = parser.parse_args()

    job_trace_file = args.job_trace_file
    log_file = args.log_file
    hist_file = args.hist_file
    bins = args.bins

    overall_wait_times, monthly_dict = read_wait_times(job_trace_file)
    
    # Convert list to numpy array immediately
    overall_wait_times_hours = np.array(overall_wait_times)
    
    with open(log_file, "w") as f_out:

        def p(msg):
            print(msg)
            f_out.write(msg + "\n")

        if overall_wait_times_hours.size == 0:
            p("No valid wait times found in file.")
            return

        # Convert to hours
        overall_wait_times_hours = overall_wait_times_hours / 3600.0

        overall_stats = compute_distribution_stats(overall_wait_times_hours)
        p("===== Overall Queue Wait Time Distribution =====")
        p(f"Total jobs: {overall_stats['count']}")
        p(f"Mean wait time (hours): {overall_stats['mean']:.2f}")
        p(f"Median wait time (hours): {overall_stats['median']:.2f}")
        p(f"Max wait time (hours): {overall_stats['max']:.2f}")
        p(f"10th percentile (hours): {overall_stats['p10']:.2f}")
        p(f"25th percentile (hours): {overall_stats['p25']:.2f}")
        p(f"50th percentile (hours): {overall_stats['p50']:.2f}")
        p(f"75th percentile (hours): {overall_stats['p75']:.2f}")
        p(f"90th percentile (hours): {overall_stats['p90']:.2f}")
        p("")

        sorted_months = sorted(monthly_dict.keys())
        for m in sorted_months:
            # Convert to hours for monthly stats as well
            values_hours = np.array(monthly_dict[m]) / 3600.0
            stats_m = compute_distribution_stats(values_hours)
            p(f"===== Month {m} =====")
            p(f"Total jobs: {stats_m['count']}")
            p(f"Mean wait time (hours): {stats_m['mean']:.2f}")
            p(f"Median wait time (hours): {stats_m['median']:.2f}")
            p(f"10th percentile (hours): {stats_m['p10']:.2f}")
            p(f"25th percentile (hours): {stats_m['p25']:.2f}")
            p(f"50th percentile (hours): {stats_m['p50']:.2f}")
            p(f"75th percentile (hours): {stats_m['p75']:.2f}")
            p(f"90th percentile (hours): {stats_m['p90']:.2f}")
            p("")

        p(f"Saving histogram to {hist_file}")
        plt.figure(figsize=(8,6))
        plt.hist(overall_wait_times_hours, bins=bins, color='blue', edgecolor='black', alpha=0.7)
        plt.title("Overall Queue Wait Time Distribution")
        plt.xlabel("Wait Time (hours)")
        plt.ylabel("Number of Jobs")
        plt.grid(True, alpha=0.3)
        plt.savefig(hist_file)
        plt.close()

        p("Done.")

if __name__ == "__main__":
    main()

