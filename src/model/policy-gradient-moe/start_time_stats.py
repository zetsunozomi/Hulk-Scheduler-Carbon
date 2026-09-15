import re
import matplotlib.pyplot as plt
from collections import Counter
import sys
import argparse

def analyze_start_times(file_path, output_image_path, start_episode=0, end_episode=None):
    # 用于存储所有提取到的小时
    hours = []
    
    # 正则表达式说明：
    # 匹配 "start" 后面跟着日期，然后捕捉空格后的前两个数字（小时）
    pattern = re.compile(r'start\d{4}-\d{2}-\d{2}\s(\d{2}):')
    # 匹配 "Starting episode X"
    episode_pattern = re.compile(r'Starting episode (\d+)')
    
    current_episode = -1

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 检查是否是 episode 开始行
                ep_match = episode_pattern.search(line)
                if ep_match:
                    current_episode = int(ep_match.group(1))
                
                # 如果当前 episode 小于开始 episode，跳过
                if current_episode < start_episode:
                    continue
                # 如果设定了结束 episode 且当前 episode 超过或等于该值，跳过（前闭后开区间 [start, end)）
                if end_episode is not None and current_episode >= end_episode:
                    continue

                match = pattern.search(line)
                if match:
                    # 将提取的小时字符串转为整数
                    hour = int(match.group(1))
                    hours.append(hour)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'")
        return

    if not hours:
        print("未在日志中找到符合条件的 submit 时间。")
        return

    # 统计每个小时出现的次数
    # 确保 0-23 小时都有对应的位置，即使次数为 0
    counts = Counter(hours)
    hour_range = list(range(24))
    frequency = [counts.get(h, 0) for h in hour_range]

    # 开始绘图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(hour_range, frequency, color='skyblue', edgecolor='navy', alpha=0.7)

    # 在柱状图上方标注具体数值
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height)}', ha='center', va='bottom')

    # 图表装饰
    plt.title(f'Job Start Distribution by Hour', fontsize=15)
    if end_episode:
        plt.title(f'Job Start Distribution by Hour (Ep {start_episode}-{end_episode})', fontsize=15)
    else:
        plt.title(f'Job Start Distribution by Hour (Ep {start_episode}+)', fontsize=15)

    plt.xlabel('Hour of Day (00:00 - 23:00)', fontsize=12)
    plt.ylabel('Number of Start', fontsize=12)
    plt.xticks(hour_range)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 显示图表
    plt.tight_layout()
    plt.savefig(output_image_path)
    print(f"Saved plot to {output_image_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze start times from log file.")
    parser.add_argument("log_file_path", help="Path to the log file")
    parser.add_argument("output_image_path", help="Path to save the output image")
    parser.add_argument("--start_episode", type=int, default=0, help="Start episode (inclusive)")
    parser.add_argument("--end_episode", type=int, default=None, help="End episode (exclusive)")
    
    args = parser.parse_args()

    analyze_start_times(args.log_file_path, args.output_image_path, args.start_episode, args.end_episode)

if __name__ == "__main__":
    main()