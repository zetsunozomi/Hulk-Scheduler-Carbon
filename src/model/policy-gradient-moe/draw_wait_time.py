import re
import matplotlib.pyplot as plt
import argparse

def plot_tod_distribution(log_file_path):
    tod_values = []
    
    # 匹配模式：寻找 ActionIdx=4 且包含 ToD 的行，提取其后的浮点数/整数
    pattern = r"ActionIdx=4, Skip 1 hour at ToD ([\d.]+)"
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    # 将提取到的字符串转为浮点数并存入 list
                    tod_values.append(float(match.group(1)))
    except FileNotFoundError:
        print(f"错误：找不到文件 {log_file_path}")
        return

    if not tod_values:
        print("未在 log 中找到匹配的行。")
        return

    print(f"共提取到 {len(tod_values)} 条记录。")

    # 绘图：分布直方图
    plt.hist(tod_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    
    # 使用 LaTeX 格式设置标签（如需）
    plt.xlabel('$ToD$ (Time of Day)')
    plt.ylabel('Frequency')
    plt.title('Distribution of ToD for ActionIdx=4')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 保存图片
    output_filename = 'tod_distribution.png'
    plt.savefig(output_filename)
    print(f"直方图已保存至: {output_filename}")

parser = argparse.ArgumentParser(description="Plot training metrics from log file.")
parser.add_argument('--log_path', type=str, required=True, help='Path to the log file')

args = parser.parse_args()

plot_tod_distribution(args.log_path)