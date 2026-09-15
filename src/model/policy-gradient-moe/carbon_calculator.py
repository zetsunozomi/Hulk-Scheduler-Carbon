from datetime import datetime, timedelta
from carbon_intensity_lookup import get_carbon_in_rough_table

def get_carbon_emission(exec_time, start_time, node, cluster_name):
    """
    Calculate carbon emission based on execution time and start time.
    
    Args:
        exec_time (float): Total execution time of the job in hours.
        start_time (datetime): Start time of the job.
        node (int): Number of nodes used for the job.
        cluster_name (str): Name of the cluster.
    Returns:
        float: Total carbon emission (Proportional Sum of Intensity * Time).
    """
    total_carbon = 0.0
    remaining_time = exec_time
    current_time = start_time
    
    # 1. Handle the first (starting) hour
    # Minutes passed in the current hour
    mins_in_current_hour = current_time.minute + current_time.second/60.0 + current_time.microsecond/3600000.0
    # Time remaining in the current hour slot
    time_available_in_hour = 1.0 - (mins_in_current_hour / 60.0)
    
    # The duration to calculate for this first hour block
    duration_in_this_hour = min(remaining_time, time_available_in_hour)
    
    intensity = get_carbon_in_rough_table(current_time, current_time.hour)
    total_carbon += intensity * duration_in_this_hour
    
    remaining_time -= duration_in_this_hour
    
    # Move current_time to the start of the next hour
    current_time = (current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    
    # 2. Handle middle full hours and the final partial hour
    while remaining_time > 0:
        intensity = get_carbon_in_rough_table(current_time, current_time.hour)
        
        # If remaining time is more than an hour, take the full hour
        # If less, take the remaining fraction (the ending hour)
        duration_in_this_hour = min(remaining_time, 1.0)
        
        total_carbon += intensity * duration_in_this_hour
        
        remaining_time -= duration_in_this_hour
        current_time += timedelta(hours=1)
        
    return total_carbon * node * estimate_power_consumption_per_node(cluster_name)

def estimate_power_consumption_per_node(cluster_name):
    """
    针对特定集群节点估算功耗（单位：kW）。
    
    Lonestar6-A100 节点配置估算：
    - 3x NVIDIA A100 GPU (TDP ~300W each) -> 900W
    - 2x AMD EPYC CPU (TDP ~280W each) -> 560W
    - 内存、主板、外设及损耗 -> ~140W
    - 总计约 1.6 kW，考虑平均负载取 1.5 kW
    """
    cluster_name = cluster_name.lower()
    
    if cluster_name == "lonestar6-a100":
        # 单位：kW (千瓦)
        # 这是一个估算值，代表节点在运行深度学习作业时的平均功耗
        return 1.5
    
    elif "frontera" in cluster_name:
        # 补充：Frontera 主要是 CPU 节点 (Intel Xeon Platinum 8280)
        # 每个节点约 0.5 kW
        return 0.5
    
    else:
        print(f"Unknown cluster name: {cluster_name}")
        quit()

# ---------------------------------------------------------
# 逻辑验证：
# 如果 cluster_name = "lonestar6-a100", node = 1, exec_time = 1.0
# 且 intensity 在该小时为 400 gCO2/kWh
# total_carbon = 400 * 1.0 * 1 * 1.5 = 600 gCO2