#!/usr/bin/env python3
"""
分析Wait行为的时间依赖性
"""

import re
import sys
import numpy as np
from collections import defaultdict

def analyze_wait_behavior(log_file):
    """分析log文件中的wait行为"""
    
    # 收集数据
    wait_by_hour = defaultdict(list)  # hour -> [wait_prob, ...]
    wait_decisions = []  # [(tod, carbon, delta, did_wait), ...]
    
    current_tod = None
    current_carbon = None
    current_delta = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # 提取wait action
            if 'Skip 1 hour at ToD' in line:
                match = re.search(r'ToD ([\d.]+).*Carbon: ([\d.]+) -> ([\d.]+) \(Δ=([-\d.]+)\)', line)
                if match:
                    tod = float(match.group(1))
                    current = float(match.group(2))
                    future = float(match.group(3))
                    delta = float(match.group(4))
                    
                    hour = int(tod)
                    wait_decisions.append((tod, current, delta, True))
                    
            # 提取probability distribution
            elif 'probablity distribution' in line:
                match = re.search(r'tensor\(\[\[([\d., ]+)\]\]', line)
                if match:
                    probs = [float(x) for x in match.group(1).split(',')]
                    if len(probs) == 5:
                        wait_prob = probs[4]  # Last element is wait probability
                        
                        # 找到对应的ToD
                        # 需要从前面的行中提取
                        
    # 分析
    print("=" * 60)
    print("Wait Behavior Analysis")
    print("=" * 60)
    
    # 按时段统计
    night_hours = list(range(20, 24)) + list(range(0, 6))
    day_hours = list(range(10, 17))
    
    night_waits = [d for d in wait_decisions if int(d[0]) in night_hours]
    day_waits = [d for d in wait_decisions if int(d[0]) in day_hours]
    
    print(f"\n📊 Wait Statistics:")
    print(f"  Total wait decisions: {len(wait_decisions)}")
    print(f"  Night waits (20-6h): {len(night_waits)}")
    print(f"  Day waits (10-16h): {len(day_waits)}")
    
    if day_waits:
        ratio = len(night_waits) / len(day_waits)
        print(f"  Night/Day ratio: {ratio:.2f}x")
    
    # Carbon delta分析
    print(f"\n🌡️  Carbon Delta Analysis:")
    negative_delta_waits = [d for d in wait_decisions if d[2] < 0]
    positive_delta_waits = [d for d in wait_decisions if d[2] > 0]
    
    print(f"  Waits with Δ < 0 (future lower): {len(negative_delta_waits)}")
    print(f"  Waits with Δ > 0 (future higher): {len(positive_delta_waits)}")
    
    if negative_delta_waits:
        pct = len(negative_delta_waits) / len(wait_decisions) * 100
        print(f"  Percentage waiting when Δ < 0: {pct:.1f}%")
    
    # 平均carbon delta
    if wait_decisions:
        avg_delta = np.mean([d[2] for d in wait_decisions])
        print(f"  Average Δ when waiting: {avg_delta:.1f} g/kWh")
    
    print("\n" + "=" * 60)
    
    # 期望行为检查
    print("\n✅ Expected Behavior Check:")
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: Night waits > Day waits
    total_checks += 1
    if len(night_waits) > len(day_waits):
        print("  ✅ Night waits > Day waits")
        checks_passed += 1
    else:
        print("  ❌ Night waits <= Day waits (PROBLEM!)")
    
    # Check 2: Most waits have negative delta
    total_checks += 1
    if negative_delta_waits and len(negative_delta_waits) > len(positive_delta_waits):
        print("  ✅ Most waits occur when Δ < 0")
        checks_passed += 1
    else:
        print("  ❌ Waits not correlated with negative Δ (PROBLEM!)")
    
    # Check 3: Average delta is negative
    total_checks += 1
    if wait_decisions and avg_delta < 0:
        print("  ✅ Average Δ when waiting is negative")
        checks_passed += 1
    else:
        print("  ❌ Average Δ when waiting is not negative (PROBLEM!)")
    
    print(f"\n📈 Score: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("🎉 Model is learning time-dependent waiting!")
    elif checks_passed >= total_checks / 2:
        print("⚠️  Model is partially learning, needs more training")
    else:
        print("❌ Model is NOT learning time-dependent waiting")
        print("   → Check features, learning rate, or increase training time")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_wait.py <log_file>")
        sys.exit(1)
    
    analyze_wait_behavior(sys.argv[1])
