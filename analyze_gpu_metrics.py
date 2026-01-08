#!/usr/bin/env python3
"""
Analyze GPU metrics from DCGM monitoring logs
Calculates total energy consumption, average power, and other statistics
"""

import sys
import re
from pathlib import Path


def parse_gpu_metrics(log_file):
    """Parse GPU metrics from DCGM log file"""
    metrics = {
        'temperature': [],  # Celsius
        'power': [],        # Watts
        'energy': [],       # millijoules (cumulative)
        'gpu_util': [],     # percent
        'mem_util': [],     # percent
    }

    with open(log_file, 'r') as f:
        for line in f:
            # Skip header lines and errors
            if line.startswith('#') or 'Error' in line or not line.strip():
                continue

            # Parse data lines: GPU 0  temp  power  energy  gpu_util  mem_util
            parts = line.split()
            if len(parts) >= 6 and parts[0] == 'GPU':
                try:
                    temp = float(parts[2]) if parts[2] != '-' else 0
                    power = float(parts[3]) if parts[3] != '-' else 0
                    energy = float(parts[4]) if parts[4] != '-' else 0
                    gpu_util = float(parts[5]) if parts[5] != '-' else 0
                    mem_util = float(parts[6]) if len(parts) > 6 and parts[6] != '-' else 0

                    metrics['temperature'].append(temp)
                    metrics['power'].append(power)
                    metrics['energy'].append(energy)
                    metrics['gpu_util'].append(gpu_util)
                    metrics['mem_util'].append(mem_util)
                except (ValueError, IndexError):
                    continue

    return metrics


def analyze_metrics(metrics):
    """Calculate statistics from parsed metrics"""
    if not metrics['power']:
        return None

    # Energy consumption (difference between first and last reading)
    energy_start_mj = metrics['energy'][0]
    energy_end_mj = metrics['energy'][-1]
    energy_consumed_mj = energy_end_mj - energy_start_mj
    energy_consumed_j = energy_consumed_mj / 1000  # millijoules to joules
    energy_consumed_wh = energy_consumed_j / 3600  # joules to watt-hours
    energy_consumed_kwh = energy_consumed_wh / 1000  # watt-hours to kilowatt-hours

    # Power statistics
    avg_power = sum(metrics['power']) / len(metrics['power'])
    min_power = min(metrics['power'])
    max_power = max(metrics['power'])

    # Temperature statistics
    temps = [t for t in metrics['temperature'] if t > 0]
    avg_temp = sum(temps) / len(temps) if temps else 0
    min_temp = min(temps) if temps else 0
    max_temp = max(temps) if temps else 0
    # Check for potential throttling (typically >83C for consumer GPUs)
    throttle_samples = len([t for t in temps if t >= 83]) if temps else 0
    throttle_pct = (throttle_samples / len(temps) * 100) if temps else 0

    # GPU utilization statistics (filter out zeros for active periods)
    active_gpu_utils = [u for u in metrics['gpu_util'] if u > 0]
    avg_gpu_util = sum(active_gpu_utils) / len(active_gpu_utils) if active_gpu_utils else 0

    active_mem_utils = [u for u in metrics['mem_util'] if u > 0]
    avg_mem_util = sum(active_mem_utils) / len(active_mem_utils) if active_mem_utils else 0

    # Duration (samples at 1 second intervals)
    duration_seconds = len(metrics['power'])

    return {
        'energy_mj': energy_consumed_mj,
        'energy_j': energy_consumed_j,
        'energy_wh': energy_consumed_wh,
        'energy_kwh': energy_consumed_kwh,
        'avg_power_w': avg_power,
        'min_power_w': min_power,
        'max_power_w': max_power,
        'avg_temp_c': avg_temp,
        'min_temp_c': min_temp,
        'max_temp_c': max_temp,
        'throttle_pct': throttle_pct,
        'avg_gpu_util_pct': avg_gpu_util,
        'avg_mem_util_pct': avg_mem_util,
        'duration_seconds': duration_seconds,
        'samples': len(metrics['power'])
    }


def print_analysis(stats, log_file):
    """Print formatted analysis results"""
    print(f"\n{'='*60}")
    print(f"GPU Metrics Analysis: {Path(log_file).name}")
    print(f"{'='*60}\n")

    print(f"🌡️  Temperature:")
    print(f"   Average Temp: {stats['avg_temp_c']:.1f}°C")
    print(f"   Min Temp:     {stats['min_temp_c']:.1f}°C")
    print(f"   Max Temp:     {stats['max_temp_c']:.1f}°C")
    if stats['throttle_pct'] > 0:
        print(f"   ⚠️  THROTTLING: {stats['throttle_pct']:.1f}% of samples at ≥83°C")
    else:
        print(f"   ✓ No thermal throttling detected")

    print(f"\n⚡ Energy Consumption:")
    print(f"   Total Energy: {stats['energy_j']:.2f} J ({stats['energy_wh']:.4f} Wh)")
    print(f"   Total Energy: {stats['energy_mj']:.0f} mJ ({stats['energy_kwh']:.6f} kWh)")

    print(f"\n🔋 Power Statistics:")
    print(f"   Average Power: {stats['avg_power_w']:.2f} W")
    print(f"   Min Power:     {stats['min_power_w']:.2f} W")
    print(f"   Max Power:     {stats['max_power_w']:.2f} W")

    print(f"\n📊 GPU Utilization (during active periods):")
    print(f"   Average GPU Util:  {stats['avg_gpu_util_pct']:.1f}%")
    print(f"   Average Mem Util:  {stats['avg_mem_util_pct']:.1f}%")

    print(f"\n⏱️  Duration:")
    print(f"   Monitoring Time: {stats['duration_seconds']} seconds ({stats['duration_seconds']/60:.1f} minutes)")
    print(f"   Samples:         {stats['samples']}")

    print(f"\n{'='*60}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_gpu_metrics.py <gpu_metrics.log>")
        print("\nExample:")
        print("  python3 analyze_gpu_metrics.py results/20251125_014635/gpu_metrics.log")
        sys.exit(1)

    log_file = sys.argv[1]

    if not Path(log_file).exists():
        print(f"Error: File not found: {log_file}")
        sys.exit(1)

    # Parse and analyze
    metrics = parse_gpu_metrics(log_file)

    if not metrics['power']:
        print(f"Error: No valid metrics found in {log_file}")
        sys.exit(1)

    stats = analyze_metrics(metrics)

    if stats:
        print_analysis(stats, log_file)
    else:
        print(f"Error: Could not analyze metrics from {log_file}")
        sys.exit(1)


if __name__ == '__main__':
    main()
