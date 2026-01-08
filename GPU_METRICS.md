# GPU Telemetry Collection

## Overview

The benchmark automatically collects comprehensive GPU metrics using **NVIDIA DCGM (Data Center GPU Manager)** during benchmark execution.

## Metrics Collected

The following DCGM field IDs are monitored at 1-second intervals:

### Power & Thermal
- **155**: GPU Power Usage (Watts)
- **156**: GPU Temperature (°C)

### Utilization
- **203**: GPU Utilization (%)
- **204**: Memory Utilization (%)

### Performance Counters
- **1001**: SM (Streaming Multiprocessor) Active
- **1002**: SM Occupancy
- **1003**: Tensor Core Active
- **1004**: DRAM Active
- **1005**: FP64 Active
- **1006**: FP32 Active
- **1007**: FP16 Active
- **1008**: PCIe TX Throughput
- **1009**: PCIe RX Throughput

## Output Format

GPU metrics are saved to `/tmp/gpu_metrics.log` on the instance and included in the S3 upload as `gpu_metrics.log.gz`.

### Sample Output

```
# Entity    PWRUSG  GPUTMP  GRUTIL  MMUSG  ...
# GPU 0     250.5   72      95      80     ...
# GPU 0     252.1   73      96      81     ...
```

## Usage in Results

The GPU metrics are:
1. Collected continuously during the benchmark
2. Downloaded to local results directory
3. Compressed and uploaded to S3
4. Available for post-processing and analysis

## Analysis

To analyze GPU metrics:

```python
import pandas as pd

# Load GPU metrics
df = pd.read_csv('gpu_metrics.log', sep=r'\s+', comment='#')

# Calculate statistics
print(f"Average Power: {df['PWRUSG'].mean():.2f}W")
print(f"Peak Temperature: {df['GPUTMP'].max()}°C")
print(f"Average GPU Util: {df['GRUTIL'].mean():.1f}%")
```

## Configuration

GPU telemetry is enabled by default. To disable, modify the `_run_benchmark()` method in `benchmark_runner.py`.

## Requirements

- NVIDIA GPU with DCGM support
- CUDA 12.0+ recommended
- DCGM installed on the instance (automatically installed during setup)

## Troubleshooting

If GPU metrics are not collected:
1. Check DCGM installation: `dcgmi discovery -l`
2. Verify DCGM service: `systemctl status nvidia-dcgm`
3. Test manually: `dcgmi dmon -e 155,156 -c 5`

## References

- [NVIDIA DCGM Documentation](https://docs.nvidia.com/datacenter/dcgm/latest/)
- [DCGM Field IDs](https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/group__dcgmFieldIdentifiers.html)
