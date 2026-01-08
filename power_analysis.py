"""
Power Analysis Module for LLM Benchmarks
Analyzes GPU metrics and calculates energy consumption.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging


@dataclass
class PowerMetrics:
    """Power consumption metrics for a benchmark run"""
    # Basic info
    benchmark_name: str
    model: str
    gpu_name: str

    # Timing
    benchmark_duration_sec: float
    measurement_duration_sec: float
    active_duration_sec: float  # Time with GPU util > threshold

    # Power statistics (Watts)
    avg_power_w: float
    min_power_w: float
    max_power_w: float
    peak_power_w: float
    idle_power_w: float
    active_avg_power_w: float  # Average power during active inference

    # Energy consumption
    total_energy_joules: float
    total_energy_kwh: float
    benchmark_energy_joules: float  # Energy during active benchmark period
    benchmark_energy_kwh: float

    # GPU utilization
    avg_gpu_util: float
    max_gpu_util: float
    avg_mem_util: float
    max_mem_util: float

    # Efficiency metrics
    tokens_per_joule: float
    joules_per_token: float
    tokens_per_kwh: float

    # Paper-specific metrics (per million OUTPUT tokens)
    wh_per_mtok: float  # Wh per million output tokens
    cost_per_mtok: float  # USD per million output tokens

    # Performance data (from aiperf)
    total_tokens: int
    output_tokens: int
    input_tokens: int
    throughput_tokens_per_sec: float

    # Cost estimation (example rates)
    estimated_cost_usd: float  # Based on typical electricity rates

    # Temperature (Celsius) - may be None if not available
    avg_temp_c: Optional[float] = None
    min_temp_c: Optional[float] = None
    max_temp_c: Optional[float] = None
    active_avg_temp_c: Optional[float] = None

    # Throttling detection (cumulative time in seconds)
    thermal_throttle_sec: Optional[float] = None
    power_throttle_sec: Optional[float] = None
    throttled: bool = False  # True if any throttling detected


def parse_dcgm_metrics(log_content: str) -> List[Dict[str, float]]:
    """
    Parse DCGM dmon log output.

    Format (with temperature and throttling fields):
    #Entity   GPUTMP   POWER   TOTEC   GPUTL   MCUTL   PWRVIO   THERMVIO
    ID         C        W       mJ                      us       us
    GPU 0     45       56.955  750732145  0  0  0  0
    """
    metrics = []

    for line in log_content.strip().split('\n'):
        line = line.strip()

        # Skip headers and empty lines
        if not line or line.startswith('#') or line.startswith('ID'):
            continue

        # Parse GPU data line - try 8-column format first (with temp + throttling)
        # Format: "GPU 0     45     56.955  750732145  0  0  0  0"
        match = re.match(r'GPU\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', line)
        if match:
            metrics.append({
                'gpu_id': int(match.group(1)),
                'gpu_temp_c': int(match.group(2)),
                'power_w': float(match.group(3)),
                'total_energy_mj': int(match.group(4)),
                'gpu_util': int(match.group(5)),
                'mem_util': int(match.group(6)),
                'power_violation_us': int(match.group(7)),
                'thermal_violation_us': int(match.group(8))
            })
            continue

        # Try 6-column format (with temperature, no throttling)
        # Format: "GPU 0     45     56.955  750732145  0  0"
        match = re.match(r'GPU\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)', line)
        if match:
            metrics.append({
                'gpu_id': int(match.group(1)),
                'gpu_temp_c': int(match.group(2)),
                'power_w': float(match.group(3)),
                'total_energy_mj': int(match.group(4)),
                'gpu_util': int(match.group(5)),
                'mem_util': int(match.group(6)),
                'power_violation_us': None,
                'thermal_violation_us': None
            })
            continue

        # Fallback to 5-column format (without temperature, for older logs)
        # Format: "GPU 0     56.955  750732145  0  0"
        match = re.match(r'GPU\s+(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)', line)
        if match:
            metrics.append({
                'gpu_id': int(match.group(1)),
                'gpu_temp_c': None,
                'power_w': float(match.group(2)),
                'total_energy_mj': int(match.group(3)),
                'gpu_util': int(match.group(4)),
                'mem_util': int(match.group(5)),
                'power_violation_us': None,
                'thermal_violation_us': None
            })

    return metrics


def analyze_power(
    gpu_metrics_path: Path,
    aiperf_results_path: Path,
    benchmark_name: str,
    model_name: str,
    electricity_rate_usd_per_kwh: float = 0.12
) -> Optional[PowerMetrics]:
    """
    Analyze power consumption from GPU metrics and AIPerf results.

    Args:
        gpu_metrics_path: Path to gpu_metrics.log
        aiperf_results_path: Path to profile_export_aiperf.json
        benchmark_name: Name of the benchmark
        model_name: Model being benchmarked
        electricity_rate_usd_per_kwh: Electricity cost per kWh

    Returns:
        PowerMetrics object with analysis results
    """
    logger = logging.getLogger(__name__)

    # Read GPU metrics
    if not gpu_metrics_path.exists():
        logger.warning(f"GPU metrics file not found: {gpu_metrics_path}")
        return None

    with open(gpu_metrics_path, 'r') as f:
        gpu_log = f.read()

    metrics = parse_dcgm_metrics(gpu_log)
    if not metrics:
        logger.warning("No GPU metrics parsed from log")
        return None

    # Read AIPerf results
    if not aiperf_results_path.exists():
        logger.warning(f"AIPerf results file not found: {aiperf_results_path}")
        return None

    with open(aiperf_results_path, 'r') as f:
        aiperf_data = json.load(f)

    # Extract power data
    power_readings = [m['power_w'] for m in metrics]
    gpu_util_readings = [m['gpu_util'] for m in metrics]
    mem_util_readings = [m['mem_util'] for m in metrics]
    temp_readings = [m.get('gpu_temp_c') for m in metrics if m.get('gpu_temp_c') is not None]

    # Calculate measurement duration (1 sample per second from DCGM)
    measurement_duration_sec = len(metrics)

    # Get benchmark duration from AIPerf
    benchmark_duration_sec = aiperf_data.get('benchmark_duration', {}).get('avg', measurement_duration_sec)

    # Power statistics (all readings)
    avg_power_w = sum(power_readings) / len(power_readings)
    min_power_w = min(power_readings)
    max_power_w = max(power_readings)

    # Identify idle vs active power based on GPU utilization
    # Consider "active" when GPU util > 50% (actual inference work)
    # This excludes vLLM startup, model loading, and warmup overhead
    ACTIVE_THRESHOLD = 50  # GPU util % to consider as "active inference"

    # Separate readings by phase
    idle_readings = [p for p, u in zip(power_readings, gpu_util_readings) if u < 10]
    active_readings = [p for p, u in zip(power_readings, gpu_util_readings) if u >= ACTIVE_THRESHOLD]

    # Find indices of active period for TOTEC calculation
    active_indices = [i for i, u in enumerate(gpu_util_readings) if u >= ACTIVE_THRESHOLD]

    # Count active samples for duration calculation
    active_sample_count = len(active_readings)

    idle_power_w = sum(idle_readings) / len(idle_readings) if idle_readings else min_power_w
    peak_power_w = max(active_readings) if active_readings else max_power_w

    # Calculate total energy using TOTEC (total energy counter) if available
    # TOTEC is cumulative energy in millijoules - covers entire measurement period
    if metrics[0].get('total_energy_mj') and metrics[-1].get('total_energy_mj'):
        total_energy_mj = metrics[-1]['total_energy_mj'] - metrics[0]['total_energy_mj']
        total_energy_joules = total_energy_mj / 1000.0
    else:
        # Fallback: estimate from power readings (assumes 1 second intervals)
        total_energy_joules = sum(power_readings)  # W * 1s = J

    total_energy_kwh = total_energy_joules / 3_600_000  # J to kWh

    # Calculate energy during ACTIVE benchmark period only
    # This excludes vLLM startup, model loading, warmup, and cooldown
    # Use TOTEC delta for active period if we have indices, otherwise sum power readings
    if active_indices and len(active_indices) >= 2:
        first_active = active_indices[0]
        last_active = active_indices[-1]

        # Try to use TOTEC for more accurate energy measurement during active period
        if (metrics[first_active].get('total_energy_mj') and
            metrics[last_active].get('total_energy_mj')):
            active_energy_mj = metrics[last_active]['total_energy_mj'] - metrics[first_active]['total_energy_mj']
            benchmark_energy_joules = active_energy_mj / 1000.0
        else:
            # Fallback: sum power readings (each reading is ~1 second)
            benchmark_energy_joules = sum(active_readings)

        active_power_avg = sum(active_readings) / len(active_readings)
    elif active_readings:
        # Few active readings, just sum them
        benchmark_energy_joules = sum(active_readings)
        active_power_avg = sum(active_readings) / len(active_readings)
    else:
        # No active readings detected - use AIPerf duration with average power
        active_power_avg = avg_power_w
        benchmark_energy_joules = active_power_avg * benchmark_duration_sec

    benchmark_energy_kwh = benchmark_energy_joules / 3_600_000

    # GPU utilization stats
    avg_gpu_util = sum(gpu_util_readings) / len(gpu_util_readings) if gpu_util_readings else 0
    max_gpu_util = max(gpu_util_readings) if gpu_util_readings else 0
    avg_mem_util = sum(mem_util_readings) / len(mem_util_readings) if mem_util_readings else 0
    max_mem_util = max(mem_util_readings) if mem_util_readings else 0

    # Temperature statistics (if available)
    avg_temp_c = None
    min_temp_c = None
    max_temp_c = None
    active_avg_temp_c = None
    if temp_readings:
        avg_temp_c = sum(temp_readings) / len(temp_readings)
        min_temp_c = min(temp_readings)
        max_temp_c = max(temp_readings)
        # Active temperature (during inference)
        active_temps = [t for t, u in zip(
            [m.get('gpu_temp_c') for m in metrics],
            gpu_util_readings
        ) if t is not None and u >= ACTIVE_THRESHOLD]
        if active_temps:
            active_avg_temp_c = sum(active_temps) / len(active_temps)

    # Throttling detection (compare first and last readings - counters are cumulative)
    thermal_throttle_sec = None
    power_throttle_sec = None
    throttled = False
    if metrics and metrics[0].get('thermal_violation_us') is not None:
        first_thermal = metrics[0].get('thermal_violation_us', 0)
        last_thermal = metrics[-1].get('thermal_violation_us', 0)
        first_power = metrics[0].get('power_violation_us', 0)
        last_power = metrics[-1].get('power_violation_us', 0)
        # Convert from microseconds to seconds
        thermal_throttle_sec = (last_thermal - first_thermal) / 1_000_000
        power_throttle_sec = (last_power - first_power) / 1_000_000
        throttled = thermal_throttle_sec > 0 or power_throttle_sec > 0
        if throttled:
            logger.warning(f"Throttling detected! Thermal: {thermal_throttle_sec:.2f}s, Power: {power_throttle_sec:.2f}s")

    # Token counts from AIPerf
    total_output_tokens = int(aiperf_data.get('total_output_tokens', {}).get('avg', 0))
    total_input_tokens = int(aiperf_data.get('total_isl', {}).get('avg', 0))
    total_tokens = total_output_tokens + total_input_tokens
    throughput = aiperf_data.get('output_token_throughput', {}).get('avg', 0)

    # Efficiency metrics
    tokens_per_joule = total_tokens / benchmark_energy_joules if benchmark_energy_joules > 0 else 0
    joules_per_token = benchmark_energy_joules / total_tokens if total_tokens > 0 else 0
    tokens_per_kwh = total_tokens / benchmark_energy_kwh if benchmark_energy_kwh > 0 else 0

    # Paper-specific metrics: Wh/MTok and $/MTok (per million OUTPUT tokens)
    # Wh/MTok = (energy in Wh) / (output tokens / 1,000,000)
    benchmark_energy_wh = benchmark_energy_joules / 3600  # Convert J to Wh
    wh_per_mtok = (benchmark_energy_wh / total_output_tokens) * 1_000_000 if total_output_tokens > 0 else 0
    # $/MTok = Wh/MTok * $/kWh / 1000
    cost_per_mtok = (wh_per_mtok * electricity_rate_usd_per_kwh) / 1000 if wh_per_mtok > 0 else 0

    # Cost estimation (for this run)
    estimated_cost_usd = total_energy_kwh * electricity_rate_usd_per_kwh

    # Try to get GPU name from metrics or use placeholder
    gpu_name = "Unknown GPU"

    return PowerMetrics(
        benchmark_name=benchmark_name,
        model=model_name,
        gpu_name=gpu_name,
        benchmark_duration_sec=round(benchmark_duration_sec, 2),
        measurement_duration_sec=measurement_duration_sec,
        active_duration_sec=active_sample_count,  # Each sample is ~1 second
        avg_power_w=round(avg_power_w, 2),
        min_power_w=round(min_power_w, 2),
        max_power_w=round(max_power_w, 2),
        peak_power_w=round(peak_power_w, 2),
        idle_power_w=round(idle_power_w, 2),
        active_avg_power_w=round(active_power_avg, 2),
        total_energy_joules=round(total_energy_joules, 2),
        total_energy_kwh=round(total_energy_kwh, 6),
        benchmark_energy_joules=round(benchmark_energy_joules, 2),
        benchmark_energy_kwh=round(benchmark_energy_kwh, 6),
        avg_gpu_util=round(avg_gpu_util, 1),
        max_gpu_util=round(max_gpu_util, 1),
        avg_mem_util=round(avg_mem_util, 1),
        max_mem_util=round(max_mem_util, 1),
        tokens_per_joule=round(tokens_per_joule, 4),
        joules_per_token=round(joules_per_token, 4),
        tokens_per_kwh=round(tokens_per_kwh, 0),
        wh_per_mtok=round(wh_per_mtok, 2),
        cost_per_mtok=round(cost_per_mtok, 4),
        total_tokens=total_tokens,
        output_tokens=total_output_tokens,
        input_tokens=total_input_tokens,
        throughput_tokens_per_sec=round(throughput, 2),
        estimated_cost_usd=round(estimated_cost_usd, 6),
        # Optional fields with defaults (must come last)
        avg_temp_c=round(avg_temp_c, 1) if avg_temp_c is not None else None,
        min_temp_c=round(min_temp_c, 1) if min_temp_c is not None else None,
        max_temp_c=round(max_temp_c, 1) if max_temp_c is not None else None,
        active_avg_temp_c=round(active_avg_temp_c, 1) if active_avg_temp_c is not None else None,
        thermal_throttle_sec=round(thermal_throttle_sec, 2) if thermal_throttle_sec is not None else None,
        power_throttle_sec=round(power_throttle_sec, 2) if power_throttle_sec is not None else None,
        throttled=throttled
    )


def generate_power_report(
    results_dir: Path,
    benchmarks: List[Dict[str, Any]],
    gpu_name: str = "Unknown GPU"
) -> Dict[str, Any]:
    """
    Generate a comprehensive power analysis report for all benchmarks.

    Args:
        results_dir: Directory containing benchmark results
        benchmarks: List of benchmark configs with names and models
        gpu_name: Name of the GPU

    Returns:
        Dictionary with full power analysis report
    """
    logger = logging.getLogger(__name__)

    report = {
        "report_generated": datetime.now().isoformat(),
        "gpu": gpu_name,
        "summary": {},
        "benchmarks": []
    }

    total_energy_j = 0
    total_tokens = 0

    # Find all benchmark subdirectories
    benchmark_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])

    for bench_dir in benchmark_dirs:
        gpu_metrics_path = bench_dir / "gpu_metrics.log"
        aiperf_path = bench_dir / "profile_export_aiperf.json"

        # Extract benchmark name from directory (format: 001_benchmark_name)
        dir_name = bench_dir.name
        parts = dir_name.split('_', 1)
        bench_name = parts[1] if len(parts) > 1 else dir_name

        # Try to find model name from aiperf results
        model_name = "Unknown"
        if aiperf_path.exists():
            try:
                with open(aiperf_path) as f:
                    aiperf_data = json.load(f)
                    model_names = aiperf_data.get('input_config', {}).get('endpoint', {}).get('model_names', [])
                    if model_names:
                        model_name = model_names[0]
            except Exception as e:
                logger.debug(f"Could not read model name: {e}")

        metrics = analyze_power(gpu_metrics_path, aiperf_path, bench_name, model_name)

        if metrics:
            metrics.gpu_name = gpu_name
            report["benchmarks"].append(asdict(metrics))
            total_energy_j += metrics.benchmark_energy_joules
            total_tokens += metrics.total_tokens

    # Calculate summary
    if report["benchmarks"]:
        report["summary"] = {
            "total_benchmarks": len(report["benchmarks"]),
            "total_energy_joules": round(total_energy_j, 2),
            "total_energy_kwh": round(total_energy_j / 3_600_000, 6),
            "total_tokens_processed": total_tokens,
            "overall_efficiency_tokens_per_joule": round(total_tokens / total_energy_j, 4) if total_energy_j > 0 else 0,
            "overall_efficiency_joules_per_token": round(total_energy_j / total_tokens, 4) if total_tokens > 0 else 0,
            "estimated_total_cost_usd": round(sum(b.get('estimated_cost_usd', 0) for b in report["benchmarks"]), 6)
        }

    return report


def save_power_report(report: Dict[str, Any], output_path: Path) -> None:
    """Save power report to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)


def print_power_summary(report: Dict[str, Any]) -> str:
    """Generate a human-readable power summary."""
    lines = []
    lines.append("=" * 60)
    lines.append("POWER CONSUMPTION ANALYSIS")
    lines.append("=" * 60)
    lines.append(f"GPU: {report.get('gpu', 'Unknown')}")
    lines.append(f"Report Generated: {report.get('report_generated', 'Unknown')}")
    lines.append("")

    summary = report.get("summary", {})
    if summary:
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total Benchmarks: {summary.get('total_benchmarks', 0)}")
        lines.append(f"Total Energy: {summary.get('total_energy_joules', 0):.2f} J ({summary.get('total_energy_kwh', 0)*1000:.4f} Wh)")
        lines.append(f"Total Tokens: {summary.get('total_tokens_processed', 0):,}")
        lines.append(f"Efficiency: {summary.get('overall_efficiency_tokens_per_joule', 0):.2f} tokens/J")
        lines.append(f"Est. Cost: ${summary.get('estimated_total_cost_usd', 0):.6f}")
        lines.append("")

    lines.append("BENCHMARK DETAILS")
    lines.append("-" * 40)

    for bench in report.get("benchmarks", []):
        lines.append(f"\n{bench.get('benchmark_name', 'Unknown')}:")
        lines.append(f"  Model: {bench.get('model', 'Unknown')}")
        lines.append(f"  Duration: benchmark={bench.get('benchmark_duration_sec', 0):.1f}s, active={bench.get('active_duration_sec', 0):.0f}s, total={bench.get('measurement_duration_sec', 0):.0f}s")
        lines.append(f"  Power: active_avg={bench.get('active_avg_power_w', 0):.1f}W, peak={bench.get('peak_power_w', 0):.1f}W, idle={bench.get('idle_power_w', 0):.1f}W")
        lines.append(f"  Energy (active): {bench.get('benchmark_energy_joules', 0):.2f} J ({bench.get('benchmark_energy_kwh', 0)*1000:.4f} Wh)")
        lines.append(f"  Energy (total incl. startup): {bench.get('total_energy_joules', 0):.2f} J ({bench.get('total_energy_kwh', 0)*1000:.4f} Wh)")
        lines.append(f"  GPU Util: avg={bench.get('avg_gpu_util', 0):.0f}%, max={bench.get('max_gpu_util', 0):.0f}%")
        # Add temperature if available
        if bench.get('avg_temp_c') is not None:
            lines.append(f"  Temperature: avg={bench.get('avg_temp_c', 0):.0f}°C, max={bench.get('max_temp_c', 0):.0f}°C, active_avg={bench.get('active_avg_temp_c', 0):.0f}°C")
        # Add throttling warning if detected
        if bench.get('throttled'):
            thermal = bench.get('thermal_throttle_sec', 0)
            power = bench.get('power_throttle_sec', 0)
            lines.append(f"  ⚠️  THROTTLING DETECTED: thermal={thermal:.2f}s, power={power:.2f}s")
        lines.append(f"  Tokens: {bench.get('total_tokens', 0):,} total, {bench.get('output_tokens', 0):,} output ({bench.get('throughput_tokens_per_sec', 0):.1f} tok/s)")
        lines.append(f"  Efficiency: {bench.get('tokens_per_joule', 0):.2f} tokens/J, {bench.get('joules_per_token', 0):.4f} J/token")
        lines.append(f"  Paper metrics: {bench.get('wh_per_mtok', 0):.2f} Wh/MTok, ${bench.get('cost_per_mtok', 0):.4f}/MTok")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    # Test with example data
    import sys

    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        # Default test path
        results_dir = Path("results/20251126_184619_RTX_5090_Test")

    if results_dir.exists():
        report = generate_power_report(results_dir, [], gpu_name="NVIDIA GeForce RTX 5090")
        print(print_power_summary(report))

        # Save report
        report_path = results_dir / "power_analysis.json"
        save_power_report(report, report_path)
        print(f"\nReport saved to: {report_path}")
    else:
        print(f"Results directory not found: {results_dir}")
