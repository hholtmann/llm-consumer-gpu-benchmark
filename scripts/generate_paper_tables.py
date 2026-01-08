#!/usr/bin/env python3
"""
Generate LaTeX tables for paper from benchmark results.

Usage:
    python scripts/generate_paper_tables.py [--results-dir results/] [--output results/paper_tables_filled.tex]
"""

import json
import os
import re
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List
from collections import defaultdict


@dataclass
class BenchmarkResult:
    """Parsed benchmark result."""
    benchmark_name: str
    model: str
    precision: str
    gpu_config: str
    context_tokens: int
    concurrency: int
    workload_type: str  # rag, api, agentic

    # Performance metrics
    tps: float                    # Total output tokens/sec
    tps_per_user: float           # Per-user output tokens/sec
    ttft_avg: float               # Time to first token (ms)
    ttft_p50: float               # TTFT median
    ttft_p95: float               # TTFT P95
    itl_avg: float                # Inter-token latency avg (ms)
    itl_p50: float                # ITL median
    itl_p95: float                # ITL P95
    itl_p99: float                # ITL P99

    # Optional energy/thermal metrics
    wh_per_mtok: Optional[float] = None
    avg_power_w: Optional[float] = None
    max_temp_c: Optional[float] = None
    throttle_pct: Optional[float] = None

    # Fallback flags
    used_fp8_kv_cache: bool = False  # True if FP8 KV cache was used due to OOM


def parse_precision_from_model(model: str, benchmark_name: str) -> str:
    """Extract precision from model name or benchmark name."""
    name = f"{model} {benchmark_name}".lower()
    if "nvfp4" in name:
        return "NVFP4"
    elif "w4a16" in name:
        return "W4A16"
    elif "mxfp4" in name:
        return "MXFP4"
    elif "bf16" in name:
        return "BF16"
    elif "fp16" in name:
        return "FP16"
    return "Unknown"


def parse_gpu_config(metadata: dict) -> str:
    """Extract GPU config from metadata."""
    gpu = metadata.get("gpu", "")
    suite = metadata.get("suite_name", "")

    # Try to extract from gpu field
    if "5090" in gpu or "5090" in suite:
        if "2x" in gpu or "2x" in suite:
            return "RTX 5090 2x"
        return "RTX 5090 1x"
    elif "5070" in gpu or "5070" in suite:
        if "2x" in gpu or "2x" in suite:
            return "RTX 5070 Ti 2x"
        return "RTX 5070 Ti 1x"
    elif "5060" in gpu or "5060" in suite:
        if "2x" in gpu or "2x" in suite:
            return "RTX 5060 Ti 2x"
        return "RTX 5060 Ti 1x"

    return gpu or "Unknown"


def parse_workload_type(benchmark_name: str) -> str:
    """Determine workload type from benchmark name."""
    name = benchmark_name.lower()
    if "rag" in name:
        return "rag"
    elif "api" in name:
        return "api"
    elif "agentic" in name or "lora" in name:
        return "agentic"
    return "unknown"


def parse_context_tokens(benchmark_name: str, aiperf_config: dict) -> int:
    """Extract context length from benchmark name or config."""
    name = benchmark_name.lower()

    # Try to extract from name (e.g., rag-8k, rag-16k)
    match = re.search(r"(\d+)k", name)
    if match:
        return int(match.group(1)) * 1024

    # Fall back to config
    return aiperf_config.get("synthetic_input_tokens_mean", 0)


def parse_model_short_name(model: str) -> str:
    """Get short model name for tables."""
    model_lower = model.lower()

    if "qwen3-8b" in model_lower or "qwen3_8b" in model_lower:
        return "Qwen3-8B"
    elif "gemma-3-12b" in model_lower or "gemma3-12b" in model_lower:
        return "Gemma3-12B"
    elif "gemma-3-27b" in model_lower or "gemma3-27b" in model_lower:
        return "Gemma3-27B"
    elif "gpt-oss-20b" in model_lower or "gpt_oss" in model_lower:
        return "GPT-OSS-20B"
    elif "ministral" in model_lower:
        return "Ministral-3-14B"

    return model.split("/")[-1]


def parse_gpu_metrics(gpu_metrics_file: Path) -> tuple[float, float, float]:
    """Parse gpu_metrics.log to extract average power, max temp, and throttle %.

    Column order: GPU ID, temp, power, energy, gpu_util, mem_util
    Returns: (avg_power_w, max_temp_c, throttle_pct)
    """
    if not gpu_metrics_file.exists():
        return 0.0, 0.0, 0.0

    power_readings = []
    temp_readings = []
    try:
        with open(gpu_metrics_file) as f:
            for line in f:
                line = line.strip()
                # Skip headers and empty lines
                if not line or line.startswith("#") or line.startswith("ID"):
                    continue

                parts = line.split()
                # New format: GPU 0 temp power energy gpu_util mem_util
                if len(parts) >= 4 and parts[0] == "GPU":
                    try:
                        temp_c = float(parts[2]) if parts[2] != '-' else 0
                        power_w = float(parts[3]) if parts[3] != '-' else 0
                        # Filter out idle readings (< 100W typically means idle/loading)
                        if power_w > 100:
                            power_readings.append(power_w)
                        if temp_c > 0:
                            temp_readings.append(temp_c)
                    except (ValueError, IndexError):
                        continue
    except IOError:
        return 0.0, 0.0, 0.0

    if not power_readings:
        return 0.0, 0.0, 0.0

    avg_power = sum(power_readings) / len(power_readings)
    max_temp = max(temp_readings) if temp_readings else 0.0
    # Throttling: % of samples at >= 83°C
    throttle_samples = len([t for t in temp_readings if t >= 83])
    throttle_pct = (throttle_samples / len(temp_readings) * 100) if temp_readings else 0.0

    return avg_power, max_temp, throttle_pct


def load_benchmark_result(result_dir: Path) -> Optional[BenchmarkResult]:
    """Load and parse a single benchmark result directory."""
    metadata_file = result_dir / "metadata.json"
    aiperf_file = result_dir / "profile_export_aiperf.json"
    gpu_metrics_file = result_dir / "gpu_metrics.log"

    if not metadata_file.exists() or not aiperf_file.exists():
        return None

    try:
        with open(metadata_file) as f:
            metadata = json.load(f)
        with open(aiperf_file) as f:
            aiperf = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading {result_dir}: {e}")
        return None

    # Skip failed benchmarks
    if metadata.get("status") != "success":
        return None

    benchmark_name = metadata.get("benchmark_name", "")
    model = metadata.get("model", "")
    aiperf_config = metadata.get("aiperf_config", {})

    # Extract metrics from aiperf results (handle None values)
    tps = (aiperf.get("output_token_throughput") or {}).get("avg", 0)
    tps_per_user = (aiperf.get("output_token_throughput_per_user") or {}).get("avg", 0)
    ttft = aiperf.get("time_to_first_token") or {}
    itl = aiperf.get("inter_token_latency") or {}

    # Get total output tokens and duration for energy calculation
    total_output_tokens = (aiperf.get("total_output_tokens") or {}).get("avg", 0)
    benchmark_duration_sec = (aiperf.get("benchmark_duration") or {}).get("avg", 0)

    # Parse GPU power and thermal metrics
    avg_power_w, max_temp_c, throttle_pct = parse_gpu_metrics(gpu_metrics_file)

    # Calculate Wh/MTok
    # Energy (Wh) = Power (W) * Time (h)
    # Wh/MTok = Energy (Wh) / (tokens / 1_000_000)
    wh_per_mtok = 0.0
    if avg_power_w > 0 and total_output_tokens > 0 and benchmark_duration_sec > 0:
        energy_wh = avg_power_w * (benchmark_duration_sec / 3600)
        wh_per_mtok = energy_wh / (total_output_tokens / 1_000_000)

    return BenchmarkResult(
        benchmark_name=benchmark_name,
        model=parse_model_short_name(model),
        precision=parse_precision_from_model(model, benchmark_name),
        gpu_config=parse_gpu_config(metadata),
        context_tokens=parse_context_tokens(benchmark_name, aiperf_config),
        concurrency=aiperf_config.get("concurrency", 0),
        workload_type=parse_workload_type(benchmark_name),
        tps=tps,
        tps_per_user=tps_per_user,
        ttft_avg=ttft.get("avg", 0),
        ttft_p50=ttft.get("p50", 0),
        ttft_p95=ttft.get("p95", 0),
        itl_avg=itl.get("avg", 0),
        itl_p50=itl.get("p50", 0),
        itl_p95=itl.get("p95", 0),
        itl_p99=itl.get("p99", 0),
        wh_per_mtok=wh_per_mtok,
        avg_power_w=avg_power_w,
        max_temp_c=max_temp_c,
        throttle_pct=throttle_pct,
        used_fp8_kv_cache=metadata.get("used_fp8_kv_cache", False),
    )


def load_all_results(results_dir: Path) -> List[BenchmarkResult]:
    """Load all benchmark results from the results directory."""
    results = []

    for suite_dir in results_dir.iterdir():
        if not suite_dir.is_dir() or suite_dir.name.startswith("."):
            continue

        for benchmark_dir in suite_dir.iterdir():
            if not benchmark_dir.is_dir():
                continue

            result = load_benchmark_result(benchmark_dir)
            if result:
                results.append(result)

    return results


def format_number(val: float, decimals: int = 1) -> str:
    """Format number for LaTeX table."""
    if val == 0:
        return "..."
    return f"{val:.{decimals}f}"


def generate_rag_table(results: List[BenchmarkResult], gpu_filter: str) -> str:
    """Generate RAG throughput table for a specific GPU."""

    # Filter results
    rag_results = [r for r in results
                   if r.workload_type == "rag"
                   and gpu_filter in r.gpu_config]

    if not rag_results:
        return f"% No RAG results found for {gpu_filter}\n"

    # Sort by model, precision, context
    rag_results.sort(key=lambda r: (r.model, r.precision, r.context_tokens))

    lines = []
    for r in rag_results:
        ctx = f"{r.context_tokens // 1024}k"
        gpu_short = "1x" if "1x" in r.gpu_config else "2x"
        lines.append(
            f"    {r.model} & {gpu_short} & {r.precision} & {ctx} & {r.concurrency} & "
            f"{format_number(r.tps)} & {format_number(r.tps_per_user)} & "
            f"{format_number(r.ttft_avg)} & {format_number(r.itl_avg)} & "
            f"{format_number(r.itl_p95)} & {format_number(r.itl_p99)} \\\\"
        )

    return "\n".join(lines)


def generate_api_table(results: List[BenchmarkResult]) -> str:
    """Generate API throughput table."""

    api_results = [r for r in results if r.workload_type == "api"]

    if not api_results:
        return "% No API results found\n"

    api_results.sort(key=lambda r: (r.gpu_config, r.model, r.concurrency))

    lines = []
    for r in api_results:
        lines.append(
            f"    {r.model} & {r.gpu_config} & {r.precision} & {r.concurrency} & "
            f"{format_number(r.tps)} & {format_number(r.ttft_avg)} & "
            f"{format_number(r.itl_p95)} & {format_number(r.itl_p99)} \\\\"
        )

    return "\n".join(lines)


def generate_filled_latex(results: List[BenchmarkResult], template_path: Path, output_path: Path):
    """Generate filled LaTeX tables from results."""

    # Group results by GPU
    by_gpu = defaultdict(list)
    for r in results:
        by_gpu[r.gpu_config].append(r)

    output_lines = [
        "% =============================================================================",
        "% AUTO-GENERATED PAPER TABLES - Filled from benchmark results",
        f"% Generated from {len(results)} benchmark results",
        "% =============================================================================",
        "",
    ]

    # Generate RAG tables per GPU
    for gpu in ["RTX 5090", "RTX 5070 Ti", "RTX 5060 Ti"]:
        output_lines.append(f"% --- RAG Results for {gpu} ---")
        output_lines.append(f"% Single GPU (1x)")
        output_lines.append(generate_rag_table(results, f"{gpu} 1x"))
        output_lines.append("")
        output_lines.append(f"% Dual GPU (2x)")
        output_lines.append(generate_rag_table(results, f"{gpu} 2x"))
        output_lines.append("")

    # Generate API table
    output_lines.append("% --- API Results (All GPUs) ---")
    output_lines.append(generate_api_table(results))
    output_lines.append("")

    # Generate summary CSV for easy reference
    output_lines.append("% =============================================================================")
    output_lines.append("% RAW DATA CSV FORMAT (for reference)")
    output_lines.append("% =============================================================================")
    output_lines.append("% model,gpu_config,precision,context,concurrency,workload,tps,tps_per_user,ttft_avg,itl_avg,itl_p95,power_w,wh_mtok")

    for r in sorted(results, key=lambda x: (x.gpu_config, x.model, x.workload_type)):
        output_lines.append(
            f"% {r.model},{r.gpu_config},{r.precision},{r.context_tokens},{r.concurrency},"
            f"{r.workload_type},{r.tps:.1f},{r.tps_per_user:.1f},{r.ttft_avg:.1f},{r.itl_avg:.2f},{r.itl_p95:.2f},"
            f"{r.avg_power_w or 0:.0f},{r.wh_per_mtok or 0:.1f}"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"Generated {output_path} with {len(results)} results")

    # Also generate a clean CSV
    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("model,gpu_config,precision,context_tokens,concurrency,workload_type,"
                "tps,tps_per_user,ttft_avg,ttft_p50,ttft_p95,itl_avg,itl_p50,itl_p95,itl_p99,"
                "avg_power_w,max_temp_c,throttle_pct,wh_per_mtok,usd_per_mtok,fp8_kv_cache\n")
        for r in sorted(results, key=lambda x: (x.gpu_config, x.model, x.workload_type)):
            # Calculate $/MTok at $0.12/kWh
            usd_per_mtok = (r.wh_per_mtok or 0) * 0.12 / 1000 if r.wh_per_mtok else 0
            # Add asterisk to model name if FP8 KV cache was used
            model_display = f"{r.model}*" if r.used_fp8_kv_cache else r.model
            f.write(
                f"{model_display},{r.gpu_config},{r.precision},{r.context_tokens},{r.concurrency},"
                f"{r.workload_type},{r.tps:.2f},{r.tps_per_user:.2f},"
                f"{r.ttft_avg:.2f},{r.ttft_p50:.2f},{r.ttft_p95:.2f},"
                f"{r.itl_avg:.2f},{r.itl_p50:.2f},{r.itl_p95:.2f},{r.itl_p99:.2f},"
                f"{r.avg_power_w or 0:.1f},{r.max_temp_c or 0:.1f},{r.throttle_pct or 0:.1f},"
                f"{r.wh_per_mtok or 0:.2f},{usd_per_mtok:.6f},{r.used_fp8_kv_cache}\n"
            )

    # Print warning if any results used FP8 KV cache
    fp8_results = [r for r in results if r.used_fp8_kv_cache]
    if fp8_results:
        print(f"⚠ {len(fp8_results)} benchmark(s) used FP8 KV cache fallback (marked with * in model name):")
        for r in fp8_results:
            print(f"  - {r.benchmark_name} ({r.gpu_config})")

    print(f"Generated {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables from benchmark results")
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Directory containing benchmark results")
    parser.add_argument("--output", type=Path, default=Path("results/paper_tables_filled.tex"),
                        help="Output LaTeX file")
    parser.add_argument("--template", type=Path, default=Path("results/paper_tables_latex.tex"),
                        help="Template LaTeX file")

    args = parser.parse_args()

    # Find results directory
    if not args.results_dir.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        args.results_dir = script_dir / "results"

    print(f"Loading results from {args.results_dir}")
    results = load_all_results(args.results_dir)

    if not results:
        print("No benchmark results found!")
        return 1

    print(f"Found {len(results)} benchmark results")

    # Print summary
    by_gpu = defaultdict(int)
    by_workload = defaultdict(int)
    for r in results:
        by_gpu[r.gpu_config] += 1
        by_workload[r.workload_type] += 1

    print("\nResults by GPU:")
    for gpu, count in sorted(by_gpu.items()):
        print(f"  {gpu}: {count}")

    print("\nResults by workload:")
    for wl, count in sorted(by_workload.items()):
        print(f"  {wl}: {count}")

    # Generate output
    generate_filled_latex(results, args.template, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
