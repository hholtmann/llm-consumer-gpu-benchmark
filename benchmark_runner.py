#!/usr/bin/env python3
"""
LLM Benchmark Runner for Vast.ai
Orchestrates vLLM deployment, AIPerf benchmarking, and result collection on Vast.ai instances.
Supports both single benchmarks and batch benchmark suites.
"""

import os
import sys
import time
import json
import logging
import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table

from vast_manager import VastManager
from result_uploader import ResultUploader
from utils import setup_logging, validate_config, format_duration
from power_analysis import generate_power_report, save_power_report, print_power_summary


console = Console()

# LoRA adapter configuration: maps adapter names to HuggingFace repo IDs
# These will be downloaded to /models/loras/<adapter-name> on the instance
# Repo IDs can be overridden via environment variables (see .env.example)
def get_lora_adapters():
    return {
        'customer-support-faq': os.getenv('LORA_CUSTOMER_SUPPORT_FAQ', 'holtmann/qwen3-8b-customer-support-faq-lora'),
        'technical-docs': os.getenv('LORA_TECHNICAL_DOCS', 'holtmann/qwen3-8b-technical-docs-lora'),
        'json-output': os.getenv('LORA_JSON_OUTPUT', 'holtmann/qwen3-8b-json-output-lora'),
    }

LORA_BASE_PATH = '/models/loras'


class BenchmarkSuiteRunner:
    """Orchestrator for running benchmark suites on a single Vast.ai instance"""

    def __init__(self, suite_config: Dict[str, Any], args: argparse.Namespace):
        """
        Initialize the benchmark suite runner.

        Args:
            suite_config: Suite configuration from YAML
            args: Command-line arguments
        """
        self.suite_config = suite_config
        self.args = args
        self.logger = logging.getLogger(__name__)

        # Extract instance and S3 config
        self.instance_config = suite_config['instance']
        self.s3_config = suite_config['s3']
        self.benchmarks = suite_config['benchmarks']

        # Build compatible config dict for VastManager and ResultUploader
        self.config = self._build_legacy_config()

        # Initialize managers
        self.vast_manager = VastManager(self.config, self.logger)
        self.result_uploader = ResultUploader(self.config, self.logger)

        # Runtime state
        self.instance_id: Optional[int] = None
        self.instance_ip: Optional[str] = None
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.suite_name = suite_config.get('name', 'unnamed_suite')
        self.results_base_dir = Path("results") / f"{self.run_id}_{self._sanitize_name(self.suite_name)}"
        self.results_base_dir.mkdir(parents=True, exist_ok=True)

        # Track current model for cleanup decisions
        self.current_model: Optional[str] = None

        # Track benchmark results
        self.benchmark_results: List[Dict[str, Any]] = []

    def _build_legacy_config(self) -> Dict[str, Any]:
        """Build config dict compatible with VastManager and ResultUploader"""
        return {
            'vast': {
                'gpu_name': self.instance_config['gpu_type'],
                'num_gpus': self.instance_config['gpu_count'],
                'disk_space': self.instance_config['disk_space_gb'],
                'image': self.instance_config.get('image', 'holtmann/llm-benchmark:latest'),
                'max_bid_price': self.instance_config.get('max_bid_price', 1.0),
                'ssh_timeout': self.instance_config.get('ssh_timeout', 900),
                'prebuilt_image': True
            },
            'vllm': {
                'port': 8000,
                'host': '0.0.0.0',
                'startup_timeout': 600
            },
            'aiperf': {},
            's3': {
                'upload_json': self.s3_config.get('upload_json', True),
                'upload_csv': self.s3_config.get('upload_csv', True),
                'upload_logs': self.s3_config.get('upload_logs', True),
                'compress_logs': self.s3_config.get('compress_logs', True),
                'timestamp_format': self.s3_config.get('timestamp_format', '%Y%m%d_%H%M%S')
            }
        }

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for use in file paths"""
        return name.replace(' ', '_').replace('/', '_').replace(':', '_')

    def run(self) -> bool:
        """
        Execute the complete benchmark suite.

        Returns:
            bool: True if all benchmarks succeeded, False if any failed
        """
        start_time = time.time()

        try:
            # Display suite header
            self._display_suite_header()

            # Step 1: Provision instance
            if not self._provision_instance():
                return False

            # Step 2: Setup environment
            if not self._setup_environment():
                return False

            # Step 3: Run all benchmarks
            all_success = True
            for i, benchmark in enumerate(self.benchmarks):
                benchmark_name = benchmark.get('name', f'benchmark_{i+1}')
                console.print(f"\n{'='*60}")
                console.print(f"[bold cyan]Benchmark {i+1}/{len(self.benchmarks)}: {benchmark_name}[/bold cyan]")
                console.print(f"{'='*60}")

                success = self._run_single_benchmark(benchmark, i)

                result = {
                    'name': benchmark_name,
                    'model': benchmark['model'],
                    'success': success,
                    'timestamp': datetime.now().isoformat()
                }
                self.benchmark_results.append(result)

                if success:
                    console.print(f"[green]✓[/green] Benchmark '{benchmark_name}' completed successfully")
                else:
                    console.print(f"[red]✗[/red] Benchmark '{benchmark_name}' failed")
                    all_success = False

                # Check if we need to cleanup the model for the next benchmark
                self._maybe_cleanup_model(benchmark, i)

            # Step 4: Generate power analysis (only for performance benchmarks)
            has_performance_benchmarks = any(b.get('type', 'performance') == 'performance' for b in self.benchmarks)
            if has_performance_benchmarks:
                self._generate_power_analysis()

            # Step 5: Upload suite summary
            self._upload_suite_summary()

            # Final summary
            elapsed = time.time() - start_time
            self._display_final_summary(elapsed)

            return all_success

        except KeyboardInterrupt:
            console.print("\n[bold red]✗ Benchmark suite interrupted by user[/bold red]")
            return False
        except Exception as e:
            self.logger.error(f"Benchmark suite failed: {e}", exc_info=True)
            console.print(f"\n[bold red]✗ Benchmark suite failed: {e}[/bold red]")
            return False
        finally:
            self._cleanup()

    def _display_suite_header(self):
        """Display suite information header"""
        suite_name = self.suite_config.get('name', 'Unnamed Suite')
        description = self.suite_config.get('description', '')

        header_text = f"[bold cyan]Benchmark Suite: {suite_name}[/bold cyan]\n"
        if description:
            header_text += f"[dim]{description}[/dim]\n"
        header_text += f"\nGPU: [green]{self.instance_config['gpu_type']} x{self.instance_config['gpu_count']}[/green]\n"
        header_text += f"Benchmarks: [yellow]{len(self.benchmarks)}[/yellow]\n"
        header_text += f"Run ID: [dim]{self.run_id}[/dim]"

        console.print(Panel.fit(header_text, border_style="cyan"))

    def _provision_instance(self) -> bool:
        """Provision a Vast.ai instance"""
        console.print("\n[bold]Step 1: Provisioning Vast.ai instance[/bold]")

        try:
            # Check if using existing instance
            if self.args.instance_id:
                console.print(f"Using existing instance ID: {self.args.instance_id}")
                instance_info = self.vast_manager.get_instance_info(self.args.instance_id)
                if not instance_info:
                    console.print(f"[red]✗[/red] Could not find instance {self.args.instance_id}")
                    return False

                self.instance_id = self.args.instance_id
                # Get IP from instance info - try different possible field names
                self.instance_ip = instance_info.get('public_ipaddr') or instance_info.get('ssh_host') or instance_info.get('ip')

                if not self.instance_ip:
                    console.print(f"[yellow]Warning:[/yellow] Could not determine IP, using instance ID for SSH")
                    self.instance_ip = str(self.args.instance_id)

                console.print(f"[green]✓[/green] Using existing instance: ID={self.instance_id}, IP={self.instance_ip}")
                return True

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Searching for available instances...", total=None)

                instance_id, instance_ip = self.vast_manager.create_instance(
                    gpu_name=self.instance_config['gpu_type'],
                    num_gpus=self.instance_config['gpu_count'],
                    disk_space=self.instance_config['disk_space_gb'],
                    image=self.instance_config.get('image', 'holtmann/llm-benchmark:latest'),
                    min_download_speed=self.instance_config.get('min_download_speed', 1000.0),
                    blacklist_machines=self.instance_config.get('blacklist_machines', [])
                )

                self.instance_id = instance_id
                self.instance_ip = instance_ip

                progress.update(task, description=f"Instance {instance_id} created at {instance_ip}")

            console.print(f"[green]✓[/green] Instance provisioned: ID={instance_id}, IP={instance_ip}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to provision instance: {e}", exc_info=True)
            console.print(f"[red]✗[/red] Failed to provision instance: {e}")
            return False

    def _setup_environment(self) -> bool:
        """Setup environment on the instance"""
        console.print("\n[bold]Step 2: Setting up environment[/bold]")

        try:
            ssh_timeout = self.instance_config.get('ssh_timeout', 900)
            console.print(f"Waiting for SSH connection (timeout: {ssh_timeout}s)...")

            if not self.vast_manager.wait_for_ssh(self.instance_id, timeout=ssh_timeout):
                console.print("[red]✗[/red] SSH connection timeout")
                return False

            # Using pre-built image - verify installations
            console.print("Using pre-built Docker image - verifying installations...")

            # Fix libcuda.so symlink
            self.vast_manager.execute_command(
                self.instance_id,
                "ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so 2>/dev/null || true"
            )

            # Setup HuggingFace token if available
            hf_token = os.getenv('HF_TOKEN')
            if hf_token:
                console.print("Setting up HuggingFace token...")
                self.vast_manager.execute_command(
                    self.instance_id,
                    f"export HF_TOKEN='{hf_token}' && huggingface-cli login --token $HF_TOKEN 2>/dev/null || true"
                )

            # Verify vLLM and aiperf
            verify_commands = [
                "python3 -m vllm.entrypoints.openai.api_server --help > /dev/null 2>&1",
                "aiperf --help > /dev/null 2>&1"
            ]

            for cmd in verify_commands:
                success = self.vast_manager.execute_command(self.instance_id, cmd)
                if not success:
                    console.print(f"[red]✗[/red] Verification failed: {cmd}")
                    return False

            console.print("[green]✓[/green] Environment verified")
            return True

        except Exception as e:
            self.logger.error(f"Environment setup failed: {e}", exc_info=True)
            console.print(f"[red]✗[/red] Environment setup failed: {e}")
            return False

    def _run_single_benchmark(self, benchmark: Dict[str, Any], index: int) -> bool:
        """
        Run a single benchmark from the suite.

        Args:
            benchmark: Benchmark configuration
            index: Benchmark index in the suite

        Returns:
            bool: True if successful
        """
        benchmark_name = benchmark.get('name', f'benchmark_{index+1}')
        model = benchmark['model']
        benchmark_type = benchmark.get('type', 'performance')
        vllm_config = benchmark.get('vllm', {})
        aiperf_config = benchmark.get('aiperf', {})
        lm_eval_config = benchmark.get('lm_eval', {})

        # Create results directory for this benchmark
        results_dir = self.results_base_dir / f"{index+1:03d}_{self._sanitize_name(benchmark_name)}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Skip if benchmark already completed successfully (check all previous runs for this suite)
        benchmark_dir_name = f"{index+1:03d}_{self._sanitize_name(benchmark_name)}"
        suite_pattern = f"*_{self._sanitize_name(self.suite_name)}"
        for prev_run_dir in Path("results").glob(suite_pattern):
            prev_benchmark_dir = prev_run_dir / benchmark_dir_name
            if (prev_benchmark_dir / 'profile_export_aiperf.json').exists():
                console.print(f"[cyan]⏭[/cyan] Skipping '{benchmark_name}' - already completed in {prev_run_dir.name}")
                return True

        try:
            # Save benchmark config
            config_file = results_dir / 'benchmark_config.json'
            config_file.write_text(json.dumps(benchmark, indent=2))

            if benchmark_type == 'quality':
                # Quality benchmark using lm_eval (lm_eval manages vLLM itself)
                return self._run_quality_benchmark(benchmark_name, model, vllm_config, lm_eval_config, results_dir, benchmark)
            else:
                # Performance benchmark using AIPerf (requires vLLM server)
                return self._run_performance_benchmark(benchmark_name, model, vllm_config, aiperf_config, results_dir, benchmark)

        except Exception as e:
            self.logger.error(f"Benchmark '{benchmark_name}' failed: {e}", exc_info=True)
            self._save_error(results_dir, str(e))
            self._collect_vllm_logs(results_dir)
            self._stop_vllm()
            self._upload_failed_benchmark(benchmark_name, model, results_dir, benchmark)
            return False

    def _run_performance_benchmark(self, benchmark_name: str, model: str, vllm_config: Dict[str, Any],
                                   aiperf_config: Dict[str, Any], results_dir: Path, benchmark: Dict[str, Any]) -> bool:
        """Run a performance benchmark using AIPerf"""
        try:
            # Track if we used FP8 KV cache as fallback
            used_fp8_kv_cache = False

            # Step 1: Download model if needed
            if model != self.current_model:
                console.print(f"Downloading model: [yellow]{model}[/yellow]")
                # Model will be downloaded when vLLM starts

            # Step 2: Start vLLM server (pass concurrency to auto-set max-num-seqs)
            concurrency = aiperf_config.get('concurrency')
            vllm_ready = False

            # First attempt with original config
            if self._start_vllm(model, vllm_config, concurrency=concurrency):
                if self._wait_for_vllm():
                    vllm_ready = True
                else:
                    # Check if it was an OOM error
                    if self._check_vllm_oom():
                        console.print("\n[yellow]⚠ OOM detected - retrying with FP8 KV cache + gpu_memory_utilization=0.95[/yellow]")
                        self._stop_vllm()

                        # Retry with FP8 KV cache and higher GPU memory utilization
                        vllm_config_fp8 = vllm_config.copy()
                        vllm_config_fp8['kv_cache_dtype'] = 'fp8'
                        vllm_config_fp8['gpu_memory_utilization'] = 0.95
                        used_fp8_kv_cache = True

                        if self._start_vllm(model, vllm_config_fp8, concurrency=concurrency):
                            if self._wait_for_vllm():
                                vllm_ready = True
                                console.print("[green]✓ vLLM started with FP8 KV cache + gpu_memory_utilization=0.95[/green]")
                            else:
                                self._save_error(results_dir, "vLLM server did not become ready (even with FP8 KV cache)")
                                self._collect_vllm_logs(results_dir)
                                self._stop_vllm()
                                self._upload_failed_benchmark(benchmark_name, model, results_dir, benchmark)
                                return False
                    else:
                        self._save_error(results_dir, "vLLM server did not become ready")
                        self._collect_vllm_logs(results_dir)
                        self._stop_vllm()
                        self._upload_failed_benchmark(benchmark_name, model, results_dir, benchmark)
                        return False
            else:
                self._save_error(results_dir, "Failed to start vLLM server")
                self._collect_vllm_logs(results_dir)
                self._upload_failed_benchmark(benchmark_name, model, results_dir, benchmark)
                return False

            if not vllm_ready:
                return False

            self.current_model = model

            # Store FP8 KV cache flag for metadata
            benchmark['_used_fp8_kv_cache'] = used_fp8_kv_cache

            # Extract LoRA adapter names for aiperf model switching
            lora_adapters = None
            if vllm_config.get('enable_lora') and vllm_config.get('lora_modules'):
                lora_adapters = [m['name'] for m in vllm_config['lora_modules']]

            # Step 4: Run aiperf benchmark
            if not self._run_aiperf(model, aiperf_config, results_dir, lora_adapters=lora_adapters):
                self._save_error(results_dir, "AIPerf benchmark failed")
                self._collect_vllm_logs(results_dir)
                self._stop_vllm()
                self._upload_failed_benchmark(benchmark_name, model, results_dir, benchmark)
                return False

            # Step 5: Collect results
            self._collect_results(results_dir)

            # Step 6: Stop vLLM server
            self._stop_vllm()

            # Step 7: Upload results to S3
            self._upload_benchmark_results(benchmark_name, model, results_dir, benchmark)

            return True

        except Exception as e:
            self.logger.error(f"Performance benchmark '{benchmark_name}' failed: {e}", exc_info=True)
            self._save_error(results_dir, str(e))
            self._collect_vllm_logs(results_dir)
            self._stop_vllm()
            self._upload_failed_benchmark(benchmark_name, model, results_dir, benchmark)
            return False

    def _run_quality_benchmark(self, benchmark_name: str, model: str, vllm_config: Dict[str, Any],
                               lm_eval_config: Dict[str, Any], results_dir: Path, benchmark: Dict[str, Any]) -> bool:
        """Run a quality benchmark using lm_eval (lm_eval manages vLLM internally)"""
        console.print(f"Running lm_eval quality benchmark: [yellow]{benchmark_name}[/yellow]")
        console.print(f"Model: [cyan]{model}[/cyan]")

        try:
            num_gpus = self.instance_config['gpu_count']

            # Build model_args for lm_eval vLLM backend
            model_args = f"pretrained={model}"
            model_args += f",tensor_parallel_size={num_gpus}"

            # Add vllm config to model_args
            if 'gpu_memory_utilization' in vllm_config:
                model_args += f",gpu_memory_utilization={vllm_config['gpu_memory_utilization']}"
            if 'max_model_len' in vllm_config:
                model_args += f",max_model_len={vllm_config['max_model_len']}"
            if 'dtype' in vllm_config:
                model_args += f",dtype={vllm_config['dtype']}"
            if 'max_num_seqs' in vllm_config:
                model_args += f",max_num_seqs={vllm_config['max_num_seqs']}"
            if 'enable_chunked_prefill' in vllm_config:
                model_args += f",enable_chunked_prefill={vllm_config['enable_chunked_prefill']}"
            if 'max_num_batched_tokens' in vllm_config:
                model_args += f",max_num_batched_tokens={vllm_config['max_num_batched_tokens']}"
            if 'enforce_eager' in vllm_config:
                model_args += f",enforce_eager={vllm_config['enforce_eager']}"

            # Add trust_remote_code for some models
            if lm_eval_config.get('trust_remote_code', True):
                model_args += ",trust_remote_code=True"

            # Get lm_eval parameters
            tasks = lm_eval_config.get('tasks', 'hellaswag')
            num_fewshot = lm_eval_config.get('num_fewshot', 0)
            batch_size = lm_eval_config.get('batch_size', 'auto')
            limit = lm_eval_config.get('limit', None)

            # Get timeout - default 3 hours for MMLU, 1 hour for others
            default_timeout = 10800 if 'mmlu' in tasks.lower() else 3600
            timeout = lm_eval_config.get('timeout', default_timeout)

            console.print(f"Tasks: [yellow]{tasks}[/yellow]")
            console.print(f"Few-shot: [yellow]{num_fewshot}[/yellow]")
            console.print(f"Tensor Parallel: [yellow]{num_gpus}[/yellow]")

            # HF token for gated models
            hf_token = os.getenv('HF_TOKEN')
            token_env = f"HF_TOKEN='{hf_token}' " if hf_token else ""

            # Build lm_eval command
            lm_eval_cmd = f"""{token_env}lm_eval --model vllm \\
  --model_args "{model_args}" \\
  --tasks {tasks} \\
  --num_fewshot {num_fewshot} \\
  --batch_size {batch_size} \\
  --output_path /tmp/lm_eval_results \\
  --log_samples"""

            if limit:
                lm_eval_cmd += f" \\\n  --limit {limit}"

            # Add any additional lm_eval parameters
            for key, value in lm_eval_config.items():
                if key not in ['tasks', 'num_fewshot', 'batch_size', 'limit', 'trust_remote_code', 'timeout']:
                    # Handle different value types
                    if isinstance(value, bool):
                        # Boolean flags: only add if True (e.g., --apply_chat_template)
                        if value:
                            lm_eval_cmd += f" \\\n  --{key}"
                    elif isinstance(value, str) and any(c in value for c in ' :<>/'):
                        # Quote strings with spaces, colons, or shell special chars
                        lm_eval_cmd += f' \\\n  --{key} "{value}"'
                    else:
                        lm_eval_cmd += f" \\\n  --{key} {value}"

            # Log the full command for debugging
            console.print(f"\n[dim]lm_eval command:[/dim]")
            console.print(f"[cyan]{lm_eval_cmd}[/cyan]\n")

            # Write lm_eval command to a script file to avoid quoting issues
            script_content = f"""#!/bin/bash
# Increase file descriptor limit to avoid "Too many open files" error
ulimit -n 65536 2>/dev/null || ulimit -n 4096 2>/dev/null || true

# Log the command being run
echo "=== LM_EVAL COMMAND ===" >> /tmp/lm_eval.log
echo '{lm_eval_cmd.replace("'", "'\"'\"'")}' >> /tmp/lm_eval.log
echo "=== END COMMAND ===" >> /tmp/lm_eval.log

{lm_eval_cmd} >> /tmp/lm_eval.log 2>&1
echo 'LMEVAL_SUCCESS' >> /tmp/lm_eval.log
"""
            # Clear previous log, create script, and run in background
            self.vast_manager.execute_command(self.instance_id, "rm -f /tmp/lm_eval.log && touch /tmp/lm_eval.log", quiet=True)

            # Use heredoc to write script
            write_script_cmd = f"""cat > /tmp/run_lm_eval.sh << 'SCRIPT_EOF'
{script_content}
SCRIPT_EOF
chmod +x /tmp/run_lm_eval.sh"""
            self.vast_manager.execute_command(self.instance_id, write_script_cmd, quiet=True)

            # Run the script in background
            self.vast_manager.execute_command(self.instance_id, "nohup /tmp/run_lm_eval.sh > /dev/null 2>&1 &", quiet=True)
            time.sleep(2)  # Give the process time to start

            # Stream logs while waiting for completion
            success = self._wait_for_lm_eval(tasks, timeout=timeout)

            if not success:
                console.print("[red]✗[/red] lm_eval benchmark failed")
                # Show last 50 lines of log
                console.print("\nLast 50 lines of lm_eval.log:")
                log_content = self.vast_manager.get_file_content(self.instance_id, "/tmp/lm_eval.log")
                if log_content:
                    for line in log_content.strip().split('\n')[-50:]:
                        console.print(f"  {line}")
                self._save_error(results_dir, "lm_eval benchmark failed")
                self._collect_lm_eval_logs(results_dir)
                self._upload_failed_benchmark(benchmark_name, model, results_dir, benchmark, benchmark_type="quality")
                return False

            console.print("[green]✓[/green] lm_eval benchmark completed")

            # Collect results
            self._collect_lm_eval_results(results_dir)

            # Upload results to S3
            self._upload_benchmark_results(benchmark_name, model, results_dir, benchmark, benchmark_type="quality")

            return True

        except Exception as e:
            self.logger.error(f"lm_eval benchmark failed: {e}", exc_info=True)
            self._save_error(results_dir, str(e))
            self._collect_lm_eval_logs(results_dir)
            self._upload_failed_benchmark(benchmark_name, model, results_dir, benchmark, benchmark_type="quality")
            return False

    def _wait_for_lm_eval(self, tasks: str, timeout: int = 3600) -> bool:
        """Wait for lm_eval to complete while streaming logs"""
        console.print("[dim]Streaming lm_eval output:[/dim]\n")

        check_interval = 5
        elapsed = 0
        log_position = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(f"Running lm_eval ({tasks})...", total=None)

            while elapsed < timeout:
                # Stream new log content
                new_content, log_position = self.vast_manager.tail_file(
                    self.instance_id, "/tmp/lm_eval.log", log_position
                )

                if new_content:
                    progress.stop()  # Temporarily stop progress for log output
                    for line in new_content.splitlines():
                        line = line.strip()
                        if line:
                            self._print_lm_eval_line(line)
                    progress.start()

                # Check if lm_eval completed
                if new_content and 'LMEVAL_SUCCESS' in new_content:
                    return True

                # Check for obvious failures
                if new_content and ('OutOfMemoryError' in new_content or 'CUDA out of memory' in new_content):
                    return False

                time.sleep(check_interval)
                elapsed += check_interval

        console.print(f"\n[red]✗[/red] lm_eval did not complete within {timeout}s")
        return False

    def _print_lm_eval_line(self, line: str):
        """Print an lm_eval log line with color coding"""
        if 'error' in line.lower() or 'exception' in line.lower() or 'traceback' in line.lower():
            console.print(f"  [red]{line}[/red]")
        elif 'OutOfMemory' in line or 'CUDA out of memory' in line:
            console.print(f"  [red bold]{line}[/red bold]")
        elif 'warning' in line.lower() or 'warn' in line.lower():
            console.print(f"  [yellow]{line}[/yellow]")
        elif 'acc' in line.lower() or 'score' in line.lower() or 'result' in line.lower():
            console.print(f"  [green]{line}[/green]")
        elif '%' in line:  # Progress indicators
            console.print(f"  [magenta]{line}[/magenta]")
        else:
            console.print(f"  [dim]{line}[/dim]")

    def _collect_lm_eval_logs(self, results_dir: Path):
        """Collect lm_eval logs"""
        console.print("Collecting lm_eval logs...")

        log_files = {
            '/tmp/lm_eval.log': 'lm_eval.log'
        }

        for remote_path, local_name in log_files.items():
            content = self.vast_manager.get_file_content(self.instance_id, remote_path)
            if content:
                (results_dir / local_name).write_text(content)

    def _collect_lm_eval_results(self, results_dir: Path):
        """Collect lm_eval results"""
        console.print("Collecting lm_eval results...")

        # Collect the log file first
        log_content = self.vast_manager.get_file_content(self.instance_id, "/tmp/lm_eval.log")
        if log_content:
            (results_dir / 'lm_eval.log').write_text(log_content)

        # lm_eval saves results in: /tmp/lm_eval_results/<model_name>/results_<timestamp>.json
        # Find the actual results file
        find_cmd = "find /tmp/lm_eval_results -name 'results_*.json' -type f 2>/dev/null | head -1"
        results_path = self.vast_manager.execute_command_with_output(self.instance_id, find_cmd, quiet=True)

        if results_path:
            results_path = results_path.strip()
            if results_path:
                console.print(f"  Found results at: {results_path}")
                content = self.vast_manager.get_file_content(self.instance_id, results_path)
                if content:
                    (results_dir / 'lm_eval_results.json').write_text(content)
                    console.print("[green]✓[/green] Collected lm_eval results")
                    return

        console.print("[yellow]⚠[/yellow] Could not find lm_eval results file")

        # Try to list what's in the results directory for debugging
        ls_cmd = "find /tmp/lm_eval_results -type f 2>/dev/null"
        files_output = self.vast_manager.execute_command_with_output(self.instance_id, ls_cmd, quiet=True)
        if files_output:
            console.print(f"  Files in results dir: {files_output.strip()}")

    def _download_lora_adapters(self, vllm_config: Dict[str, Any]) -> bool:
        """Download LoRA adapters from HuggingFace if needed"""
        if not vllm_config.get('enable_lora'):
            return True

        lora_modules = vllm_config.get('lora_modules', [])
        if not lora_modules:
            return True

        console.print("Downloading LoRA adapters from HuggingFace...")

        # Get HF token for private repos
        hf_token = os.getenv('HF_TOKEN')
        token_arg = f"--token {hf_token}" if hf_token else ""

        # Create base directory
        mkdir_cmd = f"mkdir -p {LORA_BASE_PATH}"
        self.vast_manager.execute_command(self.instance_id, mkdir_cmd, quiet=True)

        for adapter in lora_modules:
            adapter_name = adapter.get('name')
            lora_adapters = get_lora_adapters()
            if adapter_name not in lora_adapters:
                console.print(f"[yellow]⚠[/yellow] Unknown LoRA adapter: {adapter_name}, skipping")
                continue

            repo_id = lora_adapters[adapter_name]
            local_path = f"{LORA_BASE_PATH}/{adapter_name}"

            console.print(f"  Downloading {repo_id} -> {local_path}")

            download_cmd = f"huggingface-cli download {repo_id} --local-dir {local_path} {token_arg}"
            result = self.vast_manager.execute_command(self.instance_id, download_cmd)

            if not result:
                console.print(f"[red]✗[/red] Failed to download LoRA adapter: {adapter_name}")
                return False

            console.print(f"  [green]✓[/green] Downloaded {adapter_name}")

        return True

    def _start_vllm(self, model: str, vllm_config: Dict[str, Any], concurrency: int = None) -> bool:
        """Start vLLM server with specified configuration (pass-through any vLLM args)

        Args:
            concurrency: If provided and max_num_seqs not in vllm_config, sets --max-num-seqs to this value
        """

        # Download LoRA adapters if needed (before starting vLLM)
        if vllm_config.get('enable_lora'):
            if not self._download_lora_adapters(vllm_config):
                return False

        console.print("Starting vLLM server...")

        try:
            num_gpus = self.instance_config['gpu_count']

            # HF token for gated models
            hf_token = os.getenv('HF_TOKEN')
            token_env = f"HF_TOKEN='{hf_token}' " if hf_token else ""

            # Build vLLM arguments from config (pass-through any parameter)
            vllm_args = []
            for key, value in vllm_config.items():
                # Convert underscores to hyphens for CLI args
                arg_name = key.replace('_', '-')

                # Handle lora_modules specially - needs name=path format with updated paths
                if key == 'lora_modules':
                    lora_args = []
                    for adapter in value:
                        adapter_name = adapter.get('name')
                        # Use the standard download path instead of config path
                        local_path = f"{LORA_BASE_PATH}/{adapter_name}"
                        lora_args.append(f"{adapter_name}={local_path}")
                    vllm_args.append(f"--lora-modules {' '.join(lora_args)}")
                elif isinstance(value, bool):
                    if value:
                        vllm_args.append(f"--{arg_name}")
                elif isinstance(value, dict):
                    # Dicts become JSON strings (e.g., chat-template-kwargs)
                    import json
                    json_value = json.dumps(value).replace('"', '\\"')
                    vllm_args.append(f"--{arg_name} \"{json_value}\"")
                elif isinstance(value, list):
                    # Lists become comma-separated
                    vllm_args.append(f"--{arg_name} {','.join(map(str, value))}")
                elif value is not None:
                    vllm_args.append(f"--{arg_name} {value}")

            # Add tensor-parallel-size if not specified
            if 'tensor_parallel_size' not in vllm_config and 'tensor-parallel-size' not in vllm_config:
                vllm_args.append(f"--tensor-parallel-size {num_gpus}")

            # Auto-set max-num-seqs to match aiperf concurrency (avoids memory issues and request queuing)
            if concurrency and 'max_num_seqs' not in vllm_config and 'max-num-seqs' not in vllm_config:
                vllm_args.append(f"--max-num-seqs {concurrency}")

            vllm_args_str = " \\\n  ".join(vllm_args)

            vllm_cmd = f"""
{token_env}python3 -m vllm.entrypoints.openai.api_server \\
  --model {model} \\
  --host 0.0.0.0 \\
  --port 8000 \\
  {vllm_args_str} \\
  > /tmp/vllm.log 2>&1 &
"""

            self.vast_manager.execute_command(self.instance_id, vllm_cmd, background=True)

            # Display key config for user
            display_config = {k: v for k, v in vllm_config.items() if k in ['max_model_len', 'gpu_memory_utilization', 'dtype', 'quantization', 'max_num_seqs']}
            # Show auto-set max_num_seqs if not in config
            if concurrency and 'max_num_seqs' not in vllm_config and 'max-num-seqs' not in vllm_config:
                display_config['max_num_seqs'] = f"{concurrency} (from concurrency)"
            console.print(f"[green]✓[/green] vLLM server started {display_config}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start vLLM: {e}", exc_info=True)
            return False

    def _wait_for_vllm(self, timeout: int = 600) -> bool:
        """Wait for vLLM server to be ready while streaming logs"""
        console.print("Waiting for vLLM server to be ready...")
        console.print("[dim]Streaming vLLM logs:[/dim]\n")

        check_interval = 5
        elapsed = 0
        log_position = 0

        while elapsed < timeout:
            # Stream new log content
            new_content, log_position = self.vast_manager.tail_file(
                self.instance_id, "/tmp/vllm.log", log_position
            )

            if new_content:
                # Print each new line with formatting
                for line in new_content.splitlines():
                    line = line.strip()
                    if line:
                        self._print_log_line(line)

            # Check if vLLM is ready (quiet=True to suppress expected errors during startup)
            check_cmd = "curl -f -s http://localhost:8000/v1/models -o /dev/null && echo 'SUCCESS'"
            if self.vast_manager.execute_command(self.instance_id, check_cmd, quiet=True):
                console.print(f"\n[green]✓[/green] vLLM server is ready! ({elapsed}s)")
                return True

            time.sleep(check_interval)
            elapsed += check_interval

            # Show progress every 30 seconds if no log activity
            if elapsed % 30 == 0 and not new_content:
                console.print(f"  [dim]... waiting ({elapsed}/{timeout}s)[/dim]")

        console.print(f"\n[red]✗[/red] vLLM server did not become ready within {timeout}s")
        return False

    def _print_log_line(self, line: str):
        """Print a log line with color coding based on content"""
        if 'error' in line.lower() or 'exception' in line.lower() or 'traceback' in line.lower():
            console.print(f"  [red]{line}[/red]")
        elif 'warning' in line.lower() or 'warn' in line.lower():
            console.print(f"  [yellow]{line}[/yellow]")
        elif 'info' in line.lower() or 'starting' in line.lower() or 'loaded' in line.lower():
            console.print(f"  [cyan]{line}[/cyan]")
        elif 'ready' in line.lower() or 'uvicorn' in line.lower() or 'application startup' in line.lower():
            console.print(f"  [green]{line}[/green]")
        elif '%' in line:  # Progress indicators (model loading)
            console.print(f"  [magenta]{line}[/magenta]")
        else:
            console.print(f"  [dim]{line}[/dim]")

    def _stop_vllm(self):
        """Stop the vLLM server"""
        console.print("Stopping vLLM server...")
        self.vast_manager.execute_command(
            self.instance_id,
            "pkill -f 'vllm.entrypoints' || true",
            quiet=True
        )
        time.sleep(2)  # Give it time to shutdown

    def _check_vllm_oom(self) -> bool:
        """Check if vLLM failed due to out of memory error"""
        try:
            # execute_command returns bool (True if exit code 0, False otherwise)
            # grep returns exit code 0 if matches found, 1 if no matches
            return self.vast_manager.execute_command(
                self.instance_id,
                "grep -i -E 'out of memory|OOM|CUDA error|torch.cuda.OutOfMemoryError|cannot allocate|larger than the available KV cache memory' /tmp/vllm.log 2>/dev/null | head -5"
            )
        except Exception:
            return False

    def _run_aiperf(self, model: str, aiperf_config: Dict[str, Any], results_dir: Path, lora_adapters: list = None) -> bool:
        """Run AIPerf benchmark (pass-through any aiperf args)

        Args:
            lora_adapters: List of LoRA adapter names for model switching (e.g., ['customer-support-faq', 'technical-docs'])
        """
        console.print("Running AIPerf benchmark...")

        try:
            # Build aiperf arguments from config (pass-through any parameter)
            aiperf_args = []

            # Set defaults for required args if not provided
            defaults = {
                'url': 'http://localhost:8000',
                'output_artifact_dir': '/tmp/aiperf_results',
                'random_seed': 42,
                'request_timeout_seconds': 120
            }

            # Merge defaults with user config (user config takes precedence)
            merged_config = {**defaults, **aiperf_config}

            for key, value in merged_config.items():
                # Convert underscores to hyphens for CLI args
                arg_name = key.replace('_', '-')

                if isinstance(value, bool):
                    if value:
                        aiperf_args.append(f"--{arg_name}")
                elif isinstance(value, dict):
                    # Dicts become JSON strings (e.g., extra_inputs for chat_template_kwargs)
                    import json
                    json_value = json.dumps(value)
                    aiperf_args.append(f"--{arg_name} '{json_value}'")
                elif isinstance(value, list):
                    # Lists become comma-separated (e.g., concurrency: [1, 4, 8])
                    aiperf_args.append(f"--{arg_name} {','.join(map(str, value))}")
                elif value is not None:
                    aiperf_args.append(f"--{arg_name} {value}")

            aiperf_args_str = " \\\n  ".join(aiperf_args)

            # Display key config for user
            display_keys = ['concurrency', 'input_sequence_length', 'output_sequence_length', 'request_count']
            for key in display_keys:
                if key in aiperf_config:
                    console.print(f"  {key}: [yellow]{aiperf_config[key]}[/yellow]")

            # Start GPU monitoring using DCGM
            console.print("Starting GPU telemetry collection...")

            # Check if nv-hostengine is running, start if not (must background it), then start dcgmi dmon
            gpu_cmd = "pgrep -x nv-hostengine >/dev/null || (nv-hostengine &); sleep 2; dcgmi dmon -e 150,155,156,203,204,240,241 -d 1000 > /tmp/gpu_metrics.log 2>&1 &"
            self.vast_manager.execute_command(self.instance_id, gpu_cmd)
            time.sleep(1)

            # Build model arguments - include LoRA adapters if provided (comma-separated)
            if lora_adapters:
                # Multiple models: use --model-names with comma-separated list
                # Also specify tokenizer since LoRA adapter names aren't HuggingFace models
                # (they share the base model's tokenizer)
                all_models = [model] + lora_adapters
                model_args = f'--model-names "{",".join(all_models)}" --tokenizer "{model}"'
                console.print(f"  LoRA adapters for switching: [yellow]{', '.join(lora_adapters)}[/yellow]")
            else:
                # Single model: use --model (original behavior)
                model_args = f'--model "{model}"'

            # Build aiperf command - write to script file to avoid quoting issues with nohup
            aiperf_script = f"""#!/bin/bash
cd /tmp
aiperf profile \\
  {model_args} \\
  {aiperf_args_str} \\
  > /tmp/aiperf.log 2>&1
if [ $? -eq 0 ]; then
    echo "AIPERF_SUCCESS" >> /tmp/aiperf.log
else
    echo "AIPERF_FAILED" >> /tmp/aiperf.log
fi
"""
            # Write script, make executable, and run with nohup
            import base64
            script_b64 = base64.b64encode(aiperf_script.encode()).decode()
            setup_cmd = f"echo '{script_b64}' | base64 -d > /tmp/run_aiperf.sh && chmod +x /tmp/run_aiperf.sh"
            self.vast_manager.execute_command(self.instance_id, setup_cmd, quiet=True)

            # Start script in background with nohup
            start_cmd = "nohup /tmp/run_aiperf.sh > /dev/null 2>&1 & echo $! > /tmp/aiperf.pid"
            self.vast_manager.execute_command(self.instance_id, start_cmd, background=False)
            time.sleep(3)  # Give it time to start

            # Verify aiperf started
            pid_check = self.vast_manager.execute_command_with_output(
                self.instance_id, "cat /tmp/aiperf.pid 2>/dev/null && ps -p $(cat /tmp/aiperf.pid) >/dev/null 2>&1 && echo 'STARTED' || echo 'NOT_STARTED'", quiet=True
            )
            if not pid_check or 'NOT_STARTED' in pid_check:
                console.print("[red]✗[/red] AIPerf failed to start")
                log_content = self.vast_manager.get_file_content(self.instance_id, "/tmp/aiperf.log")
                if log_content:
                    console.print(f"[dim]Log output:[/dim]\n{log_content[:1000]}")
                # Also show the script for debugging
                script_content = self.vast_manager.get_file_content(self.instance_id, "/tmp/run_aiperf.sh")
                if script_content:
                    console.print(f"[dim]Script:[/dim]\n{script_content[:500]}")
                return False

            console.print(f"[green]✓[/green] AIPerf started (PID: {pid_check.split()[0] if pid_check else 'unknown'})")

            # Poll for completion instead of waiting on SSH
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Running benchmark...", total=None)

                max_wait = 7200  # 2 hours max
                poll_interval = 10  # Check every 10 seconds
                waited = 0
                success = False

                ssh_retries = 0
                max_ssh_retries = 5

                while waited < max_wait:
                    time.sleep(poll_interval)
                    waited += poll_interval

                    try:
                        # Check if aiperf is still running
                        check_cmd = "ps -p $(cat /tmp/aiperf.pid 2>/dev/null) >/dev/null 2>&1 && echo 'RUNNING' || echo 'DONE'"
                        status = self.vast_manager.execute_command_with_output(self.instance_id, check_cmd, quiet=True)

                        if status is None:
                            raise Exception("SSH command returned None")

                        ssh_retries = 0  # Reset on success

                        if 'DONE' in status:
                            # Check if it succeeded
                            log_tail = self.vast_manager.execute_command_with_output(
                                self.instance_id, "tail -5 /tmp/aiperf.log", quiet=True
                            )
                            if log_tail and 'AIPERF_SUCCESS' in log_tail:
                                success = True
                            break

                        # Update progress with elapsed time and show recent log activity
                        progress.update(task, description=f"Running benchmark... ({waited}s)")

                        # Every 60 seconds, show last line of log for progress visibility
                        if waited % 60 == 0:
                            log_line = self.vast_manager.execute_command_with_output(
                                self.instance_id, "tail -1 /tmp/aiperf.log 2>/dev/null | head -c 100", quiet=True
                            )
                            if log_line and log_line.strip():
                                console.print(f"  [dim]Log: {log_line.strip()[:80]}...[/dim]")

                    except Exception as e:
                        ssh_retries += 1
                        if ssh_retries >= max_ssh_retries:
                            console.print(f"\n[red]SSH connection failed {max_ssh_retries} times, giving up[/red]")
                            break
                        console.print(f"\n[yellow]SSH hiccup ({ssh_retries}/{max_ssh_retries}), retrying in 30s...[/yellow]")
                        time.sleep(30)  # Wait before retry

                # Stop GPU monitoring
                self.vast_manager.execute_command(self.instance_id, "pkill -f 'dcgmi dmon' || true", quiet=True)

            if not success:
                console.print("[red]✗[/red] AIPerf benchmark failed")
                # Collect aiperf.log for debugging
                aiperf_log = self.vast_manager.get_file_content(self.instance_id, "/tmp/aiperf.log")
                if aiperf_log:
                    # Save locally
                    aiperf_log_path = results_dir / "aiperf.log"
                    with open(aiperf_log_path, 'w') as f:
                        f.write(aiperf_log)
                    console.print(f"  ⚠ AIPerf log saved to {aiperf_log_path}")
                    # Print last 50 lines for debugging
                    log_lines = aiperf_log.strip().split('\n')
                    console.print(f"  [dim]Last {min(50, len(log_lines))} lines of aiperf.log:[/dim]")
                    for line in log_lines[-50:]:
                        console.print(f"    [dim]{line}[/dim]")
                return False

            console.print("[green]✓[/green] AIPerf benchmark completed")
            return True

        except Exception as e:
            self.logger.error(f"AIPerf failed: {e}", exc_info=True)
            return False

    def _collect_results(self, results_dir: Path):
        """Collect benchmark results from the instance"""
        console.print("Collecting results...")

        # Discover aiperf output files
        find_cmd = "find /tmp/aiperf_results -type f 2>/dev/null"
        found_files = self.vast_manager.run_command(self.instance_id, find_cmd)

        files_to_download = {
            '/tmp/aiperf.log': 'aiperf.log',
            '/tmp/vllm.log': 'vllm.log',
            '/tmp/gpu_metrics.log': 'gpu_metrics.log'
        }

        if found_files:
            for remote_file in found_files.strip().split('\n'):
                if remote_file and remote_file.strip():
                    local_name = remote_file.split('/')[-1]
                    files_to_download[remote_file.strip()] = local_name

        # Use SFTP download (much faster than cat for large files)
        for remote_path, local_name in files_to_download.items():
            local_path = results_dir / local_name
            success = self.vast_manager.download_file(self.instance_id, remote_path, str(local_path))

            if success:
                console.print(f"  [green]✓[/green] {local_name}")
            else:
                console.print(f"  [yellow]⚠[/yellow] {local_name} (not found)")

        # Cleanup remote results
        self.vast_manager.execute_command(self.instance_id, "rm -rf /tmp/aiperf_results /tmp/aiperf.log /tmp/gpu_metrics.log")

    def _collect_vllm_logs(self, results_dir: Path):
        """Collect vLLM logs for debugging"""
        content = self.vast_manager.get_file_content(self.instance_id, "/tmp/vllm.log")
        if content:
            (results_dir / 'vllm.log').write_text(content)
            console.print("  [yellow]⚠[/yellow] vLLM logs saved for debugging")

    def _save_error(self, results_dir: Path, error_message: str):
        """Save error information"""
        error_info = {
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        }
        (results_dir / 'error.json').write_text(json.dumps(error_info, indent=2))

    def _upload_benchmark_results(self, name: str, model: str, results_dir: Path, benchmark: Dict[str, Any], benchmark_type: str = "performance"):
        """Upload results for a single benchmark"""
        console.print("Uploading results to S3...")

        metadata = {
            'benchmark_name': name,
            'model': model,
            'suite_name': self.suite_name,
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'gpu': f"{self.instance_config['gpu_type']} x{self.instance_config['gpu_count']}",
            'instance_id': self.instance_id,
            'vllm_config': benchmark.get('vllm', {}),
            'aiperf_config': benchmark.get('aiperf', {}),
            'lm_eval_config': benchmark.get('lm_eval', {}),
            'benchmark_type': benchmark_type,
            'status': 'success',
            'used_fp8_kv_cache': benchmark.get('_used_fp8_kv_cache', False)
        }

        success = self.result_uploader.upload_results(
            results_dir,
            model,
            metadata,
            benchmark_type=benchmark_type
        )

        if success:
            console.print("[green]✓[/green] Results uploaded to S3")
        else:
            console.print("[yellow]⚠[/yellow] Failed to upload results to S3")

    def _upload_failed_benchmark(self, name: str, model: str, results_dir: Path, benchmark: Dict[str, Any], benchmark_type: str = "performance"):
        """Upload failed benchmark results (error logs) to S3"""
        console.print("Uploading error logs to S3...")

        metadata = {
            'benchmark_name': name,
            'model': model,
            'suite_name': self.suite_name,
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'gpu': f"{self.instance_config['gpu_type']} x{self.instance_config['gpu_count']}",
            'instance_id': self.instance_id,
            'vllm_config': benchmark.get('vllm', {}),
            'aiperf_config': benchmark.get('aiperf', {}),
            'lm_eval_config': benchmark.get('lm_eval', {}),
            'benchmark_type': benchmark_type,
            'status': 'failed'
        }

        success = self.result_uploader.upload_results(
            results_dir,
            model,
            metadata,
            benchmark_type=benchmark_type
        )

        if success:
            console.print("[yellow]⚠[/yellow] Error logs uploaded to S3")
        else:
            console.print("[red]✗[/red] Failed to upload error logs to S3")

    def _maybe_cleanup_model(self, current_benchmark: Dict[str, Any], current_index: int):
        """Cleanup model if the next benchmark uses a different model"""
        if current_index >= len(self.benchmarks) - 1:
            return  # Last benchmark, cleanup will happen at the end

        next_benchmark = self.benchmarks[current_index + 1]
        current_model = current_benchmark['model']
        next_model = next_benchmark['model']

        if current_model != next_model:
            console.print(f"\nModel change detected: {current_model} -> {next_model}")
            console.print("Cleaning up current model to free space...")

            # Try multiple possible cache locations
            cleanup_commands = [
                f"rm -rf ~/.cache/huggingface/hub/models--{current_model.replace('/', '--')}",
                f"rm -rf /root/.cache/huggingface/hub/models--{current_model.replace('/', '--')}",
                "sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true"
            ]

            for cmd in cleanup_commands:
                self.vast_manager.execute_command(self.instance_id, cmd)

            self.current_model = None
            console.print("[green]✓[/green] Model cache cleaned up")

    def _generate_power_analysis(self):
        """Generate power consumption analysis for all benchmarks"""
        console.print("\nGenerating power consumption analysis...")

        try:
            gpu_name = f"{self.instance_config['gpu_type']}"
            report = generate_power_report(
                self.results_base_dir,
                self.benchmarks,
                gpu_name=gpu_name
            )

            if report.get("benchmarks"):
                # Save locally
                report_path = self.results_base_dir / "power_analysis.json"
                save_power_report(report, report_path)

                # Print summary to console
                summary_text = print_power_summary(report)
                console.print(f"\n[dim]{summary_text}[/dim]")

                # Upload to S3
                try:
                    safe_suite_name = self._sanitize_name(self.suite_name)
                    s3_key = f"{os.getenv('S3_RESULTS_PREFIX', 'benchmarks/')}suites/{safe_suite_name}/{self.run_id}/power_analysis.json"
                    self.result_uploader._upload_file(report_path, s3_key)
                    console.print(f"[green]✓[/green] Power analysis uploaded to S3")
                except Exception as e:
                    console.print(f"[yellow]⚠[/yellow] Failed to upload power analysis: {e}")

                console.print(f"[green]✓[/green] Power analysis saved to {report_path}")
            else:
                console.print("[yellow]⚠[/yellow] No power metrics available for analysis")

        except Exception as e:
            self.logger.error(f"Failed to generate power analysis: {e}", exc_info=True)
            console.print(f"[yellow]⚠[/yellow] Power analysis failed: {e}")

    def _upload_suite_summary(self):
        """Upload summary of all benchmarks in the suite"""
        console.print("\nUploading suite summary...")

        summary = {
            'suite_name': self.suite_name,
            'run_id': self.run_id,
            'gpu': f"{self.instance_config['gpu_type']} x{self.instance_config['gpu_count']}",
            'instance_id': self.instance_id,
            'total_benchmarks': len(self.benchmarks),
            'successful': sum(1 for r in self.benchmark_results if r['success']),
            'failed': sum(1 for r in self.benchmark_results if not r['success']),
            'results': self.benchmark_results,
            'completed_at': datetime.now().isoformat()
        }

        # Save locally
        summary_file = self.results_base_dir / 'suite_summary.json'
        summary_file.write_text(json.dumps(summary, indent=2))

        # Upload to S3
        try:
            safe_suite_name = self._sanitize_name(self.suite_name)
            s3_key = f"{os.getenv('S3_RESULTS_PREFIX', 'benchmarks/')}suites/{safe_suite_name}/{self.run_id}/suite_summary.json"

            self.result_uploader._upload_file(summary_file, s3_key)
            console.print(f"[green]✓[/green] Suite summary uploaded")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Failed to upload suite summary: {e}")

    def _display_final_summary(self, elapsed: float):
        """Display final summary of the benchmark suite"""
        successful = sum(1 for r in self.benchmark_results if r['success'])
        failed = sum(1 for r in self.benchmark_results if not r['success'])

        # Create summary table
        table = Table(title="Benchmark Suite Results")
        table.add_column("Benchmark", style="cyan")
        table.add_column("Model", style="yellow")
        table.add_column("Status", style="green")

        for result in self.benchmark_results:
            status = "[green]✓ Success[/green]" if result['success'] else "[red]✗ Failed[/red]"
            table.add_row(result['name'], result['model'], status)

        console.print("\n")
        console.print(table)

        # Summary stats
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Total benchmarks: {len(self.benchmark_results)}")
        console.print(f"  Successful: [green]{successful}[/green]")
        console.print(f"  Failed: [red]{failed}[/red]")
        console.print(f"  Total time: {format_duration(elapsed)}")
        console.print(f"  Results: {self.results_base_dir}")

        if failed == 0:
            console.print("\n[bold green]✓ All benchmarks completed successfully![/bold green]")
        else:
            console.print(f"\n[bold yellow]⚠ {failed} benchmark(s) failed[/bold yellow]")

    def _cleanup(self):
        """Cleanup resources"""
        if self.instance_id:
            # Don't terminate if using existing instance via --instance-id
            if self.args.instance_id:
                console.print(f"\n[bold]Cleanup: Keeping existing instance {self.instance_id} (--instance-id mode)[/bold]")
                console.print("[yellow]Remember to manually terminate the instance when done![/yellow]")
                return

            console.print("\n[bold]Cleanup: Terminating Vast.ai instance[/bold]")
            try:
                self.vast_manager.destroy_instance(self.instance_id)
                console.print(f"[green]✓[/green] Instance {self.instance_id} terminated")
            except Exception as e:
                self.logger.error(f"Failed to destroy instance: {e}", exc_info=True)
                console.print(f"[red]✗[/red] Failed to destroy instance: {e}")


class BenchmarkRunner:
    """Main orchestrator for single LLM benchmark on Vast.ai (legacy compatibility)"""

    def __init__(self, config: Dict[str, Any], model_name: str, args: argparse.Namespace):
        """
        Initialize the benchmark runner.

        Args:
            config: Configuration dictionary from YAML
            model_name: HuggingFace model name to benchmark
            args: Command-line arguments
        """
        self.config = config
        self.model_name = model_name
        self.args = args
        self.logger = logging.getLogger(__name__)
        self.local_mode = args.local

        # Initialize managers (only if not in local mode)
        if not self.local_mode:
            self.vast_manager = VastManager(config, self.logger)
        self.result_uploader = ResultUploader(config, self.logger)

        # Runtime state
        self.instance_id: Optional[int] = None
        self.instance_ip: Optional[str] = None
        self.results_dir = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Local mode state
        self.vllm_container_name: Optional[str] = None
        self.docker_image = config.get('vast', {}).get('image', 'holtmann/llm-benchmark:latest')

        # Track S3 path for download
        self.s3_results_path: Optional[str] = None

    def run(self) -> bool:
        """
        Execute the complete benchmark workflow.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get actual GPU config (CLI args override config, use defaults for local mode)
            if self.local_mode:
                gpu_name = self.args.gpu_name or "Local GPU"
                num_gpus = self.args.num_gpus or 1
            else:
                gpu_name = self.args.gpu_name or self.config['vast']['gpu_name']
                num_gpus = self.args.num_gpus or self.config['vast']['num_gpus']

            benchmark_type = self.args.benchmark_type
            mode_label = "Local" if self.local_mode else "Vast.ai"
            console.print(Panel.fit(
                f"[bold cyan]LLM Benchmark Runner ({mode_label})[/bold cyan]\n"
                f"Model: [yellow]{self.model_name}[/yellow]\n"
                f"GPU: [green]{gpu_name} x{num_gpus}[/green]\n"
                f"Benchmark: [magenta]{benchmark_type}[/magenta]",
                border_style="cyan"
            ))

            # Step 1: Provision instance (skip for local mode)
            if not self.local_mode:
                if not self._provision_instance():
                    return False

            # Step 2: Setup environment (skip for local mode - assume Docker is available)
            if not self.local_mode:
                if not self._setup_environment():
                    return False

            # Branch based on benchmark type
            if benchmark_type == "performance":
                # Performance benchmark workflow (AIPerf)
                # Step 3: Start vLLM server
                if not self._start_vllm():
                    return False

                # Step 4: Wait for vLLM to be ready
                if not self._wait_for_vllm():
                    return False

                # Step 5: Run AIPerf benchmark
                if not self._run_benchmark():
                    return False
            else:
                # Quality benchmark workflow (lm_eval)
                # lm_eval will start vLLM itself, so we skip manual startup
                # Step 3-5: Run lm_eval benchmark (includes vLLM startup)
                if not self._run_lm_eval():
                    return False

            # Step 6: Collect and process results
            if not self._collect_results():
                return False

            # Analyze GPU metrics if available
            self._analyze_gpu_metrics()

            # Step 7: Upload results to S3
            if not self._upload_results():
                return False

            # Step 8: Download results to local directory (if requested)
            if self.args.download_results:
                if not self._download_results():
                    console.print("[yellow]⚠[/yellow] Results uploaded but local download failed")
                    # Don't fail the whole benchmark if download fails

            console.print("\n[bold green]✓ Benchmark completed successfully![/bold green]")
            return True

        except KeyboardInterrupt:
            console.print("\n[bold red]✗ Benchmark interrupted by user[/bold red]")
            return False
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}", exc_info=True)
            console.print(f"\n[bold red]✗ Benchmark failed: {e}[/bold red]")
            return False
        finally:
            # Always cleanup
            self._cleanup()

    def _provision_instance(self) -> bool:
        """Provision a Vast.ai instance"""
        console.print("\n[bold]Step 1: Provisioning Vast.ai instance[/bold]")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Searching for available instances...", total=None)

                # Create instance
                instance_id, instance_ip = self.vast_manager.create_instance(
                    gpu_name=self.args.gpu_name or self.config['vast']['gpu_name'],
                    num_gpus=self.args.num_gpus or self.config['vast']['num_gpus'],
                    disk_space=self.config['vast']['disk_space'],
                    image=self.config['vast']['image']
                )

                self.instance_id = instance_id
                self.instance_ip = instance_ip

                progress.update(task, description=f"Instance {instance_id} created at {instance_ip}")

            console.print(f"[green]✓[/green] Instance provisioned: ID={instance_id}, IP={instance_ip}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to provision instance: {e}", exc_info=True)
            console.print(f"[red]✗[/red] Failed to provision instance: {e}")
            return False

    def _setup_environment(self) -> bool:
        """Setup environment and install dependencies on the instance"""
        console.print("\n[bold]Step 2: Setting up environment[/bold]")

        try:
            # Wait for SSH to be available
            ssh_timeout = self.config['vast'].get('ssh_timeout', 900)
            console.print(f"Waiting for SSH connection (timeout: {ssh_timeout}s)...")
            if not self.vast_manager.wait_for_ssh(self.instance_id, timeout=ssh_timeout):
                console.print("[red]✗[/red] SSH connection timeout")
                return False

            # Check if using pre-built image with packages already installed
            prebuilt = self.config['vast'].get('prebuilt_image', False)

            if prebuilt:
                console.print("Using pre-built Docker image - skipping package installation")

                # Fix libcuda.so symlink if missing (needed for Triton JIT on some vast.ai hosts)
                # Some newer GPU hosts (e.g., RTX 5090) may be missing this symlink
                console.print("Ensuring CUDA libraries are properly linked...")
                self.vast_manager.execute_command(
                    self.instance_id,
                    "ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so 2>/dev/null || true"
                )

                # Increase file descriptor limits to avoid "Too many open files" during model downloads
                console.print("Setting file descriptor limits...")
                self.vast_manager.execute_command(
                    self.instance_id,
                    "echo '* soft nofile 65536' >> /etc/security/limits.conf; echo '* hard nofile 65536' >> /etc/security/limits.conf; ulimit -n 65536 2>/dev/null || true",
                    quiet=True
                )

                console.print("Verifying vLLM, AIPerf, and lm_eval are available...")

                # Just verify installations
                verify_commands = [
                    "python3 -m vllm.entrypoints.openai.api_server --help > /dev/null 2>&1",
                    "aiperf --help > /dev/null 2>&1",
                    "lm_eval --help > /dev/null 2>&1"
                ]

                for cmd in verify_commands:
                    success = self.vast_manager.execute_command(self.instance_id, cmd)
                    if not success:
                        console.print(f"[red]✗[/red] Verification failed: {cmd}")
                        console.print("[yellow]⚠[/yellow] Make sure you're using the correct pre-built image")
                        return False

                console.print("[green]✓[/green] Pre-built environment verified")
                return True

            # Base image - need to install everything
            console.print("Installing vLLM, AIPerf, lm_eval, and dependencies...")

            setup_commands = [
                # Update package lists
                "apt-get update",
                # Install Python and pip
                "apt-get install -y python3-pip python3-venv curl wget",
                # Install NVIDIA DCGM for GPU telemetry
                "distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\\.//g') && wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb",
                "dpkg -i cuda-keyring_1.0-1_all.deb",
                "apt-get update",
                "apt-get install -y datacenter-gpu-manager",
                # Start DCGM service
                "systemctl --now enable nvidia-dcgm || nv-hostengine -n &",
                # Install dcgm-exporter for metrics collection
                "pip3 install dcgm-exporter || echo 'dcgm-exporter install skipped'",
                # Install vLLM
                "pip3 install vllm",
                # Install AIPerf
                "pip3 install aiperf",
                # Install lm-evaluation-harness
                "pip3 install lm-eval",
                # Install boto3 for S3 uploads
                "pip3 install boto3",
                # Verify installations
                "python3 -m vllm.entrypoints.openai.api_server --help > /dev/null 2>&1",
                "aiperf --help > /dev/null 2>&1",
                "lm_eval --help > /dev/null 2>&1",
                "dcgmi discovery -l || echo 'DCGM check skipped'"
            ]

            for cmd in setup_commands:
                self.logger.debug(f"Executing: {cmd}")
                success = self.vast_manager.execute_command(self.instance_id, cmd)
                if not success and "verify" not in cmd.lower():
                    console.print(f"[yellow]⚠[/yellow] Warning: Command may have failed: {cmd}")

            console.print("[green]✓[/green] Environment setup complete")
            return True

        except Exception as e:
            self.logger.error(f"Environment setup failed: {e}", exc_info=True)
            console.print(f"[red]✗[/red] Environment setup failed: {e}")
            return False

    def _start_vllm(self) -> bool:
        """Start the vLLM server (either on vast.ai instance or locally via Docker)"""
        console.print("\n[bold]Step 3: Starting vLLM server[/bold]")

        try:
            # Build vLLM command
            vllm_config = self.config['vllm']
            num_gpus = self.args.num_gpus or (1 if self.local_mode else self.config['vast']['num_gpus'])

            console.print(f"Model: [yellow]{self.model_name}[/yellow]")
            console.print(f"Tensor Parallel Size: [yellow]{num_gpus}[/yellow]")

            if self.local_mode:
                # Local mode: Start vLLM in Docker container
                import subprocess

                self.vllm_container_name = f"vllm-{int(time.time())}"

                docker_cmd = [
                    "docker", "run", "-d",
                    "--name", self.vllm_container_name,
                    "--gpus", "all",
                    "-p", f"{vllm_config['port']}:{vllm_config['port']}",
                    self.docker_image,
                    "python3", "-m", "vllm.entrypoints.openai.api_server",
                    "--model", self.model_name,
                    "--host", "0.0.0.0",
                    "--port", str(vllm_config['port']),
                    "--gpu-memory-utilization", str(vllm_config['gpu_memory_utilization']),
                    "--max-model-len", str(vllm_config.get('max_model_len', 4096)),
                    "--tensor-parallel-size", str(num_gpus)
                ]

                # Add extra args if provided
                if self.args.vllm_args:
                    docker_cmd.extend(self.args.vllm_args.split())

                result = subprocess.run(docker_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.error(f"Failed to start Docker container: {result.stderr}")
                    console.print(f"[red]✗[/red] Failed to start vLLM Docker container")
                    return False

                console.print(f"[green]✓[/green] vLLM server started in Docker (container: {self.vllm_container_name})")
            else:
                # Vast.ai mode: Execute command on remote instance
                vllm_cmd = f"""
python3 -m vllm.entrypoints.openai.api_server \\
  --model {self.model_name} \\
  --host {vllm_config['host']} \\
  --port {vllm_config['port']} \\
  --gpu-memory-utilization {vllm_config['gpu_memory_utilization']} \\
  --max-model-len {vllm_config.get('max_model_len', 4096)} \\
  --tensor-parallel-size {num_gpus} \\
  {self.args.vllm_args or vllm_config.get('extra_args', '')} \\
  > /tmp/vllm.log 2>&1 &
"""

                self.vast_manager.execute_command(
                    self.instance_id,
                    vllm_cmd,
                    background=True
                )

                console.print("[green]✓[/green] vLLM server started (running in background)")

            return True

        except Exception as e:
            self.logger.error(f"Failed to start vLLM: {e}", exc_info=True)
            console.print(f"[red]✗[/red] Failed to start vLLM: {e}")
            return False

    def _wait_for_vllm(self) -> bool:
        """Wait for vLLM server to be ready by checking /models endpoint while streaming logs"""
        console.print("\n[bold]Step 4: Waiting for vLLM server to be ready[/bold]")
        console.print("[dim]Streaming vLLM logs:[/dim]\n")

        timeout = self.config['vllm']['startup_timeout']
        check_interval = 5
        elapsed = 0
        log_position = 0

        vllm_port = self.config['vllm']['port']

        while elapsed < timeout:
            try:
                if self.local_mode:
                    # Local mode: Get logs from Docker container
                    import subprocess
                    if self.vllm_container_name:
                        result = subprocess.run(
                            ["docker", "logs", "--tail", "20", self.vllm_container_name],
                            capture_output=True, text=True, timeout=5
                        )
                        if result.stdout:
                            for line in result.stdout.splitlines()[-5:]:  # Show last 5 lines
                                line = line.strip()
                                if line:
                                    self._print_log_line(line)

                    # Check vLLM directly via HTTP
                    response = requests.get(f"http://localhost:{vllm_port}/v1/models", timeout=5)
                    if response.status_code == 200:
                        console.print(f"\n[green]✓[/green] vLLM server is ready! ({elapsed}s)")
                        self.logger.info("vLLM /v1/models endpoint is responding")
                        return True
                else:
                    # Vast.ai mode: Stream logs via SSH
                    new_content, log_position = self.vast_manager.tail_file(
                        self.instance_id, "/tmp/vllm.log", log_position
                    )

                    if new_content:
                        for line in new_content.splitlines():
                            line = line.strip()
                            if line:
                                self._print_log_line(line)

                    # Check via SSH (quiet=True to suppress expected errors during startup)
                    check_cmd = f"curl -f -s http://localhost:{vllm_port}/v1/models -o /tmp/vllm_models_check.json && echo 'SUCCESS'"

                    if self.vast_manager.execute_command(self.instance_id, check_cmd, quiet=True):
                        console.print(f"\n[green]✓[/green] vLLM server is ready! ({elapsed}s)")
                        self.logger.info("vLLM /v1/models endpoint is responding")
                        return True

            except requests.RequestException:
                pass  # Expected while vLLM is starting
            except Exception as e:
                self.logger.debug(f"vLLM check failed: {e}")

            time.sleep(check_interval)
            elapsed += check_interval

            # Show progress every 30 seconds if no activity
            if elapsed % 30 == 0:
                console.print(f"  [dim]... waiting ({elapsed}/{timeout}s)[/dim]")

        console.print(f"\n[red]✗[/red] vLLM server did not become ready within {timeout}s")

        # Fetch remaining logs for debugging
        if not self.local_mode:
            console.print("\n[yellow]Full vLLM logs:[/yellow]")
            logs = self.vast_manager.get_file_content(self.instance_id, "/tmp/vllm.log")
            if logs:
                console.print(logs[-2000:])  # Last 2000 chars

        return False

    def _print_log_line(self, line: str):
        """Print a log line with color coding based on content"""
        if 'error' in line.lower() or 'exception' in line.lower() or 'traceback' in line.lower():
            console.print(f"  [red]{line}[/red]")
        elif 'warning' in line.lower() or 'warn' in line.lower():
            console.print(f"  [yellow]{line}[/yellow]")
        elif 'info' in line.lower() or 'starting' in line.lower() or 'loaded' in line.lower():
            console.print(f"  [cyan]{line}[/cyan]")
        elif 'ready' in line.lower() or 'uvicorn' in line.lower() or 'application startup' in line.lower():
            console.print(f"  [green]{line}[/green]")
        elif '%' in line:  # Progress indicators
            console.print(f"  [magenta]{line}[/magenta]")
        else:
            console.print(f"  [dim]{line}[/dim]")

    def _run_benchmark(self) -> bool:
        """Run AIPerf benchmark (local or remote)"""
        console.print("\n[bold]Step 5: Running AIPerf benchmark[/bold]")

        try:
            aiperf_config = self.config['aiperf']
            vllm_url = f"http://localhost:{self.config['vllm']['port']}"

            console.print(f"Concurrency: [yellow]{self.args.concurrency or aiperf_config['concurrency']}[/yellow]")
            console.print(f"Request Count: [yellow]{self.args.request_count or aiperf_config['request_count']}[/yellow]")

            if self.local_mode:
                # Local mode: Run AIPerf in Docker with host networking and GPU monitoring
                import subprocess

                # Start GPU telemetry collection using DCGM
                console.print("Starting GPU telemetry collection...")
                gpu_monitor_cmd = "nv-hostengine -n 2>/dev/null; sleep 1; dcgmi dmon -e 150,155,156,203,204,240,241 -d 1000 > /tmp/gpu_metrics.log 2>&1 &"
                subprocess.run(
                    ["docker", "exec", "-d", self.vllm_container_name, "bash", "-c", gpu_monitor_cmd],
                    capture_output=True
                )
                time.sleep(2)

                warmup_count = aiperf_config.get('warmup_request_count', 0)
                warmup_arg = f"--warmup-request-count {warmup_count} " if warmup_count > 0 else ""

                aiperf_docker_cmd = [
                    "docker", "run", "--rm",
                    "--gpus", "all",
                    "--network", "host",
                    "-v", f"{self.results_dir.absolute()}:/results",
                    self.docker_image,
                    "bash", "-c",
                    f"aiperf profile "
                    f"--model {self.model_name} "
                    f"--url {vllm_url} "
                    f"--endpoint-type {aiperf_config['endpoint_type']} "
                    f"--concurrency {self.args.concurrency or aiperf_config['concurrency']} "
                    f"--request-count {self.args.request_count or aiperf_config['request_count']} "
                    f"{warmup_arg}"
                    f"--request-timeout-seconds {aiperf_config['request_timeout_seconds']} "
                    f"--random-seed {aiperf_config['random_seed']} "
                    f"--output-artifact-dir /results "
                    f"{'--streaming' if aiperf_config['streaming'] else ''}"
                ]

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Running benchmark with GPU telemetry...", total=None)

                    result = subprocess.run(aiperf_docker_cmd, capture_output=True, text=True)

                    # Stop GPU monitoring
                    subprocess.run(
                        ["docker", "exec", self.vllm_container_name, "pkill", "-f", "dcgmi dmon"],
                        capture_output=True
                    )

                    # Copy GPU metrics from container to results directory
                    subprocess.run(
                        ["docker", "cp", f"{self.vllm_container_name}:/tmp/gpu_metrics.log",
                         f"{self.results_dir}/gpu_metrics.log"],
                        capture_output=True
                    )

                    if result.returncode != 0:
                        self.logger.error(f"AIPerf failed: {result.stderr}")
                        console.print("[red]✗[/red] Benchmark execution failed")
                        return False

                console.print("[green]✓[/green] Benchmark completed with GPU telemetry")
            else:
                # Vast.ai mode: Run on remote instance with GPU telemetry
                warmup_count = aiperf_config.get('warmup_request_count', 0)
                warmup_arg = f"--warmup-request-count {warmup_count} \\" if warmup_count > 0 else ""

                aiperf_cmd = f"""
cd /tmp && aiperf profile \\
  --model {self.model_name} \\
  --url {vllm_url} \\
  --endpoint-type {aiperf_config['endpoint_type']} \\
  --concurrency {self.args.concurrency or aiperf_config['concurrency']} \\
  --request-count {self.args.request_count or aiperf_config['request_count']} \\
  {warmup_arg}
  --request-timeout-seconds {aiperf_config['request_timeout_seconds']} \\
  --random-seed {aiperf_config['random_seed']} \\
  --output-artifact-dir /tmp/aiperf_results \\
  {('--streaming' if aiperf_config['streaming'] else '')} \\
  {self.args.aiperf_args or aiperf_config.get('extra_args', '')} \\
  > /tmp/aiperf.log 2>&1 && echo "AIPERF_SUCCESS"
"""

                # Start GPU telemetry collection using DCGM
                console.print("Starting GPU telemetry collection...")
                gpu_monitor_cmd = "nv-hostengine -n 2>/dev/null; sleep 1; dcgmi dmon -e 150,155,156,203,204,240,241,1001,1002,1003,1004,1005,1006,1007,1008 -d 1000 > /tmp/gpu_metrics.log 2>&1 &"
                self.vast_manager.execute_command(self.instance_id, gpu_monitor_cmd, background=True)
                time.sleep(2)

                # Execute benchmark
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Running benchmark with GPU telemetry...", total=None)

                    success = self.vast_manager.execute_command(
                        self.instance_id,
                        aiperf_cmd,
                        background=False
                    )

                    # Stop GPU monitoring (ignore exit code - pkill returns 1 if no process found)
                    self.vast_manager.execute_command(self.instance_id, "pkill -f 'dcgmi dmon' || true", quiet=True)

                    if not success:
                        # Try to get AIPerf log for debugging
                        aiperf_log = self.vast_manager.get_file_content(self.instance_id, "/tmp/aiperf.log")
                        if aiperf_log:
                            self.logger.error(f"AIPerf log (last 1000 chars): {aiperf_log[-1000:]}")
                        console.print("[red]✗[/red] Benchmark execution failed - check aiperf.log")
                        return False

                console.print("[green]✓[/green] Benchmark completed with GPU telemetry")

            return True

        except Exception as e:
            self.logger.error(f"Benchmark execution failed: {e}", exc_info=True)
            console.print(f"[red]✗[/red] Benchmark execution failed: {e}")
            return False

    def _run_lm_eval(self) -> bool:
        """Run lm_eval quality benchmark (lm_eval starts vLLM itself, local or remote)"""
        console.print("\n[bold]Step 3-5: Running lm_eval quality benchmark[/bold]")

        try:
            # Parse lm_eval params
            lm_eval_params = {}
            if self.args.lm_eval_params:
                try:
                    lm_eval_params = json.loads(self.args.lm_eval_params)
                except json.JSONDecodeError as e:
                    console.print(f"[red]✗[/red] Invalid JSON in --lm-eval-params: {e}")
                    return False

            # Get default parameters
            tasks = lm_eval_params.get('tasks', 'hellaswag')
            num_fewshot = lm_eval_params.get('num_fewshot', 0)
            batch_size = lm_eval_params.get('batch_size', 'auto')
            limit = lm_eval_params.get('limit', None)

            # Build vLLM model arguments for lm_eval
            vllm_config = self.config['vllm']
            num_gpus = self.args.num_gpus or (1 if self.local_mode else self.config['vast']['num_gpus'])

            # lm_eval uses model_args to configure vLLM
            model_args = f"pretrained={self.model_name}"
            model_args += f",tensor_parallel_size={num_gpus}"
            model_args += f",gpu_memory_utilization={vllm_config['gpu_memory_utilization']}"
            model_args += f",max_model_len={vllm_config.get('max_model_len', 4096)}"

            # Add any additional vLLM args
            if self.args.vllm_args:
                # Parse vllm_args and convert to model_args format
                import re
                args_list = [arg.strip() for arg in self.args.vllm_args.split('--') if arg.strip()]
                for arg in args_list:
                    parts = arg.split(None, 1)
                    if len(parts) == 2:
                        key = parts[0].replace('-', '_')
                        value = parts[1]
                        model_args += f",{key}={value}"
                    elif len(parts) == 1:
                        key = parts[0].replace('-', '_')
                        model_args += f",{key}=True"

            console.print(f"Tasks: [yellow]{tasks}[/yellow]")
            console.print(f"Few-shot: [yellow]{num_fewshot}[/yellow]")
            console.print(f"Model: [yellow]{self.model_name}[/yellow]")
            console.print(f"Tensor Parallel: [yellow]{num_gpus}[/yellow]")

            if self.local_mode:
                # Local mode: Run lm_eval in Docker container
                import subprocess

                output_path = "/results"
                lm_eval_docker_cmd = [
                    "docker", "run", "--rm",
                    "--gpus", "all",
                    "-v", f"{self.results_dir.absolute()}:/results",
                    self.docker_image,
                    "lm_eval",
                    "--model", "vllm",
                    "--model_args", model_args,
                    "--tasks", tasks,
                    "--num_fewshot", str(num_fewshot),
                    "--batch_size", str(batch_size),
                    "--output_path", output_path,
                    "--log_samples"
                ]

                if limit:
                    lm_eval_docker_cmd.extend(["--limit", str(limit)])

                # Add any additional lm_eval params
                for key, value in lm_eval_params.items():
                    if key not in ['tasks', 'num_fewshot', 'batch_size', 'limit']:
                        if isinstance(value, bool):
                            # Boolean flags: only add if True
                            if value:
                                lm_eval_docker_cmd.append(f"--{key}")
                        else:
                            lm_eval_docker_cmd.extend([f"--{key}", str(value)])

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Running lm_eval benchmark (this may take a while)...", total=None)

                    result = subprocess.run(lm_eval_docker_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        self.logger.error(f"lm_eval failed: {result.stderr}")
                        console.print("[red]✗[/red] lm_eval execution failed")
                        console.print(f"\n[yellow]Error output:[/yellow]\n{result.stderr[-2000:]}")
                        return False

                console.print("[green]✓[/green] lm_eval benchmark completed")
            else:
                # Vast.ai mode: Run on remote instance
                lm_eval_cmd = f"""
lm_eval --model vllm \\
  --model_args "{model_args}" \\
  --tasks {tasks} \\
  --num_fewshot {num_fewshot} \\
  --batch_size {batch_size} \\
  --output_path /tmp/lm_eval_results \\
  --log_samples"""

                if limit:
                    lm_eval_cmd += f" \\\n  --limit {limit}"

                for key, value in lm_eval_params.items():
                    if key not in ['tasks', 'num_fewshot', 'batch_size', 'limit']:
                        if isinstance(value, bool):
                            if value:
                                lm_eval_cmd += f" \\\n  --{key}"
                        elif isinstance(value, str) and any(c in value for c in ' :<>/'):
                            lm_eval_cmd += f' \\\n  --{key} "{value}"'
                        else:
                            lm_eval_cmd += f" \\\n  --{key} {value}"

                lm_eval_cmd += " > /tmp/lm_eval.log 2>&1 && echo 'LMEVAL_SUCCESS'"

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Running lm_eval benchmark (this may take a while)...", total=None)

                    success = self.vast_manager.execute_command(
                        self.instance_id,
                        lm_eval_cmd,
                        background=False
                    )

                    if not success:
                        console.print("[red]✗[/red] lm_eval execution failed")

                        # Fetch logs for debugging
                        console.print("\n[yellow]Fetching lm_eval logs for debugging:[/yellow]")
                        logs = self.vast_manager.get_file_content(self.instance_id, "/tmp/lm_eval.log")
                        if logs:
                            console.print(logs[-2000:])

                        return False

                console.print("[green]✓[/green] lm_eval benchmark completed")

            return True

        except Exception as e:
            self.logger.error(f"lm_eval execution failed: {e}", exc_info=True)
            console.print(f"[red]✗[/red] lm_eval execution failed: {e}")
            return False

    def _collect_results(self) -> bool:
        """Collect benchmark results (local: already in results_dir, remote: download from instance)"""
        console.print("\n[bold]Step 6: Collecting results[/bold]")

        try:
            if self.local_mode:
                # Local mode: Results are already in results_dir (mounted volume)
                result_files = list(self.results_dir.glob("*"))
                if result_files:
                    console.print(f"[green]✓[/green] Results already in {self.results_dir}")
                    for f in result_files:
                        console.print(f"  - {f.name}")
                else:
                    console.print(f"[yellow]⚠[/yellow] No result files found in {self.results_dir}")
                return True

            # Vast.ai mode: Download results from remote instance
            benchmark_type = self.args.benchmark_type
            files_to_download = {}

            if benchmark_type == "performance":
                # First, discover what files aiperf actually created
                console.print("Discovering AIPerf output files...")
                find_cmd = "find /tmp/aiperf_results -type f 2>/dev/null"
                found_files = self.vast_manager.run_command(self.instance_id, find_cmd)
                if found_files:
                    console.print(f"Found files in /tmp/aiperf_results:")
                    for f in found_files.strip().split('\n'):
                        if f:
                            console.print(f"  - {f}")
                else:
                    console.print("[yellow]⚠[/yellow] No files found in /tmp/aiperf_results/")
                    # Check if directory exists at all
                    ls_result = self.vast_manager.run_command(self.instance_id, "ls -la /tmp/ | grep -i aiperf")
                    if ls_result:
                        console.print(f"  /tmp contents: {ls_result}")

                # AIPerf results - try multiple possible filenames
                files_to_download = {
                    # Logs first (most likely to exist)
                    '/tmp/aiperf.log': 'aiperf.log',
                    '/tmp/vllm.log': 'vllm.log',
                    '/tmp/gpu_metrics.log': 'gpu_metrics.log'
                }

                # Add any files found in aiperf_results directory
                if found_files:
                    for remote_file in found_files.strip().split('\n'):
                        if remote_file and remote_file.strip():
                            local_name = remote_file.split('/')[-1]
                            files_to_download[remote_file.strip()] = local_name
            else:
                # lm_eval results
                # lm_eval creates results.json in the output_path directory
                files_to_download = {
                    '/tmp/lm_eval_results/results.json': 'lm_eval_results.json',
                    '/tmp/lm_eval.log': 'lm_eval.log'
                }

                # Try to download samples if they exist
                samples_cmd = "find /tmp/lm_eval_results \\( -name '*.jsonl' -o -name 'samples_*.json' \\) 2>/dev/null | head -5"
                sample_files = self.vast_manager.run_command(self.instance_id, samples_cmd)
                if sample_files:
                    for sample_file in sample_files.strip().split('\n'):
                        if sample_file:
                            filename = sample_file.split('/')[-1]
                            files_to_download[sample_file] = filename

            for remote_path, local_name in files_to_download.items():
                local_path = self.results_dir / local_name

                content = self.vast_manager.get_file_content(self.instance_id, remote_path)

                if content:
                    local_path.write_text(content)
                    console.print(f"[green]✓[/green] Downloaded {local_name}")
                else:
                    console.print(f"[yellow]⚠[/yellow] Could not download {local_name}")

            console.print(f"[green]✓[/green] Results collected in {self.results_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to collect results: {e}", exc_info=True)
            console.print(f"[red]✗[/red] Failed to collect results: {e}")
            return False

    def _upload_results(self) -> bool:
        """Upload results to S3"""
        console.print("\n[bold]Step 7: Uploading results to S3[/bold]")

        try:
            benchmark_type = self.args.benchmark_type

            # Prepare metadata
            metadata = {
                'model': self.model_name,
                'gpu': f"{self.config['vast']['gpu_name']} x{self.config['vast']['num_gpus']}",
                'timestamp': datetime.now().isoformat(),
                'instance_id': self.instance_id,
                'benchmark_type': benchmark_type,
                'vllm_config': self.config['vllm']
            }

            # Add benchmark-specific config
            if benchmark_type == "performance":
                metadata['aiperf_config'] = self.config['aiperf']
            else:
                # Add lm_eval params if provided
                if self.args.lm_eval_params:
                    metadata['lm_eval_params'] = json.loads(self.args.lm_eval_params)

            # Upload results
            success = self.result_uploader.upload_results(
                self.results_dir,
                self.model_name,
                metadata,
                benchmark_type
            )

            if success:
                # Store S3 path for potential download
                timestamp = datetime.now().strftime(self.config['s3']['timestamp_format'])
                safe_model_name = self.model_name.replace('/', '_').replace(' ', '_')
                self.s3_results_path = f"{benchmark_type}/{safe_model_name}/{timestamp}/"

                console.print("[green]✓[/green] Results uploaded to S3")
                return True
            else:
                console.print("[red]✗[/red] Failed to upload results")
                return False

        except Exception as e:
            self.logger.error(f"Failed to upload results: {e}", exc_info=True)
            console.print(f"[red]✗[/red] Failed to upload results: {e}")
            return False

    def _analyze_gpu_metrics(self):
        """Analyze and display GPU metrics if available"""
        gpu_metrics_file = self.results_dir / 'gpu_metrics.log'

        if not gpu_metrics_file.exists():
            return

        try:
            # Parse GPU metrics
            metrics = {'power': [], 'energy': [], 'gpu_util': [], 'mem_util': []}

            with open(gpu_metrics_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or 'Error' in line or not line.strip():
                        continue

                    parts = line.split()
                    if len(parts) >= 5 and parts[0] == 'GPU':
                        try:
                            power = float(parts[2])
                            energy = float(parts[3])
                            gpu_util = float(parts[4]) if parts[4] != '-' else 0
                            mem_util = float(parts[5]) if len(parts) > 5 and parts[5] != '-' else 0

                            metrics['power'].append(power)
                            metrics['energy'].append(energy)
                            metrics['gpu_util'].append(gpu_util)
                            metrics['mem_util'].append(mem_util)
                        except (ValueError, IndexError):
                            continue

            if not metrics['power']:
                return

            # Calculate statistics
            energy_mj = metrics['energy'][-1] - metrics['energy'][0]
            energy_wh = energy_mj / 1000 / 3600
            avg_power = sum(metrics['power']) / len(metrics['power'])
            max_power = max(metrics['power'])
            active_gpu_utils = [u for u in metrics['gpu_util'] if u > 0]
            avg_gpu_util = sum(active_gpu_utils) / len(active_gpu_utils) if active_gpu_utils else 0

            # Display summary
            console.print(f"\n[bold cyan]GPU Power Consumption:[/bold cyan]")
            console.print(f"  Total Energy: {energy_wh:.4f} Wh ({energy_mj/1000:.2f} J)")
            console.print(f"  Avg Power:    {avg_power:.2f} W")
            console.print(f"  Peak Power:   {max_power:.2f} W")
            if active_gpu_utils:
                console.print(f"  Avg GPU Util: {avg_gpu_util:.1f}%")

        except Exception as e:
            self.logger.warning(f"Failed to analyze GPU metrics: {e}")


    def _download_results(self) -> bool:
        """Download results from S3 to local directory"""
        console.print("\n[bold]Step 8: Downloading results to local directory[/bold]")

        try:
            download_dir = Path(self.args.download_results)
            decompress = not self.args.no_decompress

            if not self.s3_results_path:
                console.print("[yellow]⚠[/yellow] No S3 results path available")
                return False

            console.print(f"Downloading to: [yellow]{download_dir}[/yellow]")
            console.print(f"Decompress .gz files: [yellow]{'Yes' if decompress else 'No'}[/yellow]")

            success = self.result_uploader.download_results(
                self.s3_results_path,
                download_dir,
                decompress=decompress
            )

            if success:
                console.print(f"[green]✓[/green] Results downloaded to {download_dir}")
                return True
            else:
                console.print("[red]✗[/red] Failed to download results")
                return False

        except Exception as e:
            self.logger.error(f"Failed to download results: {e}", exc_info=True)
            console.print(f"[red]✗[/red] Failed to download results: {e}")
            return False

    def _cleanup(self):
        """Cleanup resources"""
        if self.local_mode:
            # Cleanup Docker containers
            if self.vllm_container_name:
                console.print("\n[bold]Cleanup: Stopping Docker containers[/bold]")
                try:
                    self._docker_cleanup()
                    console.print(f"[green]✓[/green] Docker containers cleaned up")
                except Exception as e:
                    self.logger.error(f"Failed to cleanup Docker: {e}", exc_info=True)
                    console.print(f"[red]✗[/red] Failed to cleanup Docker: {e}")
        elif self.instance_id:
            # Cleanup vast.ai instance
            console.print("\n[bold]Cleanup: Terminating Vast.ai instance[/bold]")
            try:
                self.vast_manager.destroy_instance(self.instance_id)
                console.print(f"[green]✓[/green] Instance {self.instance_id} terminated")
            except Exception as e:
                self.logger.error(f"Failed to destroy instance: {e}", exc_info=True)
                console.print(f"[red]✗[/red] Failed to destroy instance: {e}")

    # Docker helper methods for local mode
    def _docker_run(self, command: str, detach: bool = False, name: Optional[str] = None) -> bool:
        """Run a command in a Docker container"""
        import subprocess

        docker_cmd = ["docker", "run", "--rm", "--gpus", "all"]

        if detach:
            docker_cmd.append("-d")

        if name:
            docker_cmd.extend(["--name", name])

        docker_cmd.append(self.docker_image)
        docker_cmd.extend(["bash", "-c", command])

        try:
            result = subprocess.run(docker_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"Docker command failed: {result.stderr}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Docker run failed: {e}")
            return False

    def _docker_exec(self, container_name: str, command: str) -> tuple[bool, str]:
        """Execute a command in a running Docker container"""
        import subprocess

        docker_cmd = ["docker", "exec", container_name, "bash", "-c", command]

        try:
            result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0, result.stdout
        except Exception as e:
            self.logger.error(f"Docker exec failed: {e}")
            return False, str(e)

    def _docker_cleanup(self):
        """Stop and remove Docker containers"""
        import subprocess

        if self.vllm_container_name:
            try:
                subprocess.run(["docker", "stop", self.vllm_container_name],
                             capture_output=True, timeout=30)
                subprocess.run(["docker", "rm", self.vllm_container_name],
                             capture_output=True, timeout=30)
            except Exception as e:
                self.logger.error(f"Failed to cleanup container {self.vllm_container_name}: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="LLM Benchmark Runner for Vast.ai",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a benchmark suite (recommended for multiple benchmarks)
  python benchmark_runner.py --suite configs/rtx4090_llama_suite.yaml

  # Single performance benchmark on vast.ai (legacy mode)
  python benchmark_runner.py meta-llama/Llama-2-7b-hf

  # Local benchmark using Docker
  python benchmark_runner.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local

  # Quality benchmark with lm_eval
  python benchmark_runner.py meta-llama/Llama-2-7b-hf --benchmark-type quality \\
    --lm-eval-params '{"tasks": "mmlu,hellaswag", "num_fewshot": 5}'

  # Override GPU configuration
  python benchmark_runner.py meta-llama/Llama-2-7b-hf --gpu-name RTX_4090 --num-gpus 2

  # Custom performance benchmark parameters
  python benchmark_runner.py meta-llama/Llama-2-7b-hf --concurrency 20 --request-count 500

  # With custom vLLM and AIPerf arguments
  python benchmark_runner.py meta-llama/Llama-2-7b-hf \\
    --vllm-args "--dtype float16" \\
    --aiperf-args "--benchmark-duration 300"

  # Download results to local directory after benchmark
  python benchmark_runner.py meta-llama/Llama-2-7b-hf --download-results ./my_results

  # Download without decompressing .gz files
  python benchmark_runner.py meta-llama/Llama-2-7b-hf --download-results ./my_results --no-decompress
        """
    )

    # Benchmark suite mode (new)
    parser.add_argument(
        "--suite",
        metavar="CONFIG",
        help="Path to benchmark suite configuration YAML file (e.g., configs/rtx4090_llama_suite.yaml)"
    )

    # Legacy single model mode
    parser.add_argument(
        "model",
        nargs="?",  # Make optional since --suite can be used instead
        help="HuggingFace model name to benchmark (e.g., meta-llama/Llama-2-7b-hf)"
    )

    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml) - used for legacy single-model mode"
    )

    # Benchmark type
    parser.add_argument(
        "--benchmark-type",
        choices=["performance", "quality"],
        default="performance",
        help="Type of benchmark to run: 'performance' (AIPerf) or 'quality' (lm_eval). Default: performance"
    )

    # Vast.ai configuration
    vast_group = parser.add_argument_group("Vast.ai Configuration")
    vast_group.add_argument(
        "--instance-id",
        type=int,
        help="Use existing vast.ai instance ID instead of creating a new one (for resuming failed runs)"
    )
    vast_group.add_argument(
        "--local",
        action="store_true",
        help="Run benchmark locally using Docker instead of vast.ai (default: False)"
    )
    vast_group.add_argument(
        "--gpu-name",
        help="GPU type to use (e.g., RTX_4090, A100)"
    )
    vast_group.add_argument(
        "--num-gpus",
        type=int,
        help="Number of GPUs to use"
    )

    # vLLM configuration
    vllm_group = parser.add_argument_group("vLLM Configuration")
    vllm_group.add_argument(
        "--vllm-args",
        help="Additional vLLM arguments (e.g., '--dtype float16 --quantization awq')"
    )

    # AIPerf configuration
    aiperf_group = parser.add_argument_group("AIPerf Configuration (for --benchmark-type performance)")
    aiperf_group.add_argument(
        "--concurrency",
        type=int,
        help="Number of concurrent requests"
    )
    aiperf_group.add_argument(
        "--request-count",
        type=int,
        help="Total number of requests to send"
    )
    aiperf_group.add_argument(
        "--aiperf-args",
        help="Additional AIPerf arguments"
    )

    # lm_eval configuration
    lmeval_group = parser.add_argument_group("lm_eval Configuration (for --benchmark-type quality)")
    lmeval_group.add_argument(
        "--lm-eval-params",
        help='lm_eval parameters as JSON string (e.g., \'{"tasks": "mmlu,hellaswag", "num_fewshot": 5}\')'
    )

    # Results download
    download_group = parser.add_argument_group("Results Download")
    download_group.add_argument(
        "--download-results",
        metavar="DIR",
        help="Download results from S3 to this local directory after benchmark"
    )
    download_group.add_argument(
        "--no-decompress",
        action="store_true",
        help="Don't decompress .gz files when downloading"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    # Load environment variables
    load_dotenv()

    # Parse arguments
    args = parse_arguments()

    # Determine which mode to use
    if args.suite:
        # Suite mode - run multiple benchmarks on one instance
        suite_path = Path(args.suite)
        if not suite_path.exists():
            console.print(f"[red]Error: Suite configuration file not found: {args.suite}[/red]")
            sys.exit(1)

        with open(suite_path) as f:
            suite_config = yaml.safe_load(f)

        # Substitute environment variables in config
        def substitute_env_vars(obj):
            if isinstance(obj, str):
                if obj.startswith('${') and obj.endswith('}'):
                    env_var = obj[2:-1]
                    return os.getenv(env_var, obj)
                return obj
            elif isinstance(obj, dict):
                return {k: substitute_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_env_vars(item) for item in obj]
            return obj

        suite_config = substitute_env_vars(suite_config)

        # Setup logging
        log_level = args.log_level or 'INFO'
        setup_logging(log_level, 'benchmark.log')

        # Create and run suite runner
        runner = BenchmarkSuiteRunner(suite_config, args)
        success = runner.run()
        sys.exit(0 if success else 1)

    elif args.model:
        # Legacy single model mode
        config_path = Path(args.config)
        if not config_path.exists():
            console.print(f"[red]Error: Configuration file not found: {args.config}[/red]")
            sys.exit(1)

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Validate configuration
        if not validate_config(config):
            console.print("[red]Error: Invalid configuration[/red]")
            sys.exit(1)

        # Setup logging
        log_level = args.log_level or config.get('logging', {}).get('level', 'INFO')
        setup_logging(log_level, config.get('logging', {}).get('file', 'benchmark.log'))

        # Create and run benchmark
        runner = BenchmarkRunner(config, args.model, args)
        success = runner.run()
        sys.exit(0 if success else 1)

    else:
        console.print("[red]Error: Either --suite or a model name is required[/red]")
        console.print("\nUsage:")
        console.print("  python benchmark_runner.py --suite configs/rtx4090_llama_suite.yaml")
        console.print("  python benchmark_runner.py meta-llama/Llama-3.1-8B-Instruct")
        sys.exit(1)


if __name__ == "__main__":
    main()
