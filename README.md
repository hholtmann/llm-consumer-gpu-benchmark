# LLM GPU Benchmark Suite

**Automated LLM inference benchmarking on consumer GPUs via vast.ai**

Spin up GPU instances, run comprehensive benchmarks, collect results, and tear down - all with a single command.

[![Docker Hub](https://img.shields.io/docker/pulls/holtmann/llm-benchmark)](https://hub.docker.com/r/holtmann/llm-benchmark)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **One-command benchmarking**: `python benchmark_runner.py --suite config.yaml`
- **Consumer GPU focus**: RTX 5060 Ti, 5070 Ti, 5090 (and extensible to others)
- **Multiple workloads**: RAG (long context), API (high concurrency), Agentic (LoRA multi-tenant)
- **Comprehensive metrics**: Throughput, latency (TTFT, ITL, P95/P99), power, energy efficiency
- **Cost-efficient**: Uses vast.ai spot instances, auto-terminates after completion
- **Paper-ready output**: Auto-generates LaTeX tables and CSV exports

## Quick Start

### Prerequisites

- Python 3.10+
- [vast.ai](https://vast.ai) account with API key
- AWS S3 bucket for results (optional but recommended)
- HuggingFace account with token (for gated models)

### Installation

```bash
git clone https://github.com/yourusername/llm-gpu-benchmark.git
cd llm-gpu-benchmark

# Install dependencies
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Edit .env with your API keys
```

### Your First Benchmark

```bash
# Run using one of our benchmark configs (reproduces paper results)
python benchmark_runner.py --suite research_results/results_config/rtx5090_1x.yaml

# Or create your own config from the template
cp configs/template.yaml configs/my_benchmark.yaml
# Edit configs/my_benchmark.yaml with your settings
python benchmark_runner.py --suite configs/my_benchmark.yaml

# The script will:
# 1. Find and rent a GPU on vast.ai
# 2. Set up the environment (vLLM, models)
# 3. Run all benchmarks in the config
# 4. Upload results to S3
# 5. Terminate the instance
```

## Configuration

Benchmarks are defined in YAML files. Here's the structure:

```yaml
name: RTX 5090 1x GPU
description: Benchmark suite for RTX 5090 32GB single GPU

instance:
  gpu_type: RTX 5090          # GPU to rent on vast.ai
  gpu_count: 1                # Number of GPUs
  disk_space_gb: 100          # Disk space for models
  image: holtmann/llm-benchmark:latest
  max_bid_price: 2.0          # Max $/hr for spot instance

s3:
  bucket: ${S3_BUCKET_NAME}   # From .env
  prefix: benchmarks/rtx5090_1x
  upload_json: true
  upload_csv: true

benchmarks:
  - name: qwen3-8b-nvfp4-rag-8k-c8
    model: nvidia/Qwen3-8B-NVFP4
    vllm:
      max_model_len: 9216
      gpu_memory_utilization: 0.9
      dtype: auto
    aiperf:
      endpoint_type: chat
      streaming: true
      concurrency: 8
      synthetic_input_tokens_mean: 8192
      output_tokens_mean: 512
      request_count: 500
```

## Workload Types

### RAG (Retrieval-Augmented Generation)
Long context, moderate concurrency - typical enterprise RAG pipelines.

```yaml
- name: qwen3-8b-nvfp4-rag-16k-c4
  aiperf:
    concurrency: 4
    synthetic_input_tokens_mean: 16384  # 16k context
    output_tokens_mean: 512
```

### API (High-Concurrency)
Short context, high concurrency - chatbot/API serving.

```yaml
- name: qwen3-8b-nvfp4-api-c128
  aiperf:
    concurrency: 128
    synthetic_input_tokens_mean: 256   # Short prompts
    output_tokens_mean: 256
```

### Agentic (Multi-LoRA)
LoRA adapter switching for multi-tenant deployments.

```yaml
- name: qwen3-8b-nvfp4-agentic-lora-c32
  vllm:
    enable_lora: true
    max_loras: 3
    lora_modules:
      - name: customer-support
        path: /models/loras/customer-support
      - name: code-assistant
        path: /models/loras/code-assistant
  aiperf:
    model_selection_strategy: random  # Randomly select adapter per request
```

## Metrics Collected

| Metric | Description | Unit |
|--------|-------------|------|
| `output_token_throughput` | Total tokens generated per second | tok/s |
| `output_token_throughput_per_user` | Per-request throughput | tok/s |
| `time_to_first_token` | Latency to first token (avg, P50, P95, P99) | ms |
| `inter_token_latency` | Streaming speed (avg, P50, P95, P99) | ms |
| `avg_power_w` | Average GPU power during benchmark | W |
| `wh_per_mtok` | Energy efficiency | Wh/million tokens |
| `max_temp_c` | Peak GPU temperature | °C |
| `throttle_pct` | Thermal throttling percentage | % |

## Running Multiple Configs in Parallel

Each config runs independently - perfect for parallel execution:

```bash
# Terminal 1
python benchmark_runner.py --suite research_results/results_config/rtx5060ti_1x.yaml

# Terminal 2
python benchmark_runner.py --suite research_results/results_config/rtx5070ti_1x.yaml

# Terminal 3
python benchmark_runner.py --suite research_results/results_config/rtx5090_1x.yaml

# Results go to separate S3 prefixes, no conflicts
```

## Cost Estimates

| GPU Config | Benchmarks | Est. Time | Est. Cost |
|------------|------------|-----------|-----------|
| RTX 5060 Ti 1x | 33 | ~4-5 hrs | ~$2-4 |
| RTX 5060 Ti 2x | 22 | ~3 hrs | ~$3-5 |
| RTX 5070 Ti 1x | 34 | ~4-5 hrs | ~$3-5 |
| RTX 5070 Ti 2x | 26 | ~3.5 hrs | ~$5-8 |
| RTX 5090 1x | 38 | ~5-6 hrs | ~$5-10 |
| RTX 5090 2x | 30 | ~4 hrs | ~$8-15 |

*Costs vary based on vast.ai spot pricing*

## Output Structure

Results are organized per benchmark run:

```
results/
└── 20241217_143052_RTX_5090_1x/
    ├── qwen3-8b-nvfp4-rag-8k-c8/
    │   ├── metadata.json           # Config and status
    │   ├── profile_export_aiperf.json  # Detailed metrics
    │   ├── gpu_metrics.log         # Power/temp timeline
    │   └── vllm.log               # Server logs
    ├── qwen3-8b-nvfp4-rag-16k-c4/
    │   └── ...
    └── summary.json               # Suite-level summary
```

## Generating Paper Tables

Convert results to LaTeX tables:

```bash
python scripts/generate_paper_tables.py --results-dir results/ --output paper_tables.tex
```

Output:
```latex
\begin{table}[H]
  \caption{RAG workload throughput on RTX 5090}
  \begin{tabular}{l l c c c c}
    Model & Precision & Context & TPS & TTFT & ITL P95 \\
    \midrule
    Qwen3-8B & NVFP4 & 8k & 422.4 & 565 & 18.4 \\
    Qwen3-8B & NVFP4 & 16k & 225.3 & 1474 & 38.4 \\
    ...
  \end{tabular}
\end{table}
```

## Environment Variables

Create a `.env` file:

```bash
# Required
VAST_API_KEY=your_vast_ai_api_key

# For gated models (Gemma, etc.)
HF_TOKEN=your_huggingface_token

# For S3 upload
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET_NAME=your-bucket-name

# Optional: Custom LoRA adapters
LORA_CUSTOMER_SUPPORT=holtmann/qwen3-8b-customer-support-lora
LORA_TECHNICAL_DOCS=holtmann/qwen3-8b-technical-docs-lora
```

## Adding New GPUs

1. Create a new config file:
```bash
cp configs/template.yaml configs/rtx4090_1x.yaml
```

2. Update the instance section:
```yaml
instance:
  gpu_type: RTX 4090
  gpu_count: 1
```

3. Adjust benchmarks based on VRAM (24GB for 4090):
   - Remove models that won't fit
   - Adjust max context lengths
   - Set appropriate concurrency levels

4. Run:
```bash
python benchmark_runner.py --suite configs/rtx4090_1x.yaml
```

## Model Compatibility

This suite works with **any model supported by vLLM**, including:

- Standard HuggingFace models (Llama, Mistral, Qwen, Gemma, etc.)
- Quantized models (GPTQ, AWQ, NVFP4, W4A16, etc.)
- MoE models (Mixtral, GPT-OSS, etc.)
- Any model with LoRA adapters

Simply specify the HuggingFace model ID in your config:

```yaml
model: Qwen/Qwen3-8B                       # Popular 2025 model
model: mistralai/Mistral-Small-24B-Instruct-2501  # Mistral's latest
model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    # DeepSeek R1 distilled
model: google/gemma-3-12b-it               # Google's Gemma 3
```

See [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html) for the full list.

## Docker Image

The benchmark environment is pre-built:

```bash
docker pull holtmann/llm-benchmark:latest
```

Includes:
- vLLM with NVFP4/MXFP4 support
- aiperf benchmarking tool
- CUDA 12.x + cuDNN
- HuggingFace Hub CLI
- AWS CLI for S3 uploads

Build your own:
```bash
docker build -t my-benchmark:latest -f Dockerfile .
```

## Troubleshooting

### OOM Errors
- Reduce `max_model_len` in config
- Lower `concurrency`
- Use more aggressive quantization (NVFP4 vs BF16)

### vast.ai Instance Not Found
- Increase `max_bid_price`
- Try different GPU type
- Check vast.ai availability

### Model Download Fails
- Verify `HF_TOKEN` is set
- Check model exists on HuggingFace
- Ensure sufficient disk space

### LoRA Adapter Not Found
- Check adapter name matches config
- Verify HuggingFace repo exists
- Check `LORA_*` env variables

## Citation

If you use this benchmark suite in your research, please cite:

```bibtex
@misc{knoop2026privatellminferenceconsumer,
      title={Private LLM Inference on Consumer Blackwell GPUs: A Practical Guide for Cost-Effective Local Deployment in SMEs}, 
      author={Jonathan Knoop and Hendrik Holtmann},
      year={2026},
      eprint={2601.09527},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.09527}, 
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add your GPU configs or improvements
4. Submit a pull request

Ideas for contribution:
- New GPU configurations (RTX 4090, A6000, etc.)
- Additional workload types
- Improved metrics collection
- Documentation improvements

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
- [aiperf](https://github.com/ai-dynamo/aiperf) - LLM benchmarking tool
- [vast.ai](https://vast.ai) - GPU cloud marketplace
- [NVIDIA](https://nvidia.com) - NVFP4 quantization support
