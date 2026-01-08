"""
Utility Functions
Common utilities for configuration, logging, and formatting.
"""

import os
import logging
from typing import Dict, Any
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str = "benchmark.log"):
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Reduce noise from external libraries
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        bool: True if valid, False otherwise
    """
    required_sections = ['vast', 'vllm', 'aiperf', 's3']

    for section in required_sections:
        if section not in config:
            logging.error(f"Missing required configuration section: {section}")
            return False

    # Validate Vast.ai config
    vast_required = ['gpu_name', 'num_gpus', 'disk_space', 'image']
    for key in vast_required:
        if key not in config['vast']:
            logging.error(f"Missing required vast configuration: {key}")
            return False

    # Validate vLLM config
    vllm_required = ['port', 'host']
    for key in vllm_required:
        if key not in config['vllm']:
            logging.error(f"Missing required vllm configuration: {key}")
            return False

    # Validate AIPerf config
    aiperf_required = ['endpoint_type', 'concurrency', 'request_count']
    for key in aiperf_required:
        if key not in config['aiperf']:
            logging.error(f"Missing required aiperf configuration: {key}")
            return False

    # Validate environment variables
    required_env_vars = [
        'VAST_API_KEY',
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'S3_BUCKET_NAME'
    ]

    for var in required_env_vars:
        if not os.getenv(var):
            logging.error(f"Missing required environment variable: {var}")
            return False

    return True


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing/replacing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Replace common invalid characters
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    sanitized = filename

    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')

    return sanitized


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human-readable format.

    Args:
        bytes_value: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0

    return f"{bytes_value:.2f} PB"


def get_model_size_estimate(model_name: str) -> str:
    """
    Estimate model size based on name (rough approximation).

    Args:
        model_name: HuggingFace model name

    Returns:
        Estimated size string
    """
    name_lower = model_name.lower()

    # Extract parameter count from model name
    if '405b' in name_lower or '400b' in name_lower:
        return "~810 GB"
    elif '70b' in name_lower:
        return "~140 GB"
    elif '34b' in name_lower:
        return "~68 GB"
    elif '13b' in name_lower:
        return "~26 GB"
    elif '7b' in name_lower:
        return "~14 GB"
    elif '3b' in name_lower:
        return "~6 GB"
    elif '1b' in name_lower:
        return "~2 GB"
    else:
        return "Unknown"
