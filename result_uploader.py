"""
Result Uploader
Handles uploading benchmark results to S3 in multiple formats.
"""

import os
import json
import gzip
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import boto3
import pandas as pd
from botocore.exceptions import ClientError


class ResultUploader:
    """Manager for uploading benchmark results to S3"""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize result uploader.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger

        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )

        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME not found in environment variables")

        self.results_prefix = os.getenv('S3_RESULTS_PREFIX', 'benchmarks/')

    def upload_results(
        self,
        results_dir: Path,
        model_name: str,
        metadata: Dict[str, Any],
        benchmark_type: str = "performance"
    ) -> bool:
        """
        Upload benchmark results to S3 in multiple formats.

        Args:
            results_dir: Directory containing result files
            model_name: Name of the model benchmarked
            metadata: Metadata to include with results
            benchmark_type: Type of benchmark ("performance" or "quality")

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            timestamp = datetime.now().strftime(self.config['s3']['timestamp_format'])

            # Create a sanitized model name for the S3 path
            safe_model_name = model_name.replace('/', '_').replace(' ', '_')

            # Base S3 path with benchmark type folder
            s3_base_path = f"{self.results_prefix}{benchmark_type}/{safe_model_name}/{timestamp}/"

            uploaded_files = []

            if benchmark_type == "performance":
                # Handle AIPerf results
                # 1. Upload JSON results (from AIPerf)
                if self.config['s3']['upload_json']:
                    # Check for AIPerf JSONL output
                    jsonl_file = results_dir / 'aiperf_results.jsonl'
                    if jsonl_file.exists():
                        # Upload AIPerf JSONL
                        self._upload_file(jsonl_file, f"{s3_base_path}aiperf_results.jsonl")
                        uploaded_files.append('aiperf_results.jsonl')

                        # Create and upload consolidated JSON with metadata
                        consolidated = self._create_consolidated_json(jsonl_file, metadata)
                        consolidated_path = results_dir / 'results_consolidated.json'
                        consolidated_path.write_text(json.dumps(consolidated, indent=2))
                        self._upload_file(consolidated_path, f"{s3_base_path}results_consolidated.json")
                        uploaded_files.append('results_consolidated.json')

                    # Check for AIPerf JSON output (if generated)
                    json_file = results_dir / 'aiperf_results.json'
                    if json_file.exists():
                        self._upload_file(json_file, f"{s3_base_path}aiperf_results.json")
                        uploaded_files.append('aiperf_results.json')

                # 2. Convert and upload CSV results
                if self.config['s3']['upload_csv']:
                    # Convert AIPerf JSONL to CSV
                    jsonl_file = results_dir / 'aiperf_results.jsonl'
                    if jsonl_file.exists():
                        csv_path = self._convert_to_csv(jsonl_file, results_dir)
                        if csv_path:
                            self._upload_file(csv_path, f"{s3_base_path}aiperf_results.csv")
                            uploaded_files.append('aiperf_results.csv')

                # 3. Upload logs
                if self.config['s3']['upload_logs']:
                    log_files = ['aiperf.log', 'vllm.log', 'gpu_metrics.log']

                    for log_file in log_files:
                        log_path = results_dir / log_file
                        if log_path.exists():
                            # Compress logs if configured
                            if self.config['s3'].get('compress_logs', True):
                                compressed_path = self._compress_file(log_path)
                                self._upload_file(
                                    compressed_path,
                                    f"{s3_base_path}logs/{log_file}.gz"
                                )
                                uploaded_files.append(f"logs/{log_file}.gz")
                            else:
                                self._upload_file(log_path, f"{s3_base_path}logs/{log_file}")
                                uploaded_files.append(f"logs/{log_file}")

            else:  # benchmark_type == "quality"
                # Handle lm_eval results
                # 1. Upload JSON results (from lm_eval)
                if self.config['s3']['upload_json']:
                    # lm_eval creates results.json
                    lm_eval_results = results_dir / 'lm_eval_results.json'
                    if lm_eval_results.exists():
                        self._upload_file(lm_eval_results, f"{s3_base_path}lm_eval_results.json")
                        uploaded_files.append('lm_eval_results.json')

                        # Create consolidated JSON with metadata
                        consolidated = self._create_lm_eval_consolidated(lm_eval_results, metadata)
                        consolidated_path = results_dir / 'results_consolidated.json'
                        consolidated_path.write_text(json.dumps(consolidated, indent=2))
                        self._upload_file(consolidated_path, f"{s3_base_path}results_consolidated.json")
                        uploaded_files.append('results_consolidated.json')

                # 2. Upload sample files if they exist
                for file in results_dir.glob('*.jsonl'):
                    self._upload_file(file, f"{s3_base_path}samples/{file.name}")
                    uploaded_files.append(f"samples/{file.name}")

                for file in results_dir.glob('samples_*.json'):
                    self._upload_file(file, f"{s3_base_path}samples/{file.name}")
                    uploaded_files.append(f"samples/{file.name}")

                # 3. Upload logs
                if self.config['s3']['upload_logs']:
                    log_files = ['lm_eval.log']

                    for log_file in log_files:
                        log_path = results_dir / log_file
                        if log_path.exists():
                            # Compress logs if configured
                            if self.config['s3'].get('compress_logs', True):
                                compressed_path = self._compress_file(log_path)
                                self._upload_file(
                                    compressed_path,
                                    f"{s3_base_path}logs/{log_file}.gz"
                                )
                                uploaded_files.append(f"logs/{log_file}.gz")
                            else:
                                self._upload_file(log_path, f"{s3_base_path}logs/{log_file}")
                                uploaded_files.append(f"logs/{log_file}")

            # 4. Upload metadata separately
            metadata_path = results_dir / 'metadata.json'
            metadata_path.write_text(json.dumps(metadata, indent=2))
            self._upload_file(metadata_path, f"{s3_base_path}metadata.json")
            uploaded_files.append('metadata.json')

            self.logger.info(f"Uploaded {len(uploaded_files)} files to S3: {uploaded_files}")
            self.logger.info(f"S3 location: s3://{self.bucket_name}/{s3_base_path}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to upload results: {e}", exc_info=True)
            return False

    def _upload_file(self, local_path: Path, s3_key: str) -> bool:
        """
        Upload a single file to S3.

        Args:
            local_path: Path to local file
            s3_key: S3 object key

        Returns:
            bool: True if successful
        """
        try:
            self.logger.debug(f"Uploading {local_path} to s3://{self.bucket_name}/{s3_key}")

            self.s3_client.upload_file(
                str(local_path),
                self.bucket_name,
                s3_key
            )

            self.logger.debug(f"Successfully uploaded {s3_key}")
            return True

        except ClientError as e:
            self.logger.error(f"S3 upload failed for {s3_key}: {e}", exc_info=True)
            return False

    def _create_consolidated_json(
        self,
        jsonl_path: Path,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a consolidated JSON file with metadata and results.

        Args:
            jsonl_path: Path to JSONL results file
            metadata: Metadata to include

        Returns:
            Dictionary with consolidated data
        """
        results = []

        if jsonl_path.exists():
            with open(jsonl_path, 'r') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))

        # Calculate summary statistics if results exist
        summary = {}
        if results:
            summary = self._calculate_summary_stats(results)

        return {
            'metadata': metadata,
            'summary': summary,
            'results': results
        }

    def _create_lm_eval_consolidated(
        self,
        results_path: Path,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a consolidated JSON file for lm_eval results with metadata.

        Args:
            results_path: Path to lm_eval results.json file
            metadata: Metadata to include

        Returns:
            Dictionary with consolidated data
        """
        results = {}

        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)

        return {
            'metadata': metadata,
            'results': results
        }

    def _calculate_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate summary statistics from benchmark results.

        Args:
            results: List of result dictionaries

        Returns:
            Dictionary with summary statistics
        """
        # This is a simplified version - adjust based on actual AIPerf output format
        try:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(results)

            summary = {
                'total_requests': len(results),
                'timestamp': datetime.now().isoformat()
            }

            # Add statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                summary[f'{col}_mean'] = float(df[col].mean())
                summary[f'{col}_median'] = float(df[col].median())
                summary[f'{col}_std'] = float(df[col].std())
                summary[f'{col}_min'] = float(df[col].min())
                summary[f'{col}_max'] = float(df[col].max())

            return summary

        except Exception as e:
            self.logger.warning(f"Failed to calculate summary stats: {e}")
            return {'error': str(e)}

    def _convert_to_csv(self, jsonl_path: Path, output_dir: Path) -> Path:
        """
        Convert JSONL results to CSV format.

        Args:
            jsonl_path: Path to JSONL file
            output_dir: Directory to save CSV file

        Returns:
            Path to CSV file, or None if failed
        """
        try:
            if not jsonl_path.exists():
                return None

            results = []
            with open(jsonl_path, 'r') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))

            if not results:
                return None

            # Convert to DataFrame and save as CSV
            df = pd.DataFrame(results)
            csv_path = output_dir / 'results.csv'
            df.to_csv(csv_path, index=False)

            self.logger.debug(f"Converted results to CSV: {csv_path}")
            return csv_path

        except Exception as e:
            self.logger.error(f"Failed to convert to CSV: {e}", exc_info=True)
            return None

    def _compress_file(self, file_path: Path) -> Path:
        """
        Compress a file using gzip.

        Args:
            file_path: Path to file to compress

        Returns:
            Path to compressed file
        """
        compressed_path = file_path.parent / f"{file_path.name}.gz"

        with open(file_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                f_out.writelines(f_in)

        self.logger.debug(f"Compressed {file_path} to {compressed_path}")
        return compressed_path

    def download_results(
        self,
        s3_path: str,
        local_dir: Path,
        decompress: bool = True
    ) -> bool:
        """
        Download benchmark results from S3 to a local directory.

        Args:
            s3_path: S3 path (e.g., "benchmarks/performance/model_name/timestamp/")
            local_dir: Local directory to download to
            decompress: Whether to decompress .gz files after download

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure local directory exists
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)

            # Normalize S3 path
            if not s3_path.startswith(self.results_prefix):
                s3_path = f"{self.results_prefix}{s3_path}"
            if not s3_path.endswith('/'):
                s3_path += '/'

            self.logger.info(f"Downloading from s3://{self.bucket_name}/{s3_path} to {local_dir}")

            # List all objects in the S3 path
            paginator = self.s3_client.get_paginator('list_objects_v2')
            downloaded_files = []

            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=s3_path):
                for obj in page.get('Contents', []):
                    s3_key = obj['Key']
                    # Get relative path from the base s3_path
                    relative_path = s3_key[len(s3_path):]
                    if not relative_path:
                        continue

                    local_file_path = local_dir / relative_path
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)

                    self.logger.debug(f"Downloading {s3_key} to {local_file_path}")
                    self.s3_client.download_file(
                        self.bucket_name,
                        s3_key,
                        str(local_file_path)
                    )
                    downloaded_files.append(relative_path)

                    # Decompress .gz files if requested
                    if decompress and local_file_path.suffix == '.gz':
                        decompressed_path = local_file_path.with_suffix('')
                        with gzip.open(local_file_path, 'rb') as f_in:
                            with open(decompressed_path, 'wb') as f_out:
                                f_out.write(f_in.read())
                        self.logger.debug(f"Decompressed {local_file_path} to {decompressed_path}")

            self.logger.info(f"Downloaded {len(downloaded_files)} files to {local_dir}")
            return True

        except ClientError as e:
            self.logger.error(f"S3 download failed: {e}", exc_info=True)
            return False
        except Exception as e:
            self.logger.error(f"Failed to download results: {e}", exc_info=True)
            return False

    def get_latest_results_path(
        self,
        model_name: str,
        benchmark_type: str = "performance"
    ) -> str:
        """
        Get the S3 path to the latest results for a model.

        Args:
            model_name: Name of the model
            benchmark_type: Type of benchmark ("performance" or "quality")

        Returns:
            S3 path to latest results, or empty string if not found
        """
        try:
            safe_model_name = model_name.replace('/', '_').replace(' ', '_')
            prefix = f"{self.results_prefix}{benchmark_type}/{safe_model_name}/"

            # List all timestamp folders
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                Delimiter='/'
            )

            # Get common prefixes (folders)
            folders = [p['Prefix'] for p in response.get('CommonPrefixes', [])]

            if not folders:
                self.logger.warning(f"No results found for {model_name}")
                return ""

            # Sort by timestamp (folder names are timestamps) and get latest
            latest = sorted(folders)[-1]
            self.logger.info(f"Latest results path: {latest}")
            return latest

        except Exception as e:
            self.logger.error(f"Failed to get latest results path: {e}", exc_info=True)
            return ""

    def list_available_results(
        self,
        benchmark_type: str = None
    ) -> list:
        """
        List all available benchmark results in S3.

        Args:
            benchmark_type: Filter by benchmark type (None for all)

        Returns:
            List of available result paths
        """
        try:
            results = []

            if benchmark_type:
                prefixes = [f"{self.results_prefix}{benchmark_type}/"]
            else:
                prefixes = [
                    f"{self.results_prefix}performance/",
                    f"{self.results_prefix}quality/"
                ]

            for prefix in prefixes:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix,
                    Delimiter='/'
                )

                for p in response.get('CommonPrefixes', []):
                    model_prefix = p['Prefix']
                    # List timestamp folders under each model
                    model_response = self.s3_client.list_objects_v2(
                        Bucket=self.bucket_name,
                        Prefix=model_prefix,
                        Delimiter='/'
                    )
                    for tp in model_response.get('CommonPrefixes', []):
                        results.append(tp['Prefix'])

            return sorted(results)

        except Exception as e:
            self.logger.error(f"Failed to list results: {e}", exc_info=True)
            return []
