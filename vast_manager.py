"""
Vast.ai Instance Manager
Handles instance provisioning, command execution, and resource management.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, Tuple, List

from vastai_sdk import VastAI
from ssh_helper import SSHConnection


class VastManager:
    """Manager for Vast.ai instance operations"""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize Vast.ai manager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger

        # Initialize Vast.ai SDK
        api_key = os.getenv('VAST_API_KEY')
        if not api_key:
            raise ValueError("VAST_API_KEY not found in environment variables")

        self.sdk = VastAI(api_key=api_key)

        # Cache for instance SSH connections
        self._ssh_connections: Dict[int, SSHConnection] = {}

    def create_instance(
        self,
        gpu_name: str,
        num_gpus: int,
        disk_space: int,
        image: str,
        min_download_speed: float = 1000.0,
        blacklist_machines: Optional[List[int]] = None
    ) -> Tuple[int, str]:
        """
        Create a Vast.ai instance with specified configuration.

        Args:
            gpu_name: GPU type (e.g., "RTX_4090")
            num_gpus: Number of GPUs
            disk_space: Disk space in GB
            image: Docker image to use
            blacklist_machines: List of machine IDs to avoid
            min_download_speed: Minimum download speed in Mbps (default 200)

        Returns:
            Tuple of (instance_id, instance_ip)
        """
        self.logger.info(f"Creating instance: {gpu_name} x{num_gpus}, disk={disk_space}GB, min_download={min_download_speed}Mbps")

        try:
            # Search for all offers - the SDK's gpu_name filter doesn't work reliably
            # so we fetch more offers and filter manually
            self.logger.info(f"Searching for {gpu_name} with CUDA 12.9+...")

            offers = self.sdk.search_offers(
                disk_space=str(disk_space),
                order="dph_total",  # Cheapest first
                limit=2000  # Get more offers to find the GPU we need
            )

            # Filter for CUDA 12.9+, exact GPU name match, exact GPU count, AND minimum download speed
            cuda_offers = []
            blacklist = set(blacklist_machines or [])
            if isinstance(offers, list):
                for offer in offers:
                    cuda_ver = offer.get('cuda_max_good', 0)
                    offer_gpu = offer.get('gpu_name', '')
                    offer_num_gpus = offer.get('num_gpus', 0)
                    inet_down = offer.get('inet_down', 0) or 0  # Download speed in Mbps
                    machine_id = offer.get('machine_id', 0)
                    # Skip blacklisted machines
                    if machine_id in blacklist:
                        self.logger.debug(f"Skipping blacklisted machine: {machine_id}")
                        continue
                    # Exact GPU name match (e.g., "RTX 4090" should not match "RTX 5090")
                    # AND exact GPU count match
                    # AND minimum download speed requirement
                    if (cuda_ver and float(cuda_ver) >= 12.9
                        and offer_gpu == gpu_name
                        and offer_num_gpus == num_gpus
                        and inet_down >= min_download_speed):
                        cuda_offers.append(offer)

            if not cuda_offers:
                raise ValueError(f"No offers found for {num_gpus}x {gpu_name} with CUDA 12.9+ and >{min_download_speed}Mbps download")

            self.logger.info(f"Found {len(cuda_offers)} offers with CUDA 12.9+ and >{min_download_speed}Mbps, selecting best one")
            best_offer = cuda_offers[0]
            offer_id = best_offer.get('id')
            offer_inet_down = best_offer.get('inet_down', 0)
            offer_inet_up = best_offer.get('inet_up', 0)
            self.logger.info(f"Selected offer: ID={offer_id}, download={offer_inet_down}Mbps, upload={offer_inet_up}Mbps")

            # Create instance from the best offer
            response = self.sdk.create_instance(
                id=offer_id,
                image=image,
                disk=disk_space
            )

            # Extract instance information from response
            if isinstance(response, dict):
                instance_id = response.get('new_contract')
                # Get instance details to retrieve IP
                time.sleep(5)  # Wait for instance to initialize
                instance_info = self.sdk.show_instance(id=instance_id)

                if isinstance(instance_info, dict):
                    instance_ip = instance_info.get('public_ipaddr')
                else:
                    # Parse from list response
                    instance_ip = instance_info[0].get('public_ipaddr') if instance_info else None

                self.logger.info(f"Instance created: ID={instance_id}, IP={instance_ip}")
                return instance_id, instance_ip
            else:
                raise ValueError(f"Unexpected response format: {response}")

        except Exception as e:
            self.logger.error(f"Failed to create instance: {e}", exc_info=True)
            raise

    def wait_for_ssh(self, instance_id: int, timeout: int = 300) -> bool:
        """
        Wait for SSH to be available on the instance.

        Args:
            instance_id: Instance ID
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if SSH is available, False if timeout
        """
        self.logger.info(f"Waiting for SSH on instance {instance_id}")

        start_time = time.time()
        check_interval = 10

        while time.time() - start_time < timeout:
            try:
                # Check instance status
                instance_info = self.sdk.show_instance(id=instance_id)

                # Handle both dict and list responses
                if isinstance(instance_info, dict):
                    inst_data = instance_info
                elif isinstance(instance_info, list) and instance_info:
                    inst_data = instance_info[0]
                else:
                    self.logger.debug(f"Unexpected instance_info format: {type(instance_info)}")
                    time.sleep(check_interval)
                    continue

                status = inst_data.get('actual_status', '')
                ssh_port = inst_data.get('ssh_port')

                self.logger.info(f"Instance status: {status}, SSH port: {ssh_port}")

                # Try SSH test if SSH port is available, regardless of status
                # SSH can work in 'loading' or 'running' states
                if ssh_port:
                    self.logger.info(f"SSH port detected, testing SSH connectivity (status: {status})...")
                    try:
                        # Try to establish actual SSH connection
                        ssh = self._get_ssh_connection(instance_id)
                        if ssh:
                            # Test with simple command
                            success, stdout, stderr, exit_code = ssh.execute_command("echo 'SSH_TEST_OK'", timeout=10)
                            if success:
                                self.logger.info(f"SSH is available and working! Response: {stdout.strip()}")
                                time.sleep(5)  # Brief wait for stability
                                return True
                            else:
                                self.logger.debug(f"SSH command test failed: exit_code={exit_code}")
                        else:
                            self.logger.debug("SSH connection failed (will retry)")
                    except Exception as ssh_test_error:
                        self.logger.debug(f"SSH test failed (will retry): {ssh_test_error}")

            except Exception as e:
                self.logger.warning(f"SSH check failed: {e}", exc_info=True)

            time.sleep(check_interval)

        self.logger.error(f"SSH timeout after {timeout}s")
        return False

    def _get_ssh_connection(self, instance_id: int, quiet: bool = False) -> Optional[SSHConnection]:
        """
        Get or create SSH connection for instance.

        Args:
            instance_id: Instance ID
            quiet: If True, don't log errors on failure

        Returns:
            SSH connection or None if failed
        """
        # Check if existing connection is still alive
        if instance_id in self._ssh_connections:
            ssh = self._ssh_connections[instance_id]
            if ssh.is_alive():
                return ssh
            else:
                # Connection is dead, remove it and reconnect
                if not quiet:
                    self.logger.info(f"SSH connection for instance {instance_id} is stale, reconnecting...")
                try:
                    ssh.close()
                except Exception:
                    pass
                del self._ssh_connections[instance_id]

        # Get instance details
        try:
            instance_info = self.sdk.show_instance(id=instance_id)

            # Handle both dict and list responses
            if isinstance(instance_info, dict):
                inst_data = instance_info
            elif isinstance(instance_info, list) and instance_info:
                inst_data = instance_info[0]
            else:
                if not quiet:
                    self.logger.error(f"Could not get instance info for {instance_id}")
                return None

            # Extract SSH details
            # Vast.ai uses SSH proxy hosts (ssh2.vast.ai, ssh3.vast.ai, etc.)
            ssh_host = inst_data.get('ssh_host')
            port = inst_data.get('ssh_port')

            if not ssh_host or not port:
                if not quiet:
                    self.logger.error(f"Missing SSH details: ssh_host={ssh_host}, port={port}")
                return None

            if not quiet:
                self.logger.info(f"Connecting to Vast.ai SSH proxy: {ssh_host}:{port}")

            # Create SSH connection using proxy host
            ssh = SSHConnection(host=ssh_host, port=port, logger=self.logger)

            if ssh.connect(quiet=quiet):
                self._ssh_connections[instance_id] = ssh
                return ssh
            else:
                return None

        except Exception as e:
            if not quiet:
                self.logger.error(f"Failed to create SSH connection: {e}", exc_info=True)
            return None

    def execute_command(
        self,
        instance_id: int,
        command: str,
        background: bool = False,
        quiet: bool = False
    ) -> bool:
        """
        Execute a command on the instance via SSH.

        Args:
            instance_id: Instance ID
            command: Command to execute
            background: Whether to run in background
            quiet: If True, don't log errors on failure (useful for health checks)

        Returns:
            bool: True if successful, False otherwise
        """
        # Get SSH connection
        ssh = self._get_ssh_connection(instance_id, quiet=quiet)
        if not ssh:
            if not quiet:
                self.logger.error(f"Failed to get SSH connection for instance {instance_id}")
            return False

        try:
            if background:
                # For background commands, append & and nohup
                command = f"nohup {command} > /dev/null 2>&1 &"

            # Execute command over SSH
            success, stdout, stderr, exit_code = ssh.execute_command(command, timeout=300, quiet=quiet)

            if success:
                if stdout:
                    self.logger.debug(f"Command output (first 200 chars): {stdout[:200]}")
                return True
            else:
                if not quiet:
                    self.logger.error(f"Command failed with exit code {exit_code}")
                    if stderr:
                        self.logger.error(f"STDERR (first 500 chars): {stderr[:500]}")
                    if stdout:
                        self.logger.error(f"STDOUT (first 500 chars): {stdout[:500]}")
                    # Also log last part of output which often contains the error
                    if stdout and len(stdout) > 500:
                        self.logger.error(f"STDOUT (last 500 chars): {stdout[-500:]}")
                return False

        except Exception as e:
            if not quiet:
                self.logger.error(f"Command execution failed: {e}", exc_info=True)
            return False

    def execute_command_with_output(
        self,
        instance_id: int,
        command: str,
        quiet: bool = False
    ) -> Optional[str]:
        """
        Execute a command on the instance via SSH and return stdout.

        Args:
            instance_id: Instance ID
            command: Command to execute
            quiet: If True, don't log errors on failure

        Returns:
            stdout string if successful, None otherwise
        """
        ssh = self._get_ssh_connection(instance_id, quiet=quiet)
        if not ssh:
            if not quiet:
                self.logger.error(f"Failed to get SSH connection for instance {instance_id}")
            return None

        try:
            success, stdout, stderr, exit_code = ssh.execute_command(command, timeout=300, quiet=quiet)

            if success:
                return stdout
            else:
                if not quiet:
                    self.logger.error(f"Command failed with exit code {exit_code}")
                    if stderr:
                        self.logger.error(f"STDERR: {stderr[:500]}")
                return None

        except Exception as e:
            if not quiet:
                self.logger.error(f"Command execution failed: {e}", exc_info=True)
            return None

    def execute_command_streaming(
        self,
        instance_id: int,
        command: str,
        timeout: int = 3600,
        log_file: Optional[str] = None
    ) -> bool:
        """
        Execute a command on the instance via SSH with streaming output.

        Args:
            instance_id: Instance ID
            command: Command to execute
            timeout: Timeout in seconds (default 1 hour)
            log_file: Optional remote path to also write output to

        Returns:
            bool: True if successful, False otherwise
        """
        from rich.console import Console
        console = Console()

        # Get SSH connection
        ssh = self._get_ssh_connection(instance_id)
        if not ssh:
            self.logger.error(f"Failed to get SSH connection for instance {instance_id}")
            return False

        try:
            # If log_file specified, use tee to write to both stdout and file
            if log_file:
                command = f"({command}) 2>&1 | tee {log_file}"

            # Get the underlying paramiko client
            client = ssh.client

            # Open a channel for streaming
            transport = client.get_transport()
            channel = transport.open_session()
            channel.settimeout(timeout)
            channel.exec_command(command)

            # Stream output
            success = True
            while True:
                # Check if channel has data
                if channel.recv_ready():
                    data = channel.recv(4096).decode('utf-8', errors='replace')
                    if data:
                        # Print each line with prefix
                        for line in data.splitlines():
                            console.print(f"[dim]│[/dim] {line}")

                if channel.recv_stderr_ready():
                    data = channel.recv_stderr(4096).decode('utf-8', errors='replace')
                    if data:
                        for line in data.splitlines():
                            console.print(f"[red]│[/red] {line}")

                # Check if command completed
                if channel.exit_status_ready():
                    # Drain remaining output
                    while channel.recv_ready():
                        data = channel.recv(4096).decode('utf-8', errors='replace')
                        if data:
                            for line in data.splitlines():
                                console.print(f"[dim]│[/dim] {line}")
                    while channel.recv_stderr_ready():
                        data = channel.recv_stderr(4096).decode('utf-8', errors='replace')
                        if data:
                            for line in data.splitlines():
                                console.print(f"[red]│[/red] {line}")
                    break

                import time
                time.sleep(0.1)

            exit_code = channel.recv_exit_status()
            channel.close()

            if exit_code != 0:
                self.logger.error(f"Command failed with exit code {exit_code}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Streaming command execution failed: {e}", exc_info=True)
            return False

    def get_file_content(self, instance_id: int, file_path: str, timeout: int = 600) -> Optional[str]:
        """
        Retrieve file content from the instance via SSH.

        Args:
            instance_id: Instance ID
            file_path: Path to file on the instance
            timeout: Timeout in seconds (default 600s / 10 min for large files)

        Returns:
            File content as string, or None if failed
        """
        self.logger.debug(f"Retrieving file {file_path} from instance {instance_id}")

        try:
            # Use SSH to cat file content (more reliable than SDK execute)
            ssh = self._get_ssh_connection(instance_id)
            if not ssh:
                self.logger.error(f"No SSH connection for instance {instance_id}")
                return None

            success, stdout, stderr, exit_code = ssh.execute_command(f"cat '{file_path}'", timeout=timeout)

            if success and exit_code == 0:
                return stdout
            else:
                self.logger.debug(f"File not found or error: {file_path} (exit={exit_code})")
                return None

        except Exception as e:
            self.logger.error(f"Failed to retrieve file {file_path}: {e}", exc_info=True)
            return None

    def download_file(self, instance_id: int, remote_path: str, local_path: str) -> bool:
        """
        Download a file from the instance using SFTP (faster than cat for large files).

        Args:
            instance_id: Instance ID
            remote_path: Path to file on the instance
            local_path: Local path to save file

        Returns:
            True if successful, False otherwise
        """
        self.logger.debug(f"Downloading {remote_path} from instance {instance_id}")

        try:
            ssh = self._get_ssh_connection(instance_id)
            if not ssh:
                self.logger.error(f"No SSH connection for instance {instance_id}")
                return False

            return ssh.download_file(remote_path, local_path)

        except Exception as e:
            self.logger.error(f"Failed to download file {remote_path}: {e}", exc_info=True)
            return False

    def run_command(self, instance_id: int, command: str) -> Optional[str]:
        """
        Run an arbitrary command on the instance via SSH and return stdout.

        Args:
            instance_id: Instance ID
            command: Command to execute

        Returns:
            Command stdout as string, or None if failed
        """
        self.logger.debug(f"Running command on instance {instance_id}: {command[:50]}...")

        try:
            ssh = self._get_ssh_connection(instance_id)
            if not ssh:
                self.logger.error(f"No SSH connection for instance {instance_id}")
                return None

            success, stdout, stderr, exit_code = ssh.execute_command(command, timeout=60)

            if success and exit_code == 0:
                return stdout
            else:
                self.logger.debug(f"Command failed: {command[:50]}... (exit={exit_code})")
                return None

        except Exception as e:
            self.logger.error(f"Failed to run command: {e}", exc_info=True)
            return None

    def get_instance_logs(self, instance_id: int, tail: Optional[int] = None) -> str:
        """
        Get instance logs.

        Args:
            instance_id: Instance ID
            tail: Number of lines to retrieve (None for all)

        Returns:
            Log content as string
        """
        self.logger.debug(f"Retrieving logs for instance {instance_id}")

        try:
            logs = self.sdk.logs(instance_id=instance_id, tail=tail)
            return logs if isinstance(logs, str) else str(logs)

        except Exception as e:
            self.logger.error(f"Failed to retrieve logs: {e}", exc_info=True)
            return ""

    def tail_file(self, instance_id: int, file_path: str, last_position: int = 0) -> Tuple[str, int]:
        """
        Tail a file from a given position, returning new content and updated position.

        Args:
            instance_id: Instance ID
            file_path: Path to file on the instance
            last_position: Byte position to start reading from

        Returns:
            Tuple of (new_content, new_position)
        """
        try:
            ssh = self._get_ssh_connection(instance_id, quiet=True)
            if not ssh:
                return "", last_position

            # Get file size first (quiet=True since this is called frequently during polling)
            size_cmd = f"stat -c %s '{file_path}' 2>/dev/null || echo '0'"
            success, stdout, stderr, exit_code = ssh.execute_command(size_cmd, timeout=10, quiet=True)

            if not success or not stdout.strip():
                return "", last_position

            try:
                file_size = int(stdout.strip())
            except ValueError:
                return "", last_position

            if file_size <= last_position:
                return "", last_position

            # Read from last position
            # Using dd to read specific bytes
            bytes_to_read = file_size - last_position
            read_cmd = f"dd if='{file_path}' bs=1 skip={last_position} count={bytes_to_read} 2>/dev/null"
            success, stdout, stderr, exit_code = ssh.execute_command(read_cmd, timeout=30, quiet=True)

            if success:
                return stdout, file_size
            else:
                return "", last_position

        except Exception as e:
            self.logger.debug(f"Failed to tail file {file_path}: {e}")
            return "", last_position

    def destroy_instance(self, instance_id: int) -> bool:
        """
        Terminate and destroy an instance.

        Args:
            instance_id: Instance ID

        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info(f"Destroying instance {instance_id}")

        try:
            response = self.sdk.destroy_instance(id=instance_id)
            self.logger.info(f"Instance {instance_id} destroyed: {response}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to destroy instance: {e}", exc_info=True)
            return False

    def get_instance_info(self, instance_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an instance.

        Args:
            instance_id: Instance ID

        Returns:
            Dictionary with instance information, or None if failed
        """
        try:
            info = self.sdk.show_instance(id=instance_id)

            if isinstance(info, list) and info:
                return info[0]
            elif isinstance(info, dict):
                return info
            else:
                return None

        except Exception as e:
            self.logger.error(f"Failed to get instance info: {e}", exc_info=True)
            return None
