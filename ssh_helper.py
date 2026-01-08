"""
SSH Helper for Vast.ai Instances
Provides direct SSH connectivity using paramiko.
"""

import os
import logging
import paramiko
from typing import Tuple, Optional
from pathlib import Path


class SSHConnection:
    """Manages SSH connections to Vast.ai instances"""

    def __init__(self, host: str, port: int, username: str = "root", logger: Optional[logging.Logger] = None):
        """
        Initialize SSH connection.

        Args:
            host: SSH host (IP address)
            port: SSH port
            username: SSH username (default: root for vast.ai)
            logger: Optional logger
        """
        self.host = host
        self.port = port
        self.username = username
        self.logger = logger or logging.getLogger(__name__)
        self.client = None

    def connect(self, ssh_key_path: Optional[str] = None, timeout: int = 30, quiet: bool = False) -> bool:
        """
        Establish SSH connection.

        Args:
            ssh_key_path: Path to SSH private key (default: tries ed25519 then RSA)
            timeout: Connection timeout in seconds
            quiet: If True, don't log errors on failure (useful for retry loops)

        Returns:
            bool: True if connected successfully
        """
        try:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Try ed25519 key first (vast.ai default), then RSA
            key_paths = []
            if ssh_key_path:
                key_paths.append(ssh_key_path)
            else:
                key_paths.extend([
                    os.path.expanduser("~/.ssh/id_ed25519"),
                    os.path.expanduser("~/.ssh/id_rsa")
                ])

            key = None
            for key_path in key_paths:
                if Path(key_path).exists():
                    try:
                        # Try ed25519 first
                        if 'ed25519' in key_path:
                            key = paramiko.Ed25519Key.from_private_key_file(key_path)
                            self.logger.info(f"Loaded ed25519 key from {key_path}")
                        else:
                            key = paramiko.RSAKey.from_private_key_file(key_path)
                            self.logger.info(f"Loaded RSA key from {key_path}")
                        break
                    except Exception as key_error:
                        self.logger.debug(f"Could not load key from {key_path}: {key_error}")
                        continue

            self.logger.info(f"Connecting to {self.username}@{self.host}:{self.port}")

            if key:
                self.client.connect(
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    pkey=key,
                    timeout=timeout,
                    look_for_keys=True,
                    allow_agent=True
                )
            else:
                # Try without specific key (use agent)
                self.logger.info("No key file found, trying SSH agent")
                self.client.connect(
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    timeout=timeout,
                    look_for_keys=True,
                    allow_agent=True
                )

            # Enable SSH keepalive to prevent connection drops
            transport = self.client.get_transport()
            if transport:
                transport.set_keepalive(30)  # Send keepalive every 30 seconds

            self.logger.info(f"SSH connected to {self.host}:{self.port}")
            return True

        except Exception as e:
            if not quiet:
                self.logger.error(f"SSH connection failed: {e}")
            if self.client:
                self.client.close()
                self.client = None
            return False

    def is_alive(self) -> bool:
        """Check if the SSH connection is still alive."""
        if not self.client:
            return False
        try:
            transport = self.client.get_transport()
            if transport is None or not transport.is_active():
                return False
            # Send a keepalive to verify connection is truly alive
            transport.send_ignore()
            return True
        except Exception:
            return False

    def execute_command(self, command: str, timeout: int = 300, quiet: bool = False) -> Tuple[bool, str, str, int]:
        """
        Execute command over SSH.

        Args:
            command: Command to execute
            timeout: Command timeout in seconds
            quiet: If True, don't log errors on failure (useful for health checks)

        Returns:
            Tuple of (success, stdout, stderr, exit_code)
        """
        if not self.client:
            return False, "", "SSH not connected", -1

        try:
            stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
            exit_code = stdout.channel.recv_exit_status()

            stdout_text = stdout.read().decode('utf-8', errors='replace')
            stderr_text = stderr.read().decode('utf-8', errors='replace')

            success = (exit_code == 0)

            if not success and not quiet:
                self.logger.error(f"Command failed with exit code {exit_code}")
                if stderr_text:
                    self.logger.error(f"STDERR: {stderr_text[:500]}")

            return success, stdout_text, stderr_text, exit_code

        except Exception as e:
            if not quiet:
                self.logger.error(f"Command execution failed: {e}")
            return False, "", str(e), -1

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """
        Download a file from the remote host using SFTP.

        Args:
            remote_path: Path to file on remote host
            local_path: Local path to save file

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            self.logger.error("SSH not connected")
            return False

        try:
            sftp = self.client.open_sftp()
            sftp.get(remote_path, local_path)
            sftp.close()
            return True
        except FileNotFoundError:
            self.logger.debug(f"Remote file not found: {remote_path}")
            return False
        except Exception as e:
            self.logger.error(f"SFTP download failed: {e}")
            return False

    def close(self):
        """Close SSH connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.logger.info("SSH connection closed")
