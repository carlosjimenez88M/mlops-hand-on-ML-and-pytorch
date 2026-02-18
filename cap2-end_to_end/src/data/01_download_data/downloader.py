"""
Class to handle download and storage ONLY in GCS (no local storage).
Author: Carlos Daniel Jiménez
Date: 2025-11-25
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from config import settings
from google.api_core import exceptions as gcp_exceptions
from google.cloud import storage

from models import (
    DownloadConfig,
    DownloadResult,
    FileStats,
    WandBArtifactMetadata,
)

# Try to import colored logger, fall back to basic logging if not available
try:
    # Add project root to path for colored logger
    project_root = Path(__file__).resolve().parents[4]
    utils_path = project_root / "src" / "utils"
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))

    from colored_logger import setup_colored_logger

    logger = setup_colored_logger(__name__)
except (ImportError, Exception):
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

# Constants
VALID_EXTENSIONS: Tuple[str, ...] = (".csv", ".parquet", ".json")
COMPRESSED_EXTENSIONS: Tuple[str, ...] = (".tgz", ".tar.gz")
MB_SIZE: int = 1024 * 1024
PROGRESS_LOG_INTERVAL: int = MB_SIZE


class DataDownloader:
    """
    Handles DIRECT download and storage in GCS (no local persistence).

    Attributes:
        config: Configuration for download operations
        storage_client: Google Cloud Storage client
        bucket: GCS bucket instance
    """

    def __init__(self, config: DownloadConfig) -> None:
        """
        Initialize DataDownloader with configuration.

        Args:
            config: Download configuration

        Raises:
            RuntimeError: If GCS initialization fails
        """
        self.config: DownloadConfig = config
        self.storage_client: Optional[storage.Client] = None
        self.bucket: Optional[storage.Bucket] = None

        self._init_gcs_client()

    def _init_gcs_client(self) -> None:
        """
        Initialize GCS client (REQUIRED).

        Raises:
            RuntimeError: If connection to GCS fails or bucket doesn't exist
        """
        try:
            # If GOOGLE_APPLICATION_CREDENTIALS is empty, unset it to use ADC
            import os

            if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") == "":
                os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(self.config.bucket_name)

            if not self.bucket.exists():
                raise ValueError(f"Bucket {self.config.bucket_name} does not exist")

            logger.info(f"Connected to GCS: gs://{self.config.bucket_name}")

        except Exception as e:
            logger.error(f"Error connecting to GCS: {e}")
            raise RuntimeError(
                "GCS is required for this component. Check your configuration and credentials."
            ) from e

    def download_to_memory(self) -> bytes:
        """
        Download the complete file into memory with retry logic.

        Returns:
            Content of the file as bytes

        Raises:
            requests.RequestException: If download fails after all retries
        """
        logger.info(f"Downloading from: {self.config.file_url}")

        last_exception: Optional[Exception] = None

        for attempt in range(settings.MAX_RETRIES):
            try:
                response = requests.get(
                    str(self.config.file_url), stream=True, timeout=settings.TIMEOUT
                )
                response.raise_for_status()

                content = self._download_chunks(response)

                logger.info(f"Downloaded to memory: {len(content) / MB_SIZE:.2f} MB")
                return content

            except requests.RequestException as e:
                last_exception = e
                if attempt < settings.MAX_RETRIES - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                else:
                    logger.error(f"Download failed after {settings.MAX_RETRIES} attempts")

        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        raise requests.RequestException("Download failed for unknown reason")

    def _download_chunks(self, response: requests.Response) -> bytes:
        """
        Download file in chunks with progress logging.

        Args:
            response: HTTP response object

        Returns:
            Downloaded content as bytes
        """
        content = b""
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        for chunk in response.iter_content(chunk_size=settings.CHUNK_SIZE):
            if chunk:
                content += chunk
                downloaded += len(chunk)

                # Log progress every 1MB
                if downloaded % PROGRESS_LOG_INTERVAL == 0:
                    progress = (downloaded / total_size * 100) if total_size > 0 else 0
                    logger.info(f"  Progress: {downloaded / MB_SIZE:.1f} MB ({progress:.1f}%)")

        return content

    def extract_if_compressed(self, content: bytes) -> bytes:
        """
        Extract file if compressed (in memory).

        Args:
            content: Compressed content in bytes

        Returns:
            Extracted content as bytes (or original if not compressed)

        Raises:
            FileNotFoundError: If no valid file found after extraction
        """
        url_str = str(self.config.file_url)

        if not self._is_compressed(url_str):
            return content

        logger.info("Extracting compressed file in memory...")

        return self._extract_tarball(content)

    @staticmethod
    def _is_compressed(url: str) -> bool:
        """Check if URL points to compressed file."""
        return any(url.endswith(ext) for ext in COMPRESSED_EXTENSIONS)

    def _extract_tarball(self, content: bytes) -> bytes:
        """
        Extract content from tar.gz archive.

        Args:
            content: Compressed tarball content

        Returns:
            Extracted file content

        Raises:
            FileNotFoundError: If no valid file found in tarball
        """
        compressed_path = self._write_temp_file(content)

        try:
            return self._read_from_tarball(compressed_path)
        finally:
            self._cleanup_temp_file(compressed_path)

    @staticmethod
    def _write_temp_file(content: bytes) -> str:
        """Write content to temporary file."""
        with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            return tmp_file.name

    def _read_from_tarball(self, tar_path: str) -> bytes:
        """
        Read valid file from tarball.

        Args:
            tar_path: Path to tarball file

        Returns:
            Extracted file content

        Raises:
            FileNotFoundError: If no valid file found
        """
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()

            for member in members:
                if self._is_valid_file(member.name):
                    extracted_file = tar.extractfile(member)
                    if extracted_file:
                        extracted_content = extracted_file.read()
                        logger.info(
                            f"Extracted: {member.name} ({len(extracted_content) / MB_SIZE:.2f} MB)"
                        )
                        return extracted_content

            raise FileNotFoundError(
                f"No file found with valid extensions {VALID_EXTENSIONS} in the tarball"
            )

    @staticmethod
    def _is_valid_file(filename: str) -> bool:
        """Check if filename has valid extension."""
        return any(filename.endswith(ext) for ext in VALID_EXTENSIONS)

    @staticmethod
    def _cleanup_temp_file(file_path: str) -> None:
        """Clean up temporary file."""
        try:
            os.remove(file_path)
        except OSError:
            pass

    def get_file_stats_from_bytes(self, content: bytes) -> FileStats:
        """
        Get statistics of content in memory.

        Args:
            content: File content in bytes

        Returns:
            FileStats with the statistics
        """
        file_size_mb = round(len(content) / MB_SIZE, 2)

        stats_dict: Dict[str, any] = {"file_size_mb": file_size_mb, "downloaded_at": datetime.now()}

        # Get data stats (works for CSV or Parquet)
        data_stats = self._get_data_stats(content)
        if data_stats:
            stats_dict.update(data_stats)

        return FileStats(**stats_dict)

    def _get_data_stats(self, content: bytes) -> Optional[Dict[str, any]]:
        """
        Get statistics from data content (CSV or Parquet).

        Args:
            content: Data content in bytes

        Returns:
            Dictionary with data statistics or None if failed
        """
        try:
            # Source content is CSV before optional Parquet conversion.
            df = pd.read_csv(io.BytesIO(content))

            logger.info(f"Data Stats: {len(df):,} rows, {len(df.columns)} columns")

            return {
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "columns": list(df.columns),
                "missing_values": df.isnull().sum().to_dict(),
            }

        except Exception as e:
            logger.warning(f"Could not get data stats: {e}")
            return None

    def _convert_to_parquet(self, csv_content: bytes) -> bytes:
        """
        Convert CSV content to Parquet format.

        Args:
            csv_content: CSV content in bytes

        Returns:
            Parquet content in bytes
        """
        logger.info("Converting CSV to Parquet format...")

        # Read full CSV and persist full dataset in Parquet.
        df = pd.read_csv(io.BytesIO(csv_content))

        # Convert to Parquet
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, engine="pyarrow")
        buffer.seek(0)

        parquet_content = buffer.getvalue()
        logger.info(f"Converted to Parquet: {len(parquet_content) / MB_SIZE:.2f} MB")

        return parquet_content

    def upload_to_gcs(self, content: bytes) -> str:
        """
        Upload content directly to GCS from memory.
        Converts to Parquet if output path ends with .parquet

        Args:
            content: Content in bytes (CSV format from source)

        Returns:
            Full GCS URI

        Raises:
            RuntimeError: If GCS client not initialized or upload fails
        """
        if not self.storage_client or not self.bucket:
            raise RuntimeError("GCS client not initialized")

        gcs_path = self.config.gcs_output_path

        # Convert to Parquet if needed
        upload_content = content
        if gcs_path.endswith(".parquet"):
            upload_content = self._convert_to_parquet(content)

        logger.info(f"Uploading to GCS: gs://{self.config.bucket_name}/{gcs_path}")

        try:
            blob = self.bucket.blob(gcs_path)

            # Set metadata
            blob.metadata = self._create_blob_metadata(upload_content)

            # Upload from bytes
            blob.upload_from_string(upload_content)

            gcs_uri = f"gs://{self.config.bucket_name}/{gcs_path}"
            logger.info(f"Uploaded to GCS: {gcs_uri}")

            return gcs_uri

        except gcp_exceptions.GoogleAPIError as e:
            logger.error(f"Error uploading to GCS: {e}")
            raise RuntimeError(f"Error uploading to GCS: {e}") from e

    def _create_blob_metadata(self, content: bytes) -> Dict[str, str]:
        """Create metadata dictionary for GCS blob."""
        return {
            "uploaded_at": datetime.now().isoformat(),
            "source": "mlflow_component",
            "component": "01_download_data",
            "original_url": str(self.config.file_url),
            "storage_type": "gcs_only",
            "file_size_mb": str(round(len(content) / MB_SIZE, 2)),
        }

    def run(self) -> DownloadResult:
        """
        Execute the complete download process DIRECTLY to GCS.

        Returns:
            DownloadResult with result information
        """
        try:
            # 1. Download to memory
            content = self.download_to_memory()

            # 2. Extract if necessary (in memory)
            extracted_content = self.extract_if_compressed(content)

            # 3. Get statistics
            stats = self.get_file_stats_from_bytes(extracted_content)

            # 4. Upload directly to GCS
            gcs_uri = self.upload_to_gcs(extracted_content)

            logger.info("Data stored ONLY in GCS (no local copy)")

            return DownloadResult(
                gcs_uri=gcs_uri, stats=stats, artifact_name=self.config.artifact_name, success=True
            )

        except Exception as e:
            logger.error(f"Error in download: {e}", exc_info=True)

            return self._create_error_result(e)

    def _create_error_result(self, error: Exception) -> DownloadResult:
        """Create error result when download fails."""
        error_uri = f"gs://{self.config.bucket_name}/{self.config.gcs_output_path}"

        return DownloadResult(
            gcs_uri=error_uri,
            stats=FileStats(file_size_mb=0.0, downloaded_at=datetime.now()),
            artifact_name=self.config.artifact_name,
            success=False,
            error_message=str(error),
        )

    def create_wandb_metadata(self, result: DownloadResult) -> WandBArtifactMetadata:
        """
        Create metadata for W&B artifact.

        Args:
            result: Download result

        Returns:
            W&B artifact metadata
        """
        return WandBArtifactMetadata(
            original_url=str(self.config.file_url),
            gcs_uri=result.gcs_uri,
            bucket=self.config.bucket_name,
            file_size_mb=result.stats.file_size_mb,
            n_rows=result.stats.n_rows,
            n_columns=result.stats.n_columns,
            downloaded_at=result.stats.downloaded_at.isoformat(),
        )
