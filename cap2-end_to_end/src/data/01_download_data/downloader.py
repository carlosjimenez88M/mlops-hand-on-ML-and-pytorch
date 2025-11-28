"""
Class to handle download and storage ONLY in GCS (no local storage).
Author: Carlos Daniel Hernandez
Date: 2025-11-25
"""

#=======================#
# ----- libraries ----- #
#=======================#
import io
import logging
import os
import tarfile
import tempfile
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from google.api_core import exceptions as gcp_exceptions
from google.cloud import storage
from config import settings

from models import (
    DownloadConfig,
    FileStats,
    DownloadResult,
    WandBArtifactMetadata
)

#================================#
# ---- logger configuration ---- #
#================================#
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger()

#==============================#
# ----- Class Definition ----- #
#==============================#

class DataDownloader:
    """Handles DIRECT download 
    and storage in GCS 
    (no local persistence)."""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.storage_client: Optional[storage.Client] = None
        self.bucket: Optional[storage.Bucket] = None
        
        self._init_gcs_client()
    
    def _init_gcs_client(self) -> None:
        """Initializes GCS client (REQUIRED)."""
        try:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(self.config.bucket_name)
            
            # Verify bucket exists
            if not self.bucket.exists():
                raise ValueError(f"Bucket {self.config.bucket_name} does not exist")
            
            logger.info(f"✓ Connected to GCS: gs://{self.config.bucket_name}")
            
        except Exception as e:
            logger.error(f"❌ Error connecting to GCS: {e}")
            raise RuntimeError(
                "GCS is required for this component. "
                "Check your configuration and credentials."
            ) from e
    
    def download_to_memory(self) -> bytes:
        """
        Downloads the complete file into memory.
        
        Returns:
            bytes: Content of the file.
        
        Raises:
            requests.RequestException: If the download fails.
        """
        logger.info(f"Downloading from: {self.config.file_url}")
        
        # Download with retry
        for attempt in range(settings.MAX_RETRIES):
            try:
                response = requests.get(
                    str(self.config.file_url),
                    stream=True,
                    timeout=settings.TIMEOUT
                )
                response.raise_for_status()
                
                # Read everything into memory
                content = b""
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                for chunk in response.iter_content(chunk_size=settings.CHUNK_SIZE):
                    if chunk:
                        content += chunk
                        downloaded += len(chunk)
                        
                        # Log progress every 1MB
                        if downloaded % (1024 * 1024) == 0:
                            progress = (downloaded / total_size * 100) if total_size > 0 else 0
                            logger.info(
                                f"  Progress: {downloaded / (1024*1024):.1f} MB "
                                f"({progress:.1f}%)"
                            )
                
                logger.info(f"✓ Downloaded to memory: {len(content) / (1024*1024):.2f} MB")
                return content
                
            except requests.RequestException as e:
                if attempt < settings.MAX_RETRIES - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                else:
                    logger.error(f"Download failed after {settings.MAX_RETRIES} attempts")
                    raise
    
    def extract_if_compressed(self, content: bytes) -> bytes:
        """
        Extracts file if compressed (in memory).
        
        Args:
            content: Compressed content in bytes.
        
        Returns:
            bytes: Extracted content.
        
        Raises:
            FileNotFoundError: If no valid file is found after extraction.
        """
        # Check if it is tar.gz
        url_str = str(self.config.file_url)
        if not (url_str.endswith('.tgz') or url_str.endswith('.tar.gz')):
            return content
        
        logger.info("Extracting compressed file in memory...")
        
        # Use temporary file for extraction
        with tempfile.NamedTemporaryFile(suffix='.tgz', delete=False) as tmp_compressed:
            tmp_compressed.write(content)
            tmp_compressed.flush()
            compressed_path = tmp_compressed.name
        
        try:
            with tarfile.open(compressed_path, 'r:gz') as tar:
                members = tar.getmembers()
                
                # Look for file with valid extension
                valid_extensions = ['.csv', '.parquet', '.json']
                extracted_content = None
                
                for member in members:
                    if any(member.name.endswith(ext) for ext in valid_extensions):
                        extracted_file = tar.extractfile(member)
                        if extracted_file:
                            extracted_content = extracted_file.read()
                            logger.info(f"✓ Extracted: {member.name} ({len(extracted_content) / (1024*1024):.2f} MB)")
                            break
                
                if not extracted_content:
                    raise FileNotFoundError(
                        f"No file found with valid extensions {valid_extensions} "
                        f"in the tarball"
                    )
                
                return extracted_content
                
        finally:
            # Clean up temporary file
            try:
                os.remove(compressed_path)
            except OSError:
                pass
    
    def get_file_stats_from_bytes(self, content: bytes) -> FileStats:
        """
        Gets statistics of content in memory.
        
        Args:
            content: File content in bytes.
        
        Returns:
            FileStats with the statistics.
        """
        file_size_mb = round(len(content) / (1024 * 1024), 2)
        
        stats_dict = {
            'file_size_mb': file_size_mb,
            'downloaded_at': datetime.now()
        }
        
        # If CSV, get additional stats
        gcs_path = self.config.gcs_output_path
        if gcs_path.endswith('.csv'):
            try:
                # Read CSV from bytes
                df = pd.read_csv(io.BytesIO(content))
                
                stats_dict.update({
                    'n_rows': len(df),
                    'n_columns': len(df.columns),
                    'columns': list(df.columns),
                    'missing_values': df.isnull().sum().to_dict()
                })
                
                logger.info(f"✓ CSV Stats: {len(df):,} rows, {len(df.columns)} columns")
                
            except Exception as e:
                logger.warning(f"Could not get CSV stats: {e}")
        
        return FileStats(**stats_dict)
    
    def upload_to_gcs(self, content: bytes) -> str:
        """
        Uploads content directly to GCS from memory.
        
        Args:
            content: Content in bytes.
        
        Returns:
            str: Full GCS URI.
        
        Raises:
            RuntimeError: If upload fails.
        """
        if not self.storage_client or not self.bucket:
            raise RuntimeError("GCS client not initialized")
        
        gcs_path = self.config.gcs_output_path
        
        logger.info(f"Uploading to GCS: gs://{self.config.bucket_name}/{gcs_path}")
        
        try:
            blob = self.bucket.blob(gcs_path)
            
            # Metadata
            blob.metadata = {
                "uploaded_at": datetime.now().isoformat(),
                "source": "mlflow_component",
                "component": "01_download_data",
                "original_url": str(self.config.file_url),
                "storage_type": "gcs_only",
                "file_size_mb": str(round(len(content) / (1024 * 1024), 2))
            }
            
            # Upload from bytes
            blob.upload_from_string(content)
            
            gcs_uri = f"gs://{self.config.bucket_name}/{gcs_path}"
            logger.info(f"✓ Uploaded to GCS: {gcs_uri}")
            
            return gcs_uri
            
        except gcp_exceptions.GoogleAPIError as e:
            logger.error(f"Error uploading to GCS: {e}")
            raise RuntimeError(f"Error uploading to GCS: {e}") from e
    
    def run(self) -> DownloadResult:
        """
        Executes the complete download process DIRECTLY to GCS.
        
        Returns:
            DownloadResult with result information.
        
        Raises:
            Exception: If any error occurs in the process.
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
            
            logger.info("✓ Data stored ONLY in GCS (no local copy)")
            
            return DownloadResult(
                gcs_uri=gcs_uri,
                stats=stats,
                artifact_name=self.config.artifact_name,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in download: {e}", exc_info=True)
            
            # Construct error URI
            error_uri = f"gs://{self.config.bucket_name}/{self.config.gcs_output_path}"
            
            return DownloadResult(
                gcs_uri=error_uri,
                stats=FileStats(file_size_mb=0),
                artifact_name=self.config.artifact_name,
                success=False,
                error_message=str(e)
            )
    
    def create_wandb_metadata(self, result: DownloadResult) -> WandBArtifactMetadata:
        """Creates metadata for W&B artifact."""
        return WandBArtifactMetadata(
            original_url=str(self.config.file_url),
            gcs_uri=result.gcs_uri,
            bucket=self.config.bucket_name,
            file_size_mb=result.stats.file_size_mb,
            n_rows=result.stats.n_rows,
            n_columns=result.stats.n_columns,
            downloaded_at=result.stats.downloaded_at.isoformat()
        )