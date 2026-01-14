"""
Unit tests for DataDownloader class (01_download_data module)
Author: Carlos Daniel Jiménez
Date: 2025-01-13
"""

import io
import sys
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, Mock
from datetime import datetime

import pytest
import pandas as pd
import requests
from google.api_core import exceptions as gcp_exceptions

# Add module path
download_module_path = Path(__file__).parent.parent / "src" / "data" / "01_download_data"
sys.path.insert(0, str(download_module_path))

from models import DownloadConfig, FileStats, DownloadResult
from downloader import DataDownloader, VALID_EXTENSIONS, COMPRESSED_EXTENSIONS


class TestDataDownloaderModule:
    """Test suite for DataDownloader class."""

    def test_init_success(self, download_config_dict, mock_gcs_client, monkeypatch):
        """Test successful initialization with GCS."""
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        config = DownloadConfig(**download_config_dict)
        downloader = DataDownloader(config)

        assert downloader.config == config
        assert downloader.storage_client is not None
        assert downloader.bucket is not None

    def test_init_bucket_not_exists(self, download_config_dict, mock_gcs_client, monkeypatch):
        """Test initialization fails when bucket doesn't exist."""
        mock_gcs_client['bucket'].exists.return_value = False
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        config = DownloadConfig(**download_config_dict)

        with pytest.raises(RuntimeError, match="GCS is required"):
            DataDownloader(config)

    def test_download_to_memory_success(self, download_config_dict, mock_gcs_client, monkeypatch):
        """Test successful download to memory."""
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        mock_response = Mock()
        mock_response.headers = {'content-length': '1024'}
        mock_response.raise_for_status = Mock()
        mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2', b'chunk3'])

        monkeypatch.setattr('requests.get', lambda *args, **kwargs: mock_response)

        config = DownloadConfig(**download_config_dict)
        downloader = DataDownloader(config)

        content = downloader.download_to_memory()

        assert content == b'chunk1chunk2chunk3'
        mock_response.raise_for_status.assert_called_once()

    def test_download_chunks_with_progress(self, download_config_dict, mock_gcs_client, monkeypatch):
        """Test chunk download with progress logging."""
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        config = DownloadConfig(**download_config_dict)
        downloader = DataDownloader(config)

        mock_response = Mock()
        mock_response.headers = {'content-length': str(5 * 1024 * 1024)}  # 5MB
        chunk_size = 1024 * 1024  # 1MB
        mock_response.iter_content = Mock(return_value=[b'x' * chunk_size for _ in range(5)])

        content = downloader._download_chunks(mock_response)

        assert len(content) == 5 * 1024 * 1024

    def test_extract_if_compressed_no_compression(self, download_config_dict, mock_gcs_client, monkeypatch):
        """Test extraction skips non-compressed files."""
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        config = DownloadConfig(**download_config_dict)
        downloader = DataDownloader(config)

        content = b'raw csv data'
        result = downloader.extract_if_compressed(content)

        assert result == content

    def test_is_compressed(self):
        """Test _is_compressed method."""
        assert DataDownloader._is_compressed('file.tar.gz')
        assert DataDownloader._is_compressed('file.tgz')
        assert not DataDownloader._is_compressed('file.csv')

    def test_is_valid_file(self):
        """Test _is_valid_file method."""
        assert DataDownloader._is_valid_file('data.csv')
        assert DataDownloader._is_valid_file('data.parquet')
        assert DataDownloader._is_valid_file('data.json')
        assert not DataDownloader._is_valid_file('data.txt')

    def test_get_file_stats_from_bytes_csv(self, download_config_dict, mock_gcs_client, monkeypatch, sample_housing_data):
        """Test getting stats from CSV bytes."""
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        config = DownloadConfig(**download_config_dict)
        downloader = DataDownloader(config)

        csv_buffer = io.StringIO()
        sample_housing_data.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')

        stats = downloader.get_file_stats_from_bytes(csv_bytes)

        assert stats.file_size_mb > 0
        assert stats.n_rows == len(sample_housing_data)
        assert stats.n_columns == len(sample_housing_data.columns)

    def test_upload_to_gcs_success(self, download_config_dict, mock_gcs_client, monkeypatch):
        """Test successful upload to GCS."""
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        config = DownloadConfig(**download_config_dict)
        downloader = DataDownloader(config)

        content = b'test data'
        gcs_uri = downloader.upload_to_gcs(content)

        assert gcs_uri == f"gs://{config.bucket_name}/{config.gcs_output_path}"
        mock_gcs_client['blob'].upload_from_string.assert_called_once_with(content)

    def test_run_success(self, download_config_dict, mock_gcs_client, monkeypatch, sample_housing_data):
        """Test successful full download run."""
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        csv_buffer = io.StringIO()
        sample_housing_data.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')

        mock_response = Mock()
        mock_response.headers = {'content-length': str(len(csv_bytes))}
        mock_response.raise_for_status = Mock()
        mock_response.iter_content = Mock(return_value=[csv_bytes])

        monkeypatch.setattr('requests.get', lambda *args, **kwargs: mock_response)

        config = DownloadConfig(**download_config_dict)
        downloader = DataDownloader(config)

        result = downloader.run()

        assert result.success is True
        assert result.error_message is None

    def test_run_failure(self, download_config_dict, mock_gcs_client, monkeypatch):
        """Test run returns error result on failure."""
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        def mock_get(*args, **kwargs):
            raise requests.RequestException("Download failed")

        monkeypatch.setattr('requests.get', mock_get)

        config = DownloadConfig(**download_config_dict)
        downloader = DataDownloader(config)

        result = downloader.run()

        assert result.success is False
        assert "Download failed" in result.error_message

    def test_extract_tarball_success(self, download_config_dict, mock_gcs_client, monkeypatch, tmp_path):
        """Test _extract_tarball extracts content successfully."""
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        # Create tar.gz file
        csv_data = b'col1,col2\n1,2'
        tar_path = tmp_path / "test.tar.gz"

        with tarfile.open(tar_path, 'w:gz') as tar:
            csv_info = tarfile.TarInfo(name='data.csv')
            csv_info.size = len(csv_data)
            tar.addfile(csv_info, io.BytesIO(csv_data))

        with open(tar_path, 'rb') as f:
            tar_content = f.read()

        config_dict = download_config_dict.copy()
        config_dict['file_url'] = 'https://example.com/data.tar.gz'
        config = DownloadConfig(**config_dict)
        downloader = DataDownloader(config)

        result = downloader._extract_tarball(tar_content)

        assert result == csv_data

    def test_write_temp_file(self):
        """Test _write_temp_file creates temp file."""
        content = b'test data'
        temp_path = DataDownloader._write_temp_file(content)

        import os
        assert os.path.exists(temp_path)
        assert temp_path.endswith('.tgz')

        # Cleanup
        os.remove(temp_path)

    def test_cleanup_temp_file_nonexistent(self):
        """Test _cleanup_temp_file handles non-existent files."""
        # Should not raise error
        DataDownloader._cleanup_temp_file('/nonexistent/file.txt')

    def test_get_csv_stats_error_handling(self, download_config_dict, mock_gcs_client, monkeypatch):
        """Test _get_csv_stats handles pandas exceptions."""
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        config = DownloadConfig(**download_config_dict)
        downloader = DataDownloader(config)

        # Mock pd.read_csv to raise an exception
        import pandas as pd
        original_read_csv = pd.read_csv

        def mock_read_csv(*args, **kwargs):
            raise pd.errors.ParserError("Unable to parse")

        monkeypatch.setattr('pandas.read_csv', mock_read_csv)

        invalid_data = b'some,data\n1,2'
        result = downloader._get_csv_stats(invalid_data)

        assert result is None

        # Restore original
        monkeypatch.setattr('pandas.read_csv', original_read_csv)

    def test_create_blob_metadata(self, download_config_dict, mock_gcs_client, monkeypatch):
        """Test _create_blob_metadata creates correct metadata."""
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        config = DownloadConfig(**download_config_dict)
        downloader = DataDownloader(config)

        content = b'x' * (2 * 1024 * 1024)  # 2MB
        metadata = downloader._create_blob_metadata(content)

        assert 'uploaded_at' in metadata
        assert metadata['component'] == '01_download_data'
        assert metadata['storage_type'] == 'gcs_only'
        assert float(metadata['file_size_mb']) == 2.0

    def test_create_error_result(self, download_config_dict, mock_gcs_client, monkeypatch):
        """Test _create_error_result creates proper error result."""
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        config = DownloadConfig(**download_config_dict)
        downloader = DataDownloader(config)

        error = Exception("Test error message")
        result = downloader._create_error_result(error)

        assert result.success is False
        assert result.error_message == "Test error message"
        assert result.stats.file_size_mb == 0.0

    def test_upload_to_gcs_not_initialized(self, download_config_dict, mock_gcs_client, monkeypatch):
        """Test upload fails when GCS client not initialized."""
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        config = DownloadConfig(**download_config_dict)
        downloader = DataDownloader(config)

        # Force client to None
        downloader.storage_client = None
        downloader.bucket = None

        with pytest.raises(RuntimeError, match="GCS client not initialized"):
            downloader.upload_to_gcs(b'data')

    def test_upload_to_gcs_api_error(self, download_config_dict, mock_gcs_client, monkeypatch):
        """Test upload handles GCS API error."""
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        mock_gcs_client['blob'].upload_from_string.side_effect = gcp_exceptions.GoogleAPIError("Upload failed")

        config = DownloadConfig(**download_config_dict)
        downloader = DataDownloader(config)

        with pytest.raises(RuntimeError, match="Error uploading to GCS"):
            downloader.upload_to_gcs(b'data')

    def test_download_to_memory_with_retries(self, download_config_dict, mock_gcs_client, monkeypatch):
        """Test download retries on failure."""
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise requests.RequestException("Network error")

            mock_response = Mock()
            mock_response.headers = {'content-length': '100'}
            mock_response.raise_for_status = Mock()
            mock_response.iter_content = Mock(return_value=[b'success'])
            return mock_response

        monkeypatch.setattr('requests.get', mock_get)

        config = DownloadConfig(**download_config_dict)
        downloader = DataDownloader(config)

        content = downloader.download_to_memory()

        assert content == b'success'
        assert call_count == 2

    def test_download_max_retries_exceeded(self, download_config_dict, mock_gcs_client, monkeypatch):
        """Test download fails after max retries."""
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        def mock_get(*args, **kwargs):
            raise requests.RequestException("Network error")

        monkeypatch.setattr('requests.get', mock_get)

        config = DownloadConfig(**download_config_dict)
        downloader = DataDownloader(config)

        with pytest.raises(requests.RequestException):
            downloader.download_to_memory()

    def test_create_wandb_metadata(self, download_config_dict, mock_gcs_client, monkeypatch):
        """Test creation of W&B metadata."""
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        config = DownloadConfig(**download_config_dict)
        downloader = DataDownloader(config)

        stats = FileStats(
            file_size_mb=1.5,
            n_rows=1000,
            n_columns=10,
            columns=['col1', 'col2'],
            missing_values={'col1': 5}
        )

        result = DownloadResult(
            gcs_uri=f"gs://{config.bucket_name}/{config.gcs_output_path}",
            stats=stats,
            artifact_name=config.artifact_name,
            success=True
        )

        metadata = downloader.create_wandb_metadata(result)

        assert metadata.original_url == str(config.file_url)
        assert metadata.gcs_uri == result.gcs_uri
        assert metadata.file_size_mb == 1.5
        assert metadata.n_rows == 1000

    def test_read_from_tarball_no_valid_file(self, download_config_dict, mock_gcs_client, monkeypatch, tmp_path):
        """Test _read_from_tarball raises error when no valid file found."""
        monkeypatch.setattr('downloader.storage.Client', lambda: mock_gcs_client['client'])

        # Create tar with invalid file
        tar_path = tmp_path / "test.tar.gz"
        invalid_data = b'invalid text content'

        with tarfile.open(tar_path, 'w:gz') as tar:
            info = tarfile.TarInfo(name='invalid.txt')
            info.size = len(invalid_data)
            tar.addfile(info, io.BytesIO(invalid_data))

        config = DownloadConfig(**download_config_dict)
        downloader = DataDownloader(config)

        with pytest.raises(FileNotFoundError, match="No file found with valid extensions"):
            downloader._read_from_tarball(str(tar_path))

    def test_constants(self):
        """Test that module constants are properly defined."""
        assert '.csv' in VALID_EXTENSIONS
        assert '.parquet' in VALID_EXTENSIONS
        assert '.json' in VALID_EXTENSIONS
        assert '.tar.gz' in COMPRESSED_EXTENSIONS
        assert '.tgz' in COMPRESSED_EXTENSIONS
