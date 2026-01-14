"""
Realistic Test Data Generator
Purpose: Generate realistic, corrupted, and edge-case test data
Author: Carlos Daniel Jiménez
Date: 2026-01-13

Philosophy:
- Tests should use REAL data, not mocks
- Edge cases should be REALISTIC edge cases
- Performance tests should have REAL performance assertions
"""
import io
import tarfile
import gzip
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np


class TestDataGenerator:
    """
    Generate realistic test data for housing price prediction pipeline.

    This class creates various types of test data including:
    - Valid, realistic housing data
    - Corrupted/malformed CSVs
    - Different encodings (UTF-8, UTF-16, Latin-1, with BOM)
    - Large files that could cause memory issues
    - Compressed files (tar.gz, gzip)
    """

    @staticmethod
    def generate_realistic_housing_data(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
        """
        Generate realistic California housing data.

        Based on actual California housing dataset characteristics:
        - Longitude: -124 to -114
        - Latitude: 32 to 42
        - Housing median age: 1 to 52
        - Total rooms: 2 to 39320
        - Total bedrooms: 1 to 6445
        - Population: 3 to 35682
        - Households: 1 to 6082
        - Median income: 0.5 to 15
        - Ocean proximity: categorical
        - Median house value: 15000 to 500001 (target)

        Args:
            n_rows: Number of rows to generate
            seed: Random seed for reproducibility

        Returns:
            DataFrame with realistic housing data
        """
        np.random.seed(seed)

        # Generate realistic distributions
        data = {
            'longitude': np.random.uniform(-124, -114, n_rows),
            'latitude': np.random.uniform(32, 42, n_rows),
            'housing_median_age': np.random.randint(1, 53, n_rows),
            'total_rooms': np.random.lognormal(6.5, 0.8, n_rows).astype(int),
            'total_bedrooms': np.random.lognormal(5.5, 0.7, n_rows).astype(int),
            'population': np.random.lognormal(6.0, 0.9, n_rows).astype(int),
            'households': np.random.lognormal(5.3, 0.7, n_rows).astype(int),
            'median_income': np.random.gamma(3, 2, n_rows),
            'ocean_proximity': np.random.choice(
                ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
                n_rows,
                p=[0.4, 0.35, 0.002, 0.15, 0.098]  # Realistic distribution
            ),
            'median_house_value': np.random.lognormal(12.5, 0.5, n_rows)
        }

        df = pd.DataFrame(data)

        # Add realistic missing values (total_bedrooms has ~0.5% missing in real data)
        missing_indices = np.random.choice(n_rows, size=int(n_rows * 0.005), replace=False)
        df.loc[missing_indices, 'total_bedrooms'] = np.nan

        # Clip to realistic ranges
        df['median_income'] = df['median_income'].clip(0.5, 15)
        df['median_house_value'] = df['median_house_value'].clip(15000, 500001)

        # Ensure logical constraints (bedrooms < rooms, households < population)
        df['total_bedrooms'] = df[['total_bedrooms', 'total_rooms']].min(axis=1) * 0.2
        df['households'] = df[['households', 'population']].min(axis=1) * 0.3

        return df

    @staticmethod
    def generate_corrupted_csv(corruption_type: str = 'missing_columns') -> bytes:
        """
        Generate CSV data with specific types of corruption.

        Args:
            corruption_type: Type of corruption to introduce
                - 'missing_columns': Some rows have fewer columns
                - 'extra_columns': Some rows have extra columns
                - 'invalid_numbers': Non-numeric values in numeric columns
                - 'wrong_delimiter': Uses semicolon instead of comma
                - 'missing_header': No header row
                - 'duplicate_columns': Duplicate column names
                - 'empty_lines': Random empty lines
                - 'malformed_quotes': Unbalanced quotes

        Returns:
            Corrupted CSV as bytes
        """
        if corruption_type == 'missing_columns':
            csv_data = (
                "longitude,latitude,housing_median_age,total_rooms,total_bedrooms\n"
                "-122.23,37.88,41,880,129\n"
                "-122.22,37.86,21\n"  # Missing 2 columns
                "-122.24,37.85,52,7099,1106\n"
            )
        elif corruption_type == 'extra_columns':
            csv_data = (
                "longitude,latitude,housing_median_age\n"
                "-122.23,37.88,41\n"
                "-122.22,37.86,21,999,extra,data\n"  # Extra columns
                "-122.24,37.85,52\n"
            )
        elif corruption_type == 'invalid_numbers':
            csv_data = (
                "longitude,latitude,housing_median_age,total_rooms\n"
                "-122.23,37.88,41,880\n"
                "-122.22,INVALID,21,765\n"  # Invalid number
                "-122.24,37.85,fifty-two,7099\n"  # Text instead of number
            )
        elif corruption_type == 'wrong_delimiter':
            csv_data = (
                "longitude;latitude;housing_median_age\n"
                "-122.23;37.88;41\n"
                "-122.22;37.86;21\n"
            )
        elif corruption_type == 'missing_header':
            csv_data = (
                "-122.23,37.88,41,880,129\n"
                "-122.22,37.86,21,765,235\n"
            )
        elif corruption_type == 'duplicate_columns':
            csv_data = (
                "longitude,latitude,longitude,housing_median_age\n"  # Duplicate 'longitude'
                "-122.23,37.88,-122.23,41\n"
                "-122.22,37.86,-122.22,21\n"
            )
        elif corruption_type == 'empty_lines':
            csv_data = (
                "longitude,latitude,housing_median_age\n"
                "-122.23,37.88,41\n"
                "\n"  # Empty line
                "\n"  # Empty line
                "-122.22,37.86,21\n"
                "\n"
            )
        elif corruption_type == 'malformed_quotes':
            csv_data = (
                'longitude,latitude,ocean_proximity\n'
                '-122.23,37.88,"NEAR BAY\n'  # Unbalanced quote
                '-122.22,37.86,"INLAND\n'
            )
        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")

        return csv_data.encode('utf-8')

    @staticmethod
    def generate_csv_with_encoding(encoding: str = 'utf-8', add_bom: bool = False) -> bytes:
        """
        Generate CSV with specific encoding.

        Args:
            encoding: Encoding to use ('utf-8', 'utf-16', 'latin-1', 'iso-8859-1')
            add_bom: Whether to add BOM (Byte Order Mark)

        Returns:
            CSV data encoded with specified encoding
        """
        # For latin-1, use only latin-1 compatible characters
        if encoding.lower() in ['latin-1', 'iso-8859-1']:
            csv_data = (
                "longitude,latitude,city_name,notes\n"
                "-122.23,37.88,Sao Paulo,Cafe resume naive\n"
                "-122.22,37.86,Munchen,Grosse Strasse\n"
                "-122.24,37.85,Paris,Accent aigu\n"
                "-122.25,37.84,Madrid,Espanol texto\n"
            )
        else:
            # For UTF-8 and UTF-16, use international characters
            csv_data = (
                "longitude,latitude,city_name,notes\n"
                "-122.23,37.88,São Paulo,Café résumé naïve\n"
                "-122.22,37.86,München,Größe Straße\n"
                "-122.24,37.85,北京,中文测试\n"
                "-122.25,37.84,Москва,Русский текст\n"
            )

        encoded = csv_data.encode(encoding)

        if add_bom:
            if encoding.lower() == 'utf-8':
                encoded = b'\xef\xbb\xbf' + encoded
            elif encoding.lower() == 'utf-16':
                # UTF-16 with BOM (LE)
                encoded = b'\xff\xfe' + csv_data.encode('utf-16-le')

        return encoded

    @staticmethod
    def generate_large_csv(size_mb: int = 10, seed: int = 42) -> bytes:
        """
        Generate a large CSV file to test memory handling.

        Args:
            size_mb: Target size in megabytes
            seed: Random seed

        Returns:
            Large CSV as bytes
        """
        # Estimate rows needed (assume ~100 bytes per row)
        rows_needed = (size_mb * 1024 * 1024) // 100

        df = TestDataGenerator.generate_realistic_housing_data(rows_needed, seed)

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue().encode('utf-8')

    @staticmethod
    def create_tar_gz(
        files: Dict[str, bytes],
        output_path: Path = None
    ) -> Tuple[bytes, Path]:
        """
        Create a tar.gz archive with multiple files.

        Args:
            files: Dictionary mapping filename -> content (bytes)
            output_path: Optional path to save the tar.gz file

        Returns:
            Tuple of (tar.gz content as bytes, path if saved)
        """
        buffer = io.BytesIO()

        with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
            for filename, content in files.items():
                # Create TarInfo
                info = tarfile.TarInfo(name=filename)
                info.size = len(content)

                # Add file to archive
                tar.addfile(info, io.BytesIO(content))

        tar_content = buffer.getvalue()

        # Save to disk if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(tar_content)

        return tar_content, output_path

    @staticmethod
    def create_gzip(content: bytes, output_path: Path = None) -> Tuple[bytes, Path]:
        """
        Create a gzip compressed file.

        Args:
            content: Content to compress
            output_path: Optional path to save

        Returns:
            Tuple of (gzipped content, path if saved)
        """
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode='wb') as gz:
            gz.write(content)

        gzipped = buffer.getvalue()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(gzipped)

        return gzipped, output_path


class PerformanceTestData:
    """Performance test utilities with measurable expectations."""

    @staticmethod
    def estimate_processing_time(file_size_mb: float, operation: str = 'parse_csv') -> float:
        """
        Estimate expected processing time based on file size.

        Args:
            file_size_mb: File size in megabytes
            operation: Type of operation

        Returns:
            Maximum acceptable time in seconds
        """
        # Benchmarks (adjust based on actual hardware)
        benchmarks = {
            'parse_csv': 0.5,  # seconds per MB (pandas is fast)
            'compress': 0.2,   # seconds per MB
            'decompress': 0.1, # seconds per MB
            'upload': 2.0,     # seconds per MB (network dependent)
            'download': 1.0,   # seconds per MB
            'preprocess': 0.3, # seconds per MB
            'impute': 0.4,     # seconds per MB
            'feature_engineering': 0.6,  # seconds per MB
        }

        base_time = benchmarks.get(operation, 1.0)

        # Add overhead and margin
        overhead = 1.0  # 1 second base overhead
        margin = 1.5    # 50% margin for variance

        return (file_size_mb * base_time + overhead) * margin


class PreprocessingTestData:
    """Test data specifically for preprocessing and imputation."""

    @staticmethod
    def generate_data_with_missing_patterns(
        n_rows: int = 1000,
        missing_rate: float = 0.1,
        pattern: str = 'MCAR',
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate data with realistic missing value patterns.

        Args:
            n_rows: Number of rows
            missing_rate: Proportion of missing values
            pattern: Missing pattern type
                - 'MCAR': Missing Completely At Random
                - 'MAR': Missing At Random (depends on other variables)
                - 'MNAR': Missing Not At Random (depends on missing value itself)
            seed: Random seed

        Returns:
            DataFrame with realistic missing patterns
        """
        np.random.seed(seed)

        # Generate base data
        base_df = TestDataGenerator.generate_realistic_housing_data(n_rows, seed)

        if pattern == 'MCAR':
            # Completely random - 10% missing in total_bedrooms (like real data)
            mask = np.random.random(n_rows) < missing_rate
            base_df.loc[mask, 'total_bedrooms'] = np.nan

        elif pattern == 'MAR':
            # Missing depends on another variable (older houses more likely to have missing data)
            # This is realistic - older housing records have incomplete data
            older_houses = base_df['housing_median_age'] > 40
            mask = older_houses & (np.random.random(n_rows) < missing_rate * 2)
            base_df.loc[mask, 'total_bedrooms'] = np.nan

        elif pattern == 'MNAR':
            # Missing depends on the value itself (very high bedroom counts not recorded)
            # Realistic - outliers sometimes not recorded properly
            high_bedrooms = base_df['total_bedrooms'] > base_df['total_bedrooms'].quantile(0.95)
            mask = high_bedrooms & (np.random.random(n_rows) < missing_rate * 3)
            base_df.loc[mask, 'total_bedrooms'] = np.nan

        return base_df

    @staticmethod
    def generate_data_with_outliers(
        n_rows: int = 1000,
        outlier_rate: float = 0.05,
        outlier_type: str = 'extreme',
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate data with realistic outliers.

        Args:
            n_rows: Number of rows
            outlier_rate: Proportion of outliers
            outlier_type: Type of outliers
                - 'extreme': Very extreme values (5+ std dev)
                - 'moderate': Moderately extreme (3-5 std dev)
                - 'data_entry': Realistic data entry errors (e.g., 123 years old)
            seed: Random seed

        Returns:
            DataFrame with outliers
        """
        np.random.seed(seed)

        df = TestDataGenerator.generate_realistic_housing_data(n_rows, seed)
        n_outliers = int(n_rows * outlier_rate)

        if outlier_type == 'extreme':
            # Extremely high house values (outliers in expensive areas)
            outlier_indices = np.random.choice(n_rows, n_outliers, replace=False)
            df.loc[outlier_indices, 'median_house_value'] *= np.random.uniform(5, 10, n_outliers)

        elif outlier_type == 'moderate':
            # Moderately high values (still outliers but more realistic)
            outlier_indices = np.random.choice(n_rows, n_outliers, replace=False)
            df.loc[outlier_indices, 'median_house_value'] *= np.random.uniform(2, 4, n_outliers)

        elif outlier_type == 'data_entry':
            # Data entry errors (realistic mistakes)
            outlier_indices = np.random.choice(n_rows, n_outliers, replace=False)
            # Housing age can't be > 100, but data entry might record it
            df.loc[outlier_indices, 'housing_median_age'] *= 10
            # Population can't be negative, but errors happen
            df.loc[outlier_indices[:n_outliers//2], 'population'] *= -1

        return df
