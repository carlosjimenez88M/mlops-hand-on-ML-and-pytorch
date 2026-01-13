# Unit Tests

This directory contains comprehensive unit tests for the MLOps pipeline components.

## Test Coverage

The test suite includes tests for:

1. **DataDownloader** (`test_downloader.py`)
   - GCS client initialization
   - Data download with retries
   - File extraction from compressed archives
   - File statistics computation
   - Upload to GCS
   - Error handling

2. **DataPreprocessor** (`test_preprocessor.py`)
   - GCS operations (download/upload)
   - Missing value handling strategies (drop, median, mean, mode, auto)
   - Feature engineering
   - Automatic imputation method selection
   - Error handling

3. **ImputationAnalyzer** (`test_imputation_analyzer.py`)
   - Missing values analysis
   - Correlation matrix computation
   - Simple imputation methods (median, mean)
   - KNN imputation
   - Iterative imputation with Random Forest
   - Method comparison and automatic selection
   - Visualization generation

## Running Tests

### Install Dependencies

First, ensure pytest and testing dependencies are installed:

```bash
pip install pytest pytest-cov pytest-mock
```

### Run All Tests

```bash
# From project root
cd /Users/carlosdaniel/Documents/Projects/Personal_Projects/mlops-hand-on-ML-and-pytorch/cap2-end_to_end

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_downloader.py -v
pytest tests/test_preprocessor.py -v
pytest tests/test_imputation_analyzer.py -v
```

### Run Tests by Pattern

```bash
# Run tests matching a pattern
pytest tests/ -k "test_download"

# Run tests for a specific class
pytest tests/test_downloader.py::TestDataDownloader

# Run a specific test
pytest tests/test_downloader.py::TestDataDownloader::test_init_success
```

## Test Structure

All tests follow these principles:

- **Isolation**: Each test is independent and doesn't rely on others
- **Mocking**: External dependencies (GCS, requests, MLflow) are mocked
- **Fixtures**: Common setup is shared via pytest fixtures in `conftest.py`
- **Coverage**: Tests cover both success and failure scenarios
- **Assertions**: Clear and specific assertions for expected behavior

## Fixtures

Common fixtures are defined in `conftest.py`:

- `sample_housing_data`: Sample DataFrame for testing
- `sample_housing_data_csv`: CSV file with sample data
- `mock_gcs_client`: Mocked Google Cloud Storage client
- `mock_requests_response`: Mocked HTTP response
- `download_config_dict`: Sample downloader configuration
- `preprocessing_config_dict`: Sample preprocessor configuration
- `mock_mlflow`: Mocked MLflow functions

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest tests/ --cov=src --cov-report=xml
```

## Author

Carlos Daniel Jiménez
