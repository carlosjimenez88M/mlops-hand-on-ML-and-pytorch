# Testing Philosophy & Guidelines

**Author:** Carlos Daniel Jiménez
**Date:** 2026-01-13
**Status:** Living Document

---

## Core Philosophy: Coverage ≠ Quality

```
High coverage with bad tests < Low coverage with good tests
```

### The Problem with Coverage-Driven Development

Traditional test suites aim for 80-100% code coverage. This sounds good but often leads to:

1. **Mock-Heavy Tests**
   ```python
   # Bad: High coverage, low value
   def test_download(mock_requests, mock_gcs, mock_pandas):
       result = downloader.download()
       assert mock_requests.get.called_once()  # What did this prove?
   ```

2. **Happy Path Only**
   ```python
   # Bad: Tests the easy case that never breaks
   def test_valid_csv():
       df = parse_csv("perfect,clean,data\n1,2,3")
       assert len(df) == 1  # Real CSVs are never this clean
   ```

3. **False Security**
   - 100% coverage
   - All tests pass
   - Production breaks on corrupted CSV with UTF-16 encoding

### Our Approach: Realistic, Value-Focused Testing

```python
# Good: Real data, real edge case, measurable expectation
def test_large_file_no_oom():
    data = generate_realistic_housing_data(1_000_000)  # REAL data
    start_mem = get_memory_usage()
    result = downloader.download(data)
    end_mem = get_memory_usage()

    assert result.success
    assert end_mem - start_mem < 500MB  # MEASURABLE performance expectation
    assert result.processing_time < 30s  # REAL performance requirement
```

---

## What Makes a Good Test?

### 1. Tests REAL Failure Modes

**Bad (Mock-Based):**
```python
def test_network_error(mock_requests):
    mock_requests.get.side_effect = NetworkError()
    result = downloader.download()
    assert result.failed  # Did this test the retry logic? Who knows.
```

**Good (Realistic):**
```python
def test_network_timeout_with_retry():
    """
    Real scenario: Network times out on first 2 attempts, succeeds on 3rd.
    This ACTUALLY tests that retry logic works.
    """
    call_count = 0
    def mock_get(*args):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise requests.Timeout("Network timeout")
        return real_response_data  # REAL data on success

    result = downloader.download()

    assert result.success  # Retry worked
    assert call_count == 3  # Actually retried twice
    assert result.data.shape == (20000, 10)  # REAL data validation
```

### 2. Uses REAL Data (or Realistic Fixtures)

**Bad:**
```python
df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
```

**Good:**
```python
df = TestDataGenerator.generate_realistic_housing_data(
    n_rows=1000,
    include_missing_values=True,  # Real data has missing values
    include_outliers=True,  # Real data has outliers
    realistic_distributions=True  # Lognormal for prices, not uniform random
)
```

### 3. Has Measurable Performance Expectations

**Bad:**
```python
result = process_data(df)
assert result is not None  # How slow is acceptable?
```

**Good:**
```python
expected_time = estimate_processing_time(file_size_mb=10)
start = time.time()
result = process_data(df)
elapsed = time.time() - start

assert elapsed < expected_time  # Catches performance regressions
assert memory_usage() < max_memory  # Catches memory leaks
```

### 4. Tests Edge Cases That ACTUALLY Happen

**Real edge cases we test:**
- CSVs with inconsistent column counts (common in messy data)
- UTF-16 encoding with BOM (Excel exports on Windows)
- Files larger than available RAM (must stream)
- Network timeouts during download (happens in production)
- Compressed archives with multiple files (real data sources)

**Fake edge cases to avoid:**
- Mocked exceptions that can't actually happen
- Synthetic errors that don't match reality
- Edge cases that are impossible given system constraints

---

## Test Categories

### Unit Tests (Fast, No External Dependencies)

**When to use:** Testing business logic, data transformations, calculations

```python
def test_imputation_strategy():
    """Test imputation logic with REAL missing data patterns."""
    df = TestDataGenerator.generate_data_with_missing_values(
        missing_patterns=['random', 'MCAR', 'MAR', 'MNAR']
    )

    result = ImputationAnalyzer.analyze(df)

    # Test with REAL expectations from data science literature
    assert result.strategy in ['median', 'mean', 'mode', 'knn']
    assert result.confidence > 0.7  # Should be confident in strategy choice
```

**Use mocks for:** External services only (GCS, APIs, databases)

**Don't mock:** Your own code, pandas, numpy, standard library

### Integration Tests (Slower, Uses Real External Resources)

**When to use:** Testing end-to-end workflows with real services

```python
@pytest.mark.integration
def test_real_download_from_github():
    """Download REAL housing dataset from GitHub."""
    result = downloader.download(
        url='https://raw.githubusercontent.com/.../housing.tgz'
    )

    assert result.success
    assert result.rows > 20000  # Real dataset has ~20k rows
    assert 'median_house_value' in result.columns  # Real column
```

**Run with:** `pytest -v --run-integration`

**Why mark separately:** Can fail due to network, slow, requires internet

### Performance Tests (Benchmarks with Measurable Expectations)

**When to use:** Testing performance characteristics

```python
@pytest.mark.performance
@pytest.mark.parametrize("file_size_mb", [1, 10, 50, 100])
def test_processing_performance(file_size_mb):
    """Test processing scales linearly with file size."""
    data = generate_data(size_mb=file_size_mb)
    expected_time = file_size_mb * 0.1  # 100ms per MB

    start = time.time()
    result = process(data)
    elapsed = time.time() - start

    assert elapsed < expected_time * 1.5  # 50% margin
    print(f"✅ {file_size_mb}MB: {elapsed:.2f}s (max: {expected_time:.2f}s)")
```

**Run with:** `pytest -v -m performance`

**Why separate:** Slow, can be flaky on CI, hardware-dependent

### Slow Tests (Heavy Operations)

**When to use:** Large file tests, extensive permutations

```python
@pytest.mark.slow
def test_1gb_file_processing():
    """Test processing 1GB file without OOM."""
    data = generate_data(size_mb=1000)  # 1GB

    result = process(data, streaming=True)

    assert result.success
    assert memory_used() < 500MB  # Should stream, not load all
```

**Run with:** `pytest -v -m "not slow"` (exclude from quick tests)

---

## Running Tests

### Quick Test Run (Fast Tests Only)

```bash
# Run fast unit tests (exclude slow/integration)
pytest -v -m "not slow and not integration"

# Expected: ~10-30 seconds
```

### Full Test Suite

```bash
# Run all tests including slow tests
pytest -v

# Expected: ~1-5 minutes
```

### With Integration Tests

```bash
# Run everything including real downloads
pytest -v --run-integration

# Expected: ~2-10 minutes
# Requires: Internet connection
```

### Performance Benchmarks

```bash
# Run performance tests and print timing info
pytest -v -m performance -s

# Expected: ~2-5 minutes
# Note: Results vary by hardware
```

### Coverage Report (with context)

```bash
# Generate coverage report
pytest --cov=src --cov-report=html --cov-report=term

# View report
open htmlcov/index.html
```

**Interpreting coverage:**
- 60-70%: Good if tests are realistic and valuable
- 80-90%: Good only if not achieved with mocks
- 100%: Suspicious - likely testing trivial code or using mocks

**Coverage red flags:**
- High coverage but all tests use mocks
- High coverage but no edge cases tested
- High coverage but no performance tests
- High coverage achieved by testing error messages

---

## Test Structure Guidelines

### File Organization

```
tests/
├── fixtures/
│   ├── __init__.py
│   └── test_data_generator.py      # Realistic test data
├── test_downloader_realistic.py     # Realistic downloader tests
├── test_preprocessor_realistic.py   # Real edge cases
├── conftest.py                      # Shared fixtures
└── README_TESTING_PHILOSOPHY.md     # This file
```

### Test File Template

```python
"""
Realistic Tests for [Component]
Author: [Your Name]
Date: [Date]

Philosophy:
- Tests use REAL data
- Edge cases are REALISTIC
- Performance is MEASURABLE
- Minimal mocking (external services only)

What these tests verify:
1. [Real scenario 1]
2. [Real scenario 2]
3. [Performance expectations]

What we DON'T test (and why):
- [Thing]: [Reason - e.g., "External service, mocked"]
- [Thing]: [Reason - e.g., "Trivial getter, no logic"]
"""
import pytest
from test_data_generator import TestDataGenerator

class Test[Component]Realistic:
    """Realistic tests for [Component]."""

    # Setup
    @pytest.fixture
    def realistic_data(self):
        return TestDataGenerator.generate_[type]_data()

    # Edge cases
    def test_[realistic_edge_case](self):
        """Why this matters: [Real production scenario]."""
        pass

    # Performance
    @pytest.mark.performance
    def test_performance_[operation](self):
        """Measurable performance expectations."""
        pass
```

---

## Common Anti-Patterns to Avoid

### ❌ Mock Everything

```python
# Bad: What did this test?
def test_download(mock_requests, mock_gcs, mock_pandas, mock_time):
    mock_requests.get.return_value = Mock()
    downloader.download()
    assert mock_requests.get.called
```

### ❌ Test Implementation Details

```python
# Bad: Breaks when you refactor
def test_internal_method():
    downloader._internal_helper()  # Don't test private methods directly
```

### ❌ No Measurable Expectations

```python
# Bad: How fast should it be?
def test_performance():
    result = process()
    assert result is not None  # No performance assertion!
```

### ❌ Fake Edge Cases

```python
# Bad: This can't happen
def test_negative_file_size():
    result = process(size=-100)  # File sizes can't be negative
```

### ✅ Instead: Test Public Interface with Real Data

```python
# Good: Tests public API with realistic data and expectations
def test_process_realistic_data():
    data = TestDataGenerator.generate_realistic_housing_data(10000)
    expected_time = estimate_processing_time(data.shape[0])

    start = time.time()
    result = process(data)
    elapsed = time.time() - start

    # Real assertions
    assert result.success
    assert result.rows_processed == 10000
    assert result.missing_values_handled > 0  # Real data has missing values
    assert elapsed < expected_time  # Performance expectation
```

---

## Metrics That Matter

### Instead of Coverage Percentage, Track:

1. **Bug Detection Rate**
   - How many production bugs did tests catch in dev?
   - Target: > 80% of bugs caught before production

2. **Test Reliability**
   - How many flaky tests (fail randomly)?
   - Target: < 1% flaky rate

3. **Test Speed**
   - How long for fast test suite?
   - Target: < 30 seconds for unit tests

4. **Realistic Edge Cases**
   - How many tests use real production scenarios?
   - Target: > 50% of tests use realistic data/scenarios

5. **Performance Regression Detection**
   - Do tests catch performance regressions?
   - Target: Yes, with measurable assertions

---

## When Coverage IS Useful

Coverage is a good **negative indicator**:

✅ **0% coverage** → Code is definitely untested → Add tests
✅ **Low coverage on critical path** → Need more tests
❌ **100% coverage** → Doesn't mean code is well-tested

### How to Use Coverage

1. **Find untested code paths**
   ```bash
   pytest --cov=src --cov-report=term-missing
   # Look for critical functions at 0%
   ```

2. **Prioritize critical paths**
   - Data processing: High priority
   - Logging: Low priority
   - Error messages: Low priority

3. **Don't game the metric**
   ```python
   # Bad: Adds coverage but no value
   def test_trivial_getter():
       config = Config()
       assert config.name == config.name  # Useless test
   ```

---

## Real-World Example: Housing Dataset

### Bad Test (High Coverage, Low Value)

```python
def test_download(mock_everything):
    """Test download works."""
    result = downloader.download()
    assert result is not None
    # Coverage: 95%
    # Bugs found: 0
```

### Good Test (Lower Coverage, High Value)

```python
def test_download_handles_utf16_with_bom():
    """
    Real scenario: Excel exports housing data as UTF-16 with BOM.
    This broke production last month.
    """
    utf16_csv = TestDataGenerator.generate_csv_with_encoding('utf-16', add_bom=True)

    result = downloader.process_csv(utf16_csv)

    assert result.success, "Should handle UTF-16 with BOM"
    assert result.rows > 0, "Should parse data correctly"
    assert 'median_house_value' in result.columns

    # Coverage: 60%
    # Bugs found: Would have caught last month's production bug
```

---

## Conclusion

**Remember:**
- Coverage is a tool, not a goal
- Mock external services, not your own code
- Test real edge cases that happen in production
- Have measurable performance expectations
- Value > Coverage percentage

**When in doubt:**
- Would this test have caught a real production bug?
- Am I testing mocks or real code?
- Will this test break when I refactor?
- Does this test have a measurable expectation?

If yes to first two, no to last two → Good test ✅

---

**Last Updated:** 2026-01-13
**Status:** Living document - update as we learn
