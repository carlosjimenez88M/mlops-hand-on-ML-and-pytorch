"""
Simple Integration Tests - Verify Pipeline Actually Works
No mocks, no BS, just "does it run?"
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.mark.integration
def test_main_pipeline_help_works():
    """Test: Can we run main.py --help?"""
    result = subprocess.run(
        ["python", "main.py", "--help"], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0
    assert "Hydra" in result.stdout


@pytest.mark.integration
@pytest.mark.slow
def test_check_environment_validation():
    """Test: Does environment validation catch missing vars?"""
    # This should fail if WANDB_API_KEY is not set properly
    result = subprocess.run(
        ["python", "main.py"],
        capture_output=True,
        text=True,
        timeout=30,
        env={"PATH": subprocess.os.environ["PATH"]},  # Clean env
    )

    # Should fail with clear error about missing vars
    assert "CONFIGURATION ERROR" in result.stdout or result.returncode != 0


def test_docker_compose_file_valid():
    """Test: Is docker-compose.yaml valid?"""
    result = subprocess.run(
        ["docker", "compose", "config"], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, f"docker-compose.yaml invalid: {result.stderr}"


def test_model_file_exists():
    """Test: Does trained model exist?"""
    model_path = Path("models/trained/housing_price_model.pkl")
    assert model_path.exists(), f"Model not found at {model_path}"
    assert model_path.stat().st_size > 1_000_000, "Model file too small"


def test_best_params_yaml_exists():
    """Test: Does sweep output exist?"""
    params_path = Path("src/model/06_sweep/best_params.yaml")
    assert params_path.exists(), f"best_params.yaml not found at {params_path}"


def test_api_dockerfile_exists():
    """Test: Does API Dockerfile exist and is valid?"""
    dockerfile = Path("api/Dockerfile")
    assert dockerfile.exists()

    content = dockerfile.read_text()
    assert "FROM python" in content
    assert "CMD" in content or "ENTRYPOINT" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
