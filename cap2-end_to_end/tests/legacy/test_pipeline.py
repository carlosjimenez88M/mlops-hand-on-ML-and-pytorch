#!/usr/bin/env python3
"""
Pipeline Configuration Test
Author: Carlos Daniel Jiménez
Date: 2025-01-13

Quick test to verify pipeline configuration without executing anything.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf
import yaml


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_env_file():
    """Check if .env file exists and is properly configured."""
    print_section("CHECKING ENVIRONMENT FILE")

    if not Path(".env").exists():
        print(" .env file not found")
        print("   Run: cp .env.example .env")
        return False

    load_dotenv()

    required_vars = ["GCS_BUCKET_NAME", "GCP_PROJECT_ID"]
    missing = []

    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f" {var}: {value}")
        else:
            print(f" {var}: Not set")
            missing.append(var)

    if missing:
        print(f"\n  Missing variables: {', '.join(missing)}")
        return False

    return True


def check_config_file():
    """Check if config.yaml exists and is valid."""
    print_section("CHECKING CONFIGURATION FILE")

    if not Path("config.yaml").exists():
        print(" config.yaml not found")
        return False

    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        print(" config.yaml loaded successfully")

        # Check main configuration
        if "main" in config:
            print(f"\n  Project: {config['main']['project_name']}")
            print(f"  Experiment: {config['main']['experiment_name']}")
            print(f"  Steps: {len(config['main']['execute_steps'])}")

        # List steps
        print("\n  Configured steps:")
        for i, step in enumerate(config['main']['execute_steps'], 1):
            print(f"    {i}. {step}")

        return True

    except Exception as e:
        print(f" Error loading config.yaml: {e}")
        return False


def check_mlproject_files():
    """Check if MLproject files exist for each step."""
    print_section("CHECKING MLPROJECT FILES")

    steps = {
        "01_download_data": "src/data/01_download_data/MLproject",
        "02_preprocessing_and_imputation": "src/data/02_preprocessing_and_imputation/MLproject",
        "03_feature_engineering": "src/data/03_feature_engineering/MLproject",
        "04_segregation": "src/data/04_segregation/MLproject",
        "05_model_selection": "src/model/05_model_selection/MLproject",
    }

    all_exist = True
    for step_name, mlproject_path in steps.items():
        path = Path(mlproject_path)
        if path.exists():
            print(f" {step_name}: {mlproject_path}")
        else:
            print(f" {step_name}: {mlproject_path} not found")
            all_exist = False

    return all_exist


def check_main_files():
    """Check if main.py files exist for each step."""
    print_section("CHECKING MAIN.PY FILES")

    main_files = [
        "src/data/01_download_data/main.py",
        "src/data/02_preprocessing_and_imputation/main.py",
        "src/data/03_feature_engineering/main.py",
        "src/data/04_segregation/main.py",
        "src/model/05_model_selection/main.py",
    ]

    all_exist = True
    for main_file in main_files:
        path = Path(main_file)
        if path.exists():
            print(f" {main_file}")
        else:
            print(f" {main_file} not found")
            all_exist = False

    return all_exist


def check_dependencies():
    """Check if required Python packages are installed."""
    print_section("CHECKING DEPENDENCIES")

    required_packages = [
        "mlflow",
        "hydra",
        "wandb",
        "google.cloud.storage",
        "pydantic",
        "dotenv",
        "omegaconf",
    ]

    missing = []
    for package in required_packages:
        package_name = package.split('.')[0]  # Handle submodules
        try:
            __import__(package_name)
            print(f" {package}")
        except ImportError:
            print(f" {package} not installed")
            missing.append(package)

    if missing:
        print(f"\n  Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        return False

    return True


def main():
    """Run all checks."""
    print("\n" + "=" * 70)
    print("  PIPELINE CONFIGURATION TEST")
    print("=" * 70)

    checks = [
        ("Environment file", check_env_file),
        ("Configuration file", check_config_file),
        ("MLproject files", check_mlproject_files),
        ("Main.py files", check_main_files),
        ("Dependencies", check_dependencies),
    ]

    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))

    # Summary
    print_section("SUMMARY")

    all_passed = all(result for _, result in results)

    for name, result in results:
        status = " PASSED" if result else " FAILED"
        print(f"  {status}: {name}")

    print("\n" + "=" * 70)

    if all_passed:
        print("   ALL CHECKS PASSED!")
        print("   Ready to run: python main.py")
    else:
        print("   SOME CHECKS FAILED")
        print("    Fix the issues above before running the pipeline")

    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
