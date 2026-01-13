"""
MLOps Pipeline Orchestrator
Author: Carlos Daniel Jiménez
Date: 2025-01-13

This script orchestrates the complete ML pipeline using MLflow and Hydra.
It executes each step sequentially based on the configuration.
"""

import os
import sys
from pathlib import Path
from typing import List

import hydra
import mlflow
import wandb
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def validate_environment_variables() -> None:
    """
    Validate that required environment variables are set before pipeline execution.
    Raises SystemExit if critical variables are missing.
    """
    required_vars = {
        "GCP_PROJECT_ID": "Google Cloud Project ID",
        "GCS_BUCKET_NAME": "Google Cloud Storage Bucket Name",
        "WANDB_API_KEY": "Weights & Biases API Key",
        "WANDB_PROJECT": "Weights & Biases Project Name",
    }

    missing_vars = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value or value == "your-wandb-api-key-here" or value == "":
            missing_vars.append(f"  - {var}: {description}")

    if missing_vars:
        print("\n" + "=" * 70)
        print("  CONFIGURATION ERROR")
        print("=" * 70)
        print("\nThe following required environment variables are missing or invalid:\n")
        print("\n".join(missing_vars))
        print("\nPlease set these variables in your .env file or environment.")
        print("\nExample .env file:")
        print("-" * 70)
        print("GCP_PROJECT_ID=your-project-id")
        print("GCS_BUCKET_NAME=your-bucket-name")
        print("WANDB_API_KEY=your-actual-wandb-key")
        print("WANDB_PROJECT=your-project-name")
        print("-" * 70)
        print("\nNote: You can find your W&B API key at https://wandb.ai/settings")
        print("=" * 70 + "\n")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  ENVIRONMENT VALIDATION PASSED")
    print("=" * 70)
    print("All required environment variables are set.")
    print("=" * 70 + "\n")


def setup_environment(config: DictConfig) -> None:
    """
    Setup environment variables for the pipeline execution.

    Args:
        config: Hydra configuration
    """
    # Set W&B environment variables
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Initialize wandb config
    wandb.config = OmegaConf.to_container(
        config,
        resolve=True,
        throw_on_missing=True
    )

    print("\n" + "=" * 70)
    print("  MLOPS PIPELINE ORCHESTRATOR")
    print("=" * 70)
    print(f"  Project: {config['main']['project_name']}")
    print(f"  Experiment: {config['main']['experiment_name']}")
    print(f"  GCS Bucket: {config['gcs']['bucket_name']}")
    print("=" * 70 + "\n")


def get_steps_to_execute(config: DictConfig) -> List[str]:
    """
    Parse and return the list of steps to execute.

    Args:
        config: Hydra configuration

    Returns:
        List of step names to execute
    """
    execute_steps = config['main']['execute_steps']

    if isinstance(execute_steps, str):
        steps = [s.strip() for s in execute_steps.split(',')]
    else:
        steps = list(execute_steps)

    print(f"Steps to execute: {', '.join(steps)}\n")
    return steps


def run_download_data(config: DictConfig, root_path: Path) -> None:
    """
    Execute the data download step.

    Args:
        config: Hydra configuration
        root_path: Project root path
    """
    print("\n" + "=" * 70)
    print("  STEP 1: DOWNLOAD DATA")
    print("=" * 70)

    step_path = root_path / "src" / "data" / "01_download_data"

    mlflow.run(
        uri=str(step_path),
        entry_point="main",
        env_manager="local",
        parameters={
            "file_url": config["download_data"]["file_url"],
            "artifact_name": config["download_data"]["artifact_name"],
            "artifact_type": config["download_data"]["artifact_type"],
            "artifact_description": config["download_data"]["artifact_description"],
            "gcs_output_path": config["download_data"]["gcs_output_path"],
            "bucket_name": config["gcs"]["bucket_name"],
            "wandb_project": config["main"]["project_name"],
        },
    )

    print("\nData download completed successfully!\n")


def run_preprocessing_and_imputation(config: DictConfig, root_path: Path) -> None:
    """
    Execute the preprocessing and imputation step.

    Args:
        config: Hydra configuration
        root_path: Project root path
    """
    print("\n" + "=" * 70)
    print("  STEP 2: PREPROCESSING AND IMPUTATION")
    print("=" * 70)

    step_path = root_path / "src" / "data" / "02_preprocessing_and_imputation"

    mlflow.run(
        uri=str(step_path),
        entry_point="main",
        env_manager="local",
        parameters={
            "input_artifact_name": config["preprocessing"]["input_artifact_name"],
            "gcs_input_path": config["preprocessing"]["gcs_input_path"],
            "gcs_output_path": config["preprocessing"]["gcs_output_path"],
            "artifact_name": config["preprocessing"]["artifact_name"],
            "artifact_type": config["preprocessing"]["artifact_type"],
            "artifact_description": config["preprocessing"]["artifact_description"],
            "bucket_name": config["gcs"]["bucket_name"],
            "wandb_project": config["main"]["project_name"],
            "imputation_strategy": config["preprocessing"]["imputation_strategy"],
            "create_features": config["preprocessing"]["create_features"],
        },
    )

    print("\nPreprocessing and imputation completed successfully!\n")


def run_feature_engineering(config: DictConfig, root_path: Path) -> None:
    """
    Execute the feature engineering step.

    Args:
        config: Hydra configuration
        root_path: Project root path
    """
    print("\n" + "=" * 70)
    print("  STEP 3: FEATURE ENGINEERING")
    print("=" * 70)

    step_path = root_path / "src" / "data" / "03_feature_engineering"

    mlflow.run(
        uri=str(step_path),
        entry_point="main",
        env_manager="local",
        parameters={
            "input_artifact_name": config["feature_engineering"]["input_artifact_name"],
            "gcs_input_path": config["feature_engineering"]["gcs_input_path"],
            "gcs_output_path": config["feature_engineering"]["gcs_output_path"],
            "artifact_name": config["feature_engineering"]["artifact_name"],
            "artifact_type": config["feature_engineering"]["artifact_type"],
            "artifact_description": config["feature_engineering"]["artifact_description"],
            "bucket_name": config["gcs"]["bucket_name"],
            "wandb_project": config["main"]["project_name"],
            "n_clusters": config["feature_engineering"]["n_clusters"],
            "gamma": config["feature_engineering"]["gamma"],
            "random_state": config["feature_engineering"]["random_state"],
            "optimize_hyperparams": config["feature_engineering"]["optimize_hyperparams"],
        },
    )

    print("\nFeature engineering completed successfully!\n")


def run_segregation(config: DictConfig, root_path: Path) -> None:
    """
    Execute the data segregation step.

    Args:
        config: Hydra configuration
        root_path: Project root path
    """
    print("\n" + "=" * 70)
    print("  STEP 4: DATA SEGREGATION")
    print("=" * 70)

    step_path = root_path / "src" / "data" / "04_segregation"

    mlflow.run(
        uri=str(step_path),
        entry_point="main",
        env_manager="local",
        parameters={
            "input_artifact_name": config["segregation"]["input_artifact_name"],
            "gcs_input_path": config["segregation"]["gcs_input_path"],
            "gcs_train_output_path": config["segregation"]["gcs_train_output_path"],
            "gcs_test_output_path": config["segregation"]["gcs_test_output_path"],
            "artifact_root": config["segregation"]["artifact_root"],
            "artifact_type": config["segregation"]["artifact_type"],
            "artifact_description": config["segregation"]["artifact_description"],
            "bucket_name": config["gcs"]["bucket_name"],
            "wandb_project": config["main"]["project_name"],
            "test_size": config["segregation"]["test_size"],
            "random_state": config["segregation"]["random_state"],
            "target_column": config["segregation"]["target_column"],
        },
    )

    print("\nData segregation completed successfully!\n")


def run_model_selection(config: DictConfig, root_path: Path) -> None:
    """
    Execute the model selection step.

    Args:
        config: Hydra configuration
        root_path: Project root path
    """
    print("\n" + "=" * 70)
    print("  STEP 5: MODEL SELECTION")
    print("=" * 70)

    step_path = root_path / "src" / "model" / "05_model_selection"

    mlflow.run(
        uri=str(step_path),
        entry_point="main",
        env_manager="local",
        parameters={
            "train_artifact_name": config["model_selection"]["train_artifact_name"],
            "test_artifact_name": config["model_selection"]["test_artifact_name"],
            "gcs_train_path": config["model_selection"]["gcs_train_path"],
            "gcs_test_path": config["model_selection"]["gcs_test_path"],
            "bucket_name": config["gcs"]["bucket_name"],
            "wandb_project": config["main"]["project_name"],
            "target_column": config["model_selection"]["target_column"],
            "random_state": config["model_selection"]["random_state"],
        },
    )

    print("\nModel selection completed successfully!\n")


def run_sweep(config: DictConfig, root_path: Path) -> None:
    """
    Execute the hyperparameter sweep step.

    Args:
        config: Hydra configuration
        root_path: Project root path
    """
    print("\n" + "=" * 70)
    print("  STEP 6: HYPERPARAMETER SWEEP")
    print("=" * 70)

    step_path = root_path / "src" / "model" / "06_sweep"

    mlflow.run(
        uri=str(step_path),
        entry_point="main",
        env_manager="local",
        parameters={
            "train_artifact_name": config["sweep"]["train_artifact_name"],
            "test_artifact_name": config["sweep"]["test_artifact_name"],
            "gcs_train_path": config["sweep"]["gcs_train_path"],
            "gcs_test_path": config["sweep"]["gcs_test_path"],
            "bucket_name": config["gcs"]["bucket_name"],
            "wandb_project": config["main"]["project_name"],
            "target_column": config["sweep"]["target_column"],
            "sweep_count": config["sweep"]["sweep_count"],
        },
    )

    print("\nHyperparameter sweep completed successfully!\n")


def run_registration(config: DictConfig, root_path: Path) -> None:
    """
    Execute the model registration step.

    Args:
        config: Hydra configuration
        root_path: Project root path
    """
    print("\n" + "=" * 70)
    print("  STEP 7: MODEL REGISTRATION")
    print("=" * 70)

    step_path = root_path / "src" / "model" / "07_registration"

    mlflow.run(
        uri=str(step_path),
        entry_point="main",
        env_manager="local",
        parameters={
            "bucket_name": config["gcs"]["bucket_name"],
            "gcs_train_path": config["registration"]["gcs_train_path"],
            "gcs_test_path": config["registration"]["gcs_test_path"],
            "best_params_path": config["registration"]["best_params_path"],
            "registered_model_name": config["registration"]["registered_model_name"],
            "model_stage": config["registration"]["model_stage"],
            "target_column": config["registration"]["target_column"],
            "wandb_project": config["main"]["project_name"],
        },
    )

    print("\nModel registration completed successfully!\n")


@hydra.main(
    config_path='.',
    config_name="config",
    version_base="1.3"
)
def go(config: DictConfig) -> None:
    """
    Main orchestrator function that executes the ML pipeline.

    Args:
        config: Hydra configuration loaded from config.yaml
    """
    # Validate required environment variables before starting
    validate_environment_variables()

    # Setup environment
    setup_environment(config)

    # Get project root path
    root_path = Path(hydra.utils.get_original_cwd())

    # Get steps to execute
    steps_to_execute = get_steps_to_execute(config)

    # Track start time
    import time
    start_time = time.time()

    try:
        # Execute each step based on configuration
        if "01_download_data" in steps_to_execute:
            run_download_data(config, root_path)

        if "02_preprocessing_and_imputation" in steps_to_execute:
            run_preprocessing_and_imputation(config, root_path)

        if "03_feature_engineering" in steps_to_execute:
            run_feature_engineering(config, root_path)

        if "04_segregation" in steps_to_execute:
            run_segregation(config, root_path)

        if "05_model_selection" in steps_to_execute:
            run_model_selection(config, root_path)

        if "06_sweep" in steps_to_execute:
            run_sweep(config, root_path)

        if "07_registration" in steps_to_execute:
            run_registration(config, root_path)

        # Print summary
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("  PIPELINE EXECUTION SUMMARY")
        print("=" * 70)
        print(f"  All steps completed successfully!")
        print(f"  Total execution time: {elapsed_time:.2f} seconds")
        print(f"  GCS Bucket: gs://{config['gcs']['bucket_name']}")
        print(f"  W&B Project: {config['main']['project_name']}")
        print("=" * 70 + "\n")

    except Exception as e:
        print("\n" + "=" * 70)
        print("  PIPELINE EXECUTION FAILED")
        print("=" * 70)
        print(f"  Error: {str(e)}")
        print("=" * 70 + "\n")
        raise


if __name__ == "__main__":
    go()
