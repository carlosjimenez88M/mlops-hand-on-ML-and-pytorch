"""
Main script for model registration to MLflow.

This script:
1. Reads best hyperparameters from sweep results
2. Trains final model with optimized parameters
3. Evaluates model comprehensively
4. Registers model to MLflow Model Registry
5. Saves model locally and logs to W&B
"""
import argparse
import logging
import yaml
import mlflow
import mlflow.sklearn
import wandb
from pathlib import Path
from mlflow.tracking import MlflowClient

from config import RegistrationConfig
from models import RegistrationResult
from utils import (
    download_data_from_gcs,
    prepare_data,
    train_final_model,
    evaluate_model,
    save_model_locally,
    create_feature_importance_plot
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def register_model_to_mlflow(
    model,
    model_name: str,
    model_stage: str,
    params: dict,
    metrics: dict,
    feature_columns: list,
    target_column: str,
    gcs_train_path: str,
    gcs_test_path: str
) -> tuple:
    """
    Register model to MLflow Model Registry.

    Args:
        model: Trained sklearn model
        model_name: Name for registered model
        model_stage: Model stage (Staging/Production)
        params: Model hyperparameters
        metrics: Model evaluation metrics
        feature_columns: List of feature column names
        target_column: Target column name
        gcs_train_path: Path to training data
        gcs_test_path: Path to test data

    Returns:
        Tuple of (model_uri, model_version, run_id)
    """
    logger.info("=" * 70)
    logger.info("REGISTERING MODEL TO MLFLOW")
    logger.info("=" * 70)

    client = MlflowClient()
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model")
    logger.info(f"Model logged to MLflow: {model_uri}")

    # Create or get registered model
    try:
        client.create_registered_model(
            name=model_name,
            description=f"Housing price prediction model using Random Forest"
        )
        logger.info(f"Created new registered model: {model_name}")
    except Exception as e:
        if "already exists" in str(e):
            logger.info(f"Registered model already exists: {model_name}")
        else:
            raise

    # Create model version
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id
    )
    logger.info(f"Created model version: {model_version.version}")

    # Transition to specified stage
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage=model_stage
    )
    logger.info(f"Transitioned model to stage: {model_stage}")

    # Create comprehensive description
    description = f"""
# Housing Price Prediction Model

**Algorithm:** Random Forest Regressor

## Hyperparameters
- n_estimators: {params['n_estimators']}
- max_depth: {params.get('max_depth', 'None')}
- min_samples_split: {params['min_samples_split']}
- min_samples_leaf: {params['min_samples_leaf']}
- max_features: {params.get('max_features', 'sqrt')}

## Performance Metrics

### Primary Metrics
- **MAE**: {metrics['mae']:.2f}
- **RMSE**: {metrics['rmse']:.2f}
- **R²**: {metrics['r2']:.4f}

### Percentage Error Metrics
- **MAPE**: {metrics['mape']:.2f}%
- **SMAPE**: {metrics['smape']:.2f}%
- **wMAPE**: {metrics['wmape']:.2f}%
- **Median APE**: {metrics['median_ape']:.2f}%

### Prediction Accuracy
- Within 5%: {metrics['within_5pct']:.1f}%
- Within 10%: {metrics['within_10pct']:.1f}%
- Within 15%: {metrics['within_15pct']:.1f}%

## Features
Number of features: {len(feature_columns)}
Target: {target_column}

## Data Sources
- Training: {gcs_train_path}
- Testing: {gcs_test_path}
"""

    client.update_model_version(
        name=model_name,
        version=model_version.version,
        description=description
    )

    # Add searchable tags
    tags = {
        "algorithm": "RandomForest",
        "framework": "sklearn",
        "mae": f"{metrics['mae']:.2f}",
        "rmse": f"{metrics['rmse']:.2f}",
        "r2": f"{metrics['r2']:.4f}",
        "mape": f"{metrics['mape']:.2f}",
        "smape": f"{metrics['smape']:.2f}",
        "wmape": f"{metrics['wmape']:.2f}",
        "within_10pct": f"{metrics['within_10pct']:.1f}",
        "n_features": str(len(feature_columns)),
        "target": target_column,
    }

    for key, value in tags.items():
        client.set_model_version_tag(model_name, model_version.version, key, value)

    logger.info("Added tags to model version")
    logger.info("=" * 70)

    return model_uri, model_version.version, run_id


def main():
    """Main registration workflow."""
    parser = argparse.ArgumentParser(description="Register final model to MLflow")
    parser.add_argument("--bucket_name", type=str, required=True, help="GCS bucket name")
    parser.add_argument("--gcs_train_path", type=str, required=True, help="Path to training data in GCS")
    parser.add_argument("--gcs_test_path", type=str, required=True, help="Path to test data in GCS")
    parser.add_argument("--best_params_path", type=str, required=True, help="Path to best_params.yaml")
    parser.add_argument("--registered_model_name", type=str, default="housing_price_model", help="Name for registered model")
    parser.add_argument("--model_stage", type=str, default="Staging", help="Model stage")
    parser.add_argument("--target_column", type=str, default="median_house_value", help="Target column name")
    parser.add_argument("--wandb_project", type=str, required=True, help="W&B project name")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("MODEL REGISTRATION WORKFLOW")
    logger.info("=" * 70)

    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        name="model_registration",
        job_type="registration"
    )

    # Start MLflow run
    with mlflow.start_run(run_name="model_registration"):
        # Step 1: Load best parameters from sweep
        logger.info(f"\n1. Loading best parameters from: {args.best_params_path}")
        best_params_file = Path(args.best_params_path)

        if not best_params_file.exists():
            raise FileNotFoundError(
                f"Best parameters file not found: {args.best_params_path}. "
                "Please run sweep step (06_sweep) first."
            )

        with open(best_params_file, 'r') as f:
            best_params_data = yaml.safe_load(f)

        params = best_params_data['hyperparameters']
        sweep_metrics = best_params_data.get('metrics', {})
        sweep_id = best_params_data.get('sweep_id', 'unknown')

        logger.info(f"Loaded parameters from sweep: {sweep_id}")
        logger.info(f"Hyperparameters: {params}")
        logger.info(f"Sweep metrics: {sweep_metrics}")

        # Step 2: Download and prepare data
        logger.info("\n2. Downloading and preparing data from GCS")
        train_df = download_data_from_gcs(args.bucket_name, args.gcs_train_path)
        X_train, y_train = prepare_data(train_df, args.target_column)

        test_df = download_data_from_gcs(args.bucket_name, args.gcs_test_path)
        X_test, y_test = prepare_data(test_df, args.target_column)

        feature_columns = X_train.columns.tolist()
        logger.info(f"Features: {len(feature_columns)} columns")

        # Step 3: Train final model with optimized hyperparameters
        logger.info("\n3. Training final model with optimized hyperparameters")
        model = train_final_model(X_train, y_train, params)

        # Step 4: Evaluate model
        logger.info("\n4. Evaluating model on test set")
        metrics = evaluate_model(model, X_test, y_test)

        # Log parameters and metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("sweep_id", sweep_id)

        # Log to W&B
        wandb.log({
            **params,
            **metrics,
            "n_features": len(feature_columns),
            "sweep_id": sweep_id
        })

        # Step 4.5: Create and log feature importance plot
        logger.info("\n4.5. Creating feature importance visualization")
        plot_path = create_feature_importance_plot(model, feature_columns)
        if plot_path and plot_path.exists():
            wandb.log({"feature_importance": wandb.Image(str(plot_path))})
            mlflow.log_artifact(str(plot_path), artifact_path="plots")
            logger.info("Feature importance plot logged to W&B and MLflow")

        # Step 5: Register model to MLflow
        logger.info("\n5. Registering model to MLflow Model Registry")
        model_uri, model_version, run_id = register_model_to_mlflow(
            model=model,
            model_name=args.registered_model_name,
            model_stage=args.model_stage,
            params=params,
            metrics=metrics,
            feature_columns=feature_columns,
            target_column=args.target_column,
            gcs_train_path=args.gcs_train_path,
            gcs_test_path=args.gcs_test_path
        )

        # Step 6: Save model locally
        logger.info("\n6. Saving model locally")
        models_dir = Path("models/trained")
        model_path = models_dir / f"{args.registered_model_name}.pkl"
        save_model_locally(model, model_path)

        # Step 7: Generate model config file
        logger.info("\n7. Generating model configuration file")

        # Get project root (4 levels up from this file)
        project_root = Path(__file__).parent.parent.parent.parent
        config_dir = project_root / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        model_config = {
            'model': {
                'name': args.registered_model_name,
                'version': str(model_version),
                'stage': args.model_stage,
                'best_model': 'RandomForest',
                'parameters': params,
                'r2_score': float(metrics['r2']),
                'mae': float(metrics['mae']),
                'rmse': float(metrics['rmse']),
                'mape': float(metrics['mape']),
                'target_variable': args.target_column,
                'num_features': len(feature_columns),
                'feature_columns': feature_columns,
                'mlflow_run_id': run_id,
                'mlflow_model_uri': model_uri,
                'local_path': str(model_path),
                'sweep_id': sweep_id
            }
        }

        config_path = config_dir / "model_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(model_config, f, default_flow_style=False, indent=2)

        logger.info(f"Model config saved to: {config_path}")

        # Log config file to MLflow
        mlflow.log_artifact(str(config_path), artifact_path="config")
        logger.info("Config logged to MLflow")

        # Create registration result
        result = RegistrationResult(
            model_name=args.registered_model_name,
            model_version=str(model_version),
            model_stage=args.model_stage,
            model_uri=model_uri,
            run_id=run_id,
            hyperparameters=params,
            metrics=metrics,
            feature_columns=feature_columns
        )

        # Log summary
        logger.info("\n" + "=" * 70)
        logger.info("REGISTRATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Model Name: {result.model_name}")
        logger.info(f"Model Version: {result.model_version}")
        logger.info(f"Model Stage: {result.model_stage}")
        logger.info(f"MLflow Run ID: {result.run_id}")
        logger.info(f"Model URI: {result.model_uri}")
        logger.info(f"Local Path: {model_path}")
        logger.info("=" * 70)

        # Log registration info to W&B
        wandb.log({
            "model_name": result.model_name,
            "model_version": result.model_version,
            "model_stage": result.model_stage,
            "mlflow_run_id": result.run_id
        })

    wandb.finish()
    logger.info("\n Model registration completed successfully!")


if __name__ == "__main__":
    main()
