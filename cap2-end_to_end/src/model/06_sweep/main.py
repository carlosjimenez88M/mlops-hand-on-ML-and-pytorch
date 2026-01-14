"""
W&B Sweep for Random Forest Hyperparameter Optimization.
Based on: https://wandb.ai/aman-arora/mlops-course-001/reports/Random-Forest-Regression
"""
import argparse
import os
import sys
import yaml
import wandb
import logging
from pathlib import Path

from utils import (
    download_data_from_gcs,
    prepare_data,
    train_random_forest,
    evaluate_model
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Module-level data cache (loaded once, reused across sweep runs)
_data_cache = {
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "feature_names": None
}


def train():
    """
    Training function called by W&B Sweep agent.
    This function is executed for each hyperparameter combination.

    Uses module-level data cache to avoid reloading data on each run.
    """
    # Initialize W&B run (managed by sweep agent)
    run = wandb.init()

    # Get hyperparameters from sweep config
    config = wandb.config

    logger.info("=" * 70)
    logger.info(f"SWEEP RUN: {run.name}")
    logger.info("=" * 70)
    logger.info(f"Hyperparameters:")
    logger.info(f"  n_estimators: {config.n_estimators}")
    logger.info(f"  max_depth: {config.max_depth}")
    logger.info(f"  min_samples_split: {config.min_samples_split}")
    logger.info(f"  min_samples_leaf: {config.min_samples_leaf}")
    logger.info(f"  max_features: {config.max_features}")

    try:
        # Prepare parameters
        params = {
            'n_estimators': int(config.n_estimators),
            'max_depth': int(config.max_depth) if config.max_depth else None,
            'min_samples_split': int(config.min_samples_split),
            'min_samples_leaf': int(config.min_samples_leaf),
            'max_features': config.max_features,
            'random_state': 42
        }

        # Train model using cached data
        model = train_random_forest(
            _data_cache["X_train"],
            _data_cache["y_train"],
            params
        )

        # Evaluate model
        metrics = evaluate_model(
            model,
            _data_cache["X_test"],
            _data_cache["y_test"]
        )

        # Log feature importances to W&B
        from utils import log_feature_importances
        feature_importances = log_feature_importances(
            model,
            _data_cache["feature_names"]
        )

        # Log metrics and feature importances to W&B
        wandb.log({
            **params,
            **metrics,
            **{f"feature_importance_{k}": v for k, v in list(feature_importances.items())[:10]}
        })

        logger.info(f" Run completed: MAPE={metrics['mape']:.2f}% | "
                   f"SMAPE={metrics['smape']:.2f}% | wMAPE={metrics['wmape']:.2f}%")

    except Exception as e:
        logger.error(f" Run failed: {str(e)}")
        # Log failure with high error score
        wandb.log({
            "error": str(e),
            "mape": 999.9,
            "smape": 999.9,
            "wmape": 999.9
        })
        raise

    finally:
        run.finish()


def main():
    """
    Main function to initialize and run the W&B Sweep.
    """
    parser = argparse.ArgumentParser(description="W&B Sweep for Random Forest Optimization")

    parser.add_argument("--train_artifact_name", type=str, required=True)
    parser.add_argument("--test_artifact_name", type=str, required=True)
    parser.add_argument("--gcs_train_path", type=str, required=True)
    parser.add_argument("--gcs_test_path", type=str, required=True)
    parser.add_argument("--bucket_name", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--target_column", type=str, default="median_house_value")
    parser.add_argument("--sweep_count", type=int, default=50)
    parser.add_argument("--sweep_config", type=str, default="sweep_config.yaml")

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("W&B SWEEP - HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 70)
    logger.info(f"Project: {args.wandb_project}")
    logger.info(f"Training data: gs://{args.bucket_name}/{args.gcs_train_path}")
    logger.info(f"Test data: gs://{args.bucket_name}/{args.gcs_test_path}")
    logger.info(f"Target: {args.target_column}")
    logger.info(f"Sweep runs: {args.sweep_count}")

    # Load data ONCE into module-level cache (shared across all sweep runs)
    logger.info("\nLoading training data...")
    train_df = download_data_from_gcs(args.bucket_name, args.gcs_train_path)
    X_train, y_train = prepare_data(train_df, args.target_column)

    logger.info("Loading test data...")
    test_df = download_data_from_gcs(args.bucket_name, args.gcs_test_path)
    X_test, y_test = prepare_data(test_df, args.target_column)

    # Store in module-level cache
    _data_cache["X_train"] = X_train
    _data_cache["X_test"] = X_test
    _data_cache["y_train"] = y_train
    _data_cache["y_test"] = y_test
    _data_cache["feature_names"] = X_train.columns.tolist()

    logger.info(f"\n Data loaded:")
    logger.info(f"  Train: {X_train.shape}")
    logger.info(f"  Test: {X_test.shape}")
    logger.info(f"  Features: {len(_data_cache['feature_names'])}")

    # Load sweep configuration
    sweep_config_path = Path(__file__).parent / args.sweep_config

    if not sweep_config_path.exists():
        raise FileNotFoundError(f"Sweep config not found: {sweep_config_path}")

    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)

    logger.info(f"\nSweep configuration:")
    logger.info(f"  Method: {sweep_config['method']}")
    logger.info(f"  Metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")

    # Initialize sweep
    logger.info("\nInitializing W&B Sweep...")
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=args.wandb_project
    )

    logger.info(f"\n Sweep created!")
    logger.info(f"  Sweep ID: {sweep_id}")
    logger.info(f"  View at: https://wandb.ai/{os.getenv('WANDB_ENTITY', 'your-entity')}/{args.wandb_project}/sweeps/{sweep_id}")

    # Run sweep agent
    logger.info(f"\n Starting sweep agent ({args.sweep_count} runs)...")
    logger.info("=" * 70)

    wandb.agent(
        sweep_id,
        function=train,
        count=args.sweep_count,
        project=args.wandb_project
    )

    logger.info("\n" + "=" * 70)
    logger.info(" SWEEP COMPLETED")
    logger.info("=" * 70)

    # Get best run from sweep
    try:
        api = wandb.Api()
        sweep = api.sweep(f"{os.getenv('WANDB_ENTITY', '')}/{args.wandb_project}/{sweep_id}")
        best_run = sweep.best_run()

        if best_run:
            logger.info("\n" + "=" * 70)
            logger.info(" BEST HYPERPARAMETERS FOUND")
            logger.info("=" * 70)
            logger.info(f"Best run: {best_run.name} ({best_run.id})")
            logger.info(f"\nPerformance Metrics:")
            logger.info(f"  MAE: {best_run.summary.get('mae', 'N/A'):.2f}")
            logger.info(f"  RMSE: {best_run.summary.get('rmse', 'N/A'):.2f}")
            logger.info(f"  R²: {best_run.summary.get('r2', 'N/A'):.4f}")
            logger.info(f"  MAPE: {best_run.summary.get('mape', 'N/A'):.2f}%")
            logger.info(f"  SMAPE: {best_run.summary.get('smape', 'N/A'):.2f}%")
            logger.info(f"  wMAPE: {best_run.summary.get('wmape', 'N/A'):.2f}%")
            logger.info(f"  Within 10%: {best_run.summary.get('within_10pct', 'N/A'):.1f}%")
            logger.info(f"\nBest hyperparameters:")
            logger.info(f"  n_estimators: {best_run.config.get('n_estimators')}")
            logger.info(f"  max_depth: {best_run.config.get('max_depth')}")
            logger.info(f"  min_samples_split: {best_run.config.get('min_samples_split')}")
            logger.info(f"  min_samples_leaf: {best_run.config.get('min_samples_leaf')}")
            logger.info(f"  max_features: {best_run.config.get('max_features')}")

            # Save best parameters to file (NO DEFAULTS - all must come from sweep)
            best_params_path = Path(__file__).parent / "best_params.yaml"

            # Validate that all required hyperparameters are present
            required_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
            missing_params = [p for p in required_params if p not in best_run.config]
            if missing_params:
                raise ValueError(
                    f"Missing required hyperparameters from sweep results: {missing_params}. "
                    "All hyperparameters must be determined by the sweep, no defaults allowed."
                )

            best_params = {
                "sweep_id": sweep_id,
                "best_run_id": best_run.id,
                "best_run_name": best_run.name,
                "hyperparameters": {
                    "n_estimators": int(best_run.config['n_estimators']),
                    "max_depth": int(best_run.config['max_depth']) if best_run.config['max_depth'] else None,
                    "min_samples_split": int(best_run.config['min_samples_split']),
                    "min_samples_leaf": int(best_run.config['min_samples_leaf']),
                    "max_features": best_run.config['max_features'],
                    "random_state": 42  # Fixed for reproducibility, not a hyperparameter to optimize
                },
                "metrics": {
                    # Primary metrics
                    "mae": float(best_run.summary.get('mae', 0)),
                    "rmse": float(best_run.summary.get('rmse', 0)),
                    "r2": float(best_run.summary.get('r2', 0)),
                    # Percentage error metrics
                    "mape": float(best_run.summary.get('mape', 0)),
                    "smape": float(best_run.summary.get('smape', 0)),
                    "wmape": float(best_run.summary.get('wmape', 0)),
                    "median_ape": float(best_run.summary.get('median_ape', 0)),
                    # Accuracy within thresholds
                    "within_5pct": float(best_run.summary.get('within_5pct', 0)),
                    "within_10pct": float(best_run.summary.get('within_10pct', 0)),
                    "within_15pct": float(best_run.summary.get('within_15pct', 0))
                },
                "sweep_url": f"https://wandb.ai/{os.getenv('WANDB_ENTITY', '')}/{args.wandb_project}/sweeps/{sweep_id}"
            }

            with open(best_params_path, 'w') as f:
                yaml.dump(best_params, f, default_flow_style=False)

            logger.info(f"\n Best parameters saved to: {best_params_path}")
            logger.info("=" * 70)

            return sweep_id, best_params

    except Exception as e:
        logger.error(f"Could not retrieve best run: {e}")
        return sweep_id, None


if __name__ == "__main__":
    main()
