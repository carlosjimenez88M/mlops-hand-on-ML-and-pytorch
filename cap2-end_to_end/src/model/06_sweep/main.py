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

# Global variables for data (loaded once, reused across sweep runs)
X_TRAIN = None
X_TEST = None
Y_TRAIN = None
Y_TEST = None
TARGET_COLUMN = None


def train():
    """
    Training function called by W&B Sweep agent.
    This function is executed for each hyperparameter combination.
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

        # Train model
        model = train_random_forest(X_TRAIN, Y_TRAIN, params)

        # Evaluate model
        metrics = evaluate_model(model, X_TEST, Y_TEST)

        # Log metrics to W&B
        wandb.log({
            **params,
            **metrics
        })

        logger.info(f"✅ Run completed: MAPE={metrics['mape']:.2f}%")

    except Exception as e:
        logger.error(f"❌ Run failed: {str(e)}")
        wandb.log({"error": str(e), "mape": 999.9})  # Log failure
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

    # Load data ONCE (shared across all sweep runs)
    global X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, TARGET_COLUMN

    logger.info("\nLoading training data...")
    train_df = download_data_from_gcs(args.bucket_name, args.gcs_train_path)
    X_TRAIN, Y_TRAIN = prepare_data(train_df, args.target_column)

    logger.info("Loading test data...")
    test_df = download_data_from_gcs(args.bucket_name, args.gcs_test_path)
    X_TEST, Y_TEST = prepare_data(test_df, args.target_column)

    TARGET_COLUMN = args.target_column

    logger.info(f"\n✅ Data loaded:")
    logger.info(f"  Train: {X_TRAIN.shape}")
    logger.info(f"  Test: {X_TEST.shape}")

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

    logger.info(f"\n✅ Sweep created!")
    logger.info(f"  Sweep ID: {sweep_id}")
    logger.info(f"  View at: https://wandb.ai/{os.getenv('WANDB_ENTITY', 'your-entity')}/{args.wandb_project}/sweeps/{sweep_id}")

    # Run sweep agent
    logger.info(f"\n🚀 Starting sweep agent ({args.sweep_count} runs)...")
    logger.info("=" * 70)

    wandb.agent(
        sweep_id,
        function=train,
        count=args.sweep_count,
        project=args.wandb_project
    )

    logger.info("\n" + "=" * 70)
    logger.info("✅ SWEEP COMPLETED")
    logger.info("=" * 70)

    # Get best run from sweep
    try:
        api = wandb.Api()
        sweep = api.sweep(f"{os.getenv('WANDB_ENTITY', '')}/{args.wandb_project}/{sweep_id}")
        best_run = sweep.best_run()

        if best_run:
            logger.info("\n" + "=" * 70)
            logger.info("🏆 BEST HYPERPARAMETERS FOUND")
            logger.info("=" * 70)
            logger.info(f"Best run: {best_run.name} ({best_run.id})")
            logger.info(f"Best MAPE: {best_run.summary.get('mape', 'N/A'):.2f}%")
            logger.info(f"Within 10%: {best_run.summary.get('within_10pct', 'N/A'):.1f}%")
            logger.info(f"\nBest hyperparameters:")
            logger.info(f"  n_estimators: {best_run.config.get('n_estimators')}")
            logger.info(f"  max_depth: {best_run.config.get('max_depth')}")
            logger.info(f"  min_samples_split: {best_run.config.get('min_samples_split')}")
            logger.info(f"  min_samples_leaf: {best_run.config.get('min_samples_leaf')}")
            logger.info(f"  max_features: {best_run.config.get('max_features')}")

            # Save best parameters to file
            best_params_path = Path(__file__).parent / "best_params.yaml"
            best_params = {
                "sweep_id": sweep_id,
                "best_run_id": best_run.id,
                "best_run_name": best_run.name,
                "hyperparameters": {
                    "n_estimators": int(best_run.config.get('n_estimators', 100)),
                    "max_depth": int(best_run.config.get('max_depth', 10)) if best_run.config.get('max_depth') else None,
                    "min_samples_split": int(best_run.config.get('min_samples_split', 2)),
                    "min_samples_leaf": int(best_run.config.get('min_samples_leaf', 1)),
                    "max_features": best_run.config.get('max_features', 'sqrt'),
                    "random_state": 42
                },
                "metrics": {
                    "mape": float(best_run.summary.get('mape', 0)),
                    "rmse": float(best_run.summary.get('rmse', 0)),
                    "r2": float(best_run.summary.get('r2', 0)),
                    "within_10pct": float(best_run.summary.get('within_10pct', 0))
                },
                "sweep_url": f"https://wandb.ai/{os.getenv('WANDB_ENTITY', '')}/{args.wandb_project}/sweeps/{sweep_id}"
            }

            with open(best_params_path, 'w') as f:
                yaml.dump(best_params, f, default_flow_style=False)

            logger.info(f"\n💾 Best parameters saved to: {best_params_path}")
            logger.info("=" * 70)

            return sweep_id, best_params

    except Exception as e:
        logger.error(f"Could not retrieve best run: {e}")
        return sweep_id, None


if __name__ == "__main__":
    main()
