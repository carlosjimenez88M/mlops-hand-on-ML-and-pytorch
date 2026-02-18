"""
Imputation Analysis Module - Compares different imputation strategies
Author: Carlos Daniel Jiménez
Date: 2025-11-28
"""

import logging
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    sys.path.insert(0, str(__file__).rsplit("/", 5)[0])
    from src.utils.colored_logger import setup_colored_logger

    logger = setup_colored_logger(__name__)
except (ImportError, Exception):
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)


@dataclass
class ImputationResult:
    """Result of an imputation strategy evaluation."""

    method_name: str
    rmse: float
    imputed_values: np.ndarray
    imputer: object


class ImputationAnalyzer:
    """
    Analyzes and compares different imputation strategies.
    Automatically selects the best method based on RMSE.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str = "total_bedrooms",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialize the analyzer.

        Args:
            df: Input DataFrame
            target_column: Column to analyze for imputation
            test_size: Proportion for validation set
            random_state: Random seed for reproducibility
        """
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.results: Dict[str, ImputationResult] = {}
        self.best_method: Optional[str] = None
        self.best_imputer: Optional[object] = None

    def analyze_missing_values(self) -> Dict[str, any]:
        """
        Analyzes missing values in the target column.

        Returns:
            Dictionary with missing value statistics
        """
        missing_count = self.df[self.target_column].isnull().sum()
        missing_pct = (missing_count / len(self.df)) * 100

        stats = {
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "total_rows": len(self.df),
        }

        logger.info("=" * 70)
        logger.info(f"MISSING VALUES ANALYSIS: '{self.target_column}'")
        logger.info("=" * 70)
        logger.info(f"  Total missing values: {missing_count}")
        logger.info(f"  Percentage: {missing_pct:.2f}%")
        logger.info(f"  Total rows: {len(self.df):,}")
        logger.info("=" * 70)

        return stats

    def compute_correlation_matrix(self) -> pd.DataFrame:
        """
        Computes correlation matrix for numeric columns.

        Returns:
            Correlation matrix DataFrame
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        logger.info("\nCorrelation with target column:")
        logger.info(corr_matrix[self.target_column].sort_values(ascending=False))

        return corr_matrix

    def create_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> Figure:
        """
        Creates correlation heatmap visualization.

        Args:
            corr_matrix: Correlation matrix

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Correlation Matrix of Numeric Features", fontsize=16, fontweight="bold")
        plt.tight_layout()

        logger.info("Created correlation heatmap")

        return fig

    def prepare_validation_set(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Prepares training and validation sets for imputation comparison.

        Returns:
            Tuple of (train_set, val_set_missing, y_val_true)
        """
        logger.info("\nPreparing validation sets...")

        # Select only numeric columns
        housing_numeric = self.df.select_dtypes(include=[np.number])

        # Remove rows with missing target values
        housing_known = housing_numeric.dropna(subset=[self.target_column])

        # Split into train and validation
        train_set, val_set = train_test_split(
            housing_known, test_size=self.test_size, random_state=self.random_state
        )

        # Create validation set with masked target column
        val_set_missing = val_set.copy()
        val_set_missing[self.target_column] = np.nan

        # Save ground truth
        y_val_true = val_set[self.target_column].copy()

        logger.info(f"  Training set shape: {train_set.shape}")
        logger.info(f"  Validation set shape: {val_set.shape}")
        logger.info(f"  Ground truth values: {len(y_val_true)}")

        return train_set, val_set_missing, y_val_true

    def evaluate_simple_imputer(
        self,
        train_set: pd.DataFrame,
        val_set_missing: pd.DataFrame,
        y_val_true: pd.Series,
        strategy: str = "median",
    ) -> ImputationResult:
        """
        Evaluates Simple Imputer strategy.

        Args:
            train_set: Training data
            val_set_missing: Validation data with masked values
            y_val_true: Ground truth values
            strategy: Imputation strategy (mean, median, mode)

        Returns:
            ImputationResult object
        """
        logger.info(f"\nEvaluating Simple Imputer ({strategy})...")

        imputer = SimpleImputer(strategy=strategy)
        imputer.fit(train_set)

        val_imputed = imputer.transform(val_set_missing)

        target_col_idx = train_set.columns.get_loc(self.target_column)
        y_val_pred = val_imputed[:, target_col_idx]

        rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))

        logger.info(f"  RMSE: {rmse:.4f}")

        return ImputationResult(
            method_name=f"Simple Imputer ({strategy})",
            rmse=rmse,
            imputed_values=y_val_pred,
            imputer=imputer,
        )

    def evaluate_knn_imputer(
        self,
        train_set: pd.DataFrame,
        val_set_missing: pd.DataFrame,
        y_val_true: pd.Series,
        n_neighbors: int = 5,
    ) -> ImputationResult:
        """
        Evaluates KNN Imputer strategy with StandardScaler.

        Args:
            train_set: Training data
            val_set_missing: Validation data with masked values
            y_val_true: Ground truth values
            n_neighbors: Number of neighbors

        Returns:
            ImputationResult object
        """
        logger.info(f"\nEvaluating KNN Imputer (k={n_neighbors})...")

        # Suppress RuntimeWarnings from KNN calculations (expected with unscaled data)
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Scale data before KNN to avoid overflow issues
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_set)
            val_scaled = scaler.transform(val_set_missing)

            # Apply KNN imputation on scaled data
            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputer.fit(train_scaled)
            val_imputed_scaled = imputer.transform(val_scaled)

            # Inverse transform to get original scale
            val_imputed = scaler.inverse_transform(val_imputed_scaled)

        target_col_idx = train_set.columns.get_loc(self.target_column)
        y_val_pred = val_imputed[:, target_col_idx]

        rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))

        logger.info(f"  RMSE: {rmse:.4f}")

        # Store both scaler and imputer for later use
        return ImputationResult(
            method_name=f"KNN Imputer (k={n_neighbors})",
            rmse=rmse,
            imputed_values=y_val_pred,
            imputer=(scaler, imputer),  # Store tuple of (scaler, imputer)
        )

    def evaluate_iterative_imputer(
        self, train_set: pd.DataFrame, val_set_missing: pd.DataFrame, y_val_true: pd.Series
    ) -> ImputationResult:
        """
        Evaluates Iterative Imputer with Random Forest.

        Args:
            train_set: Training data
            val_set_missing: Validation data with masked values
            y_val_true: Ground truth values

        Returns:
            ImputationResult object
        """
        logger.info("\nEvaluating Iterative Imputer (Random Forest)...")

        estimator = RandomForestRegressor(n_jobs=-1, random_state=self.random_state)
        imputer = IterativeImputer(estimator=estimator, random_state=self.random_state)

        imputer.fit(train_set)

        val_imputed = imputer.transform(val_set_missing)

        target_col_idx = train_set.columns.get_loc(self.target_column)
        y_val_pred = val_imputed[:, target_col_idx]

        rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))

        logger.info(f"  RMSE: {rmse:.4f}")

        return ImputationResult(
            method_name="Iterative Imputer (RF)",
            rmse=rmse,
            imputed_values=y_val_pred,
            imputer=imputer,
        )

    def compare_all_methods(self) -> Dict[str, ImputationResult]:
        """
        Compares all imputation methods and selects the best one.

        Returns:
            Dictionary of all results
        """
        logger.info("\n" + "=" * 70)
        logger.info("COMPARING IMPUTATION METHODS")
        logger.info("=" * 70)

        # Prepare validation sets
        train_set, val_set_missing, y_val_true = self.prepare_validation_set()

        # Evaluate all methods
        self.results["simple_median"] = self.evaluate_simple_imputer(
            train_set, val_set_missing, y_val_true, strategy="median"
        )

        self.results["simple_mean"] = self.evaluate_simple_imputer(
            train_set, val_set_missing, y_val_true, strategy="mean"
        )

        self.results["knn"] = self.evaluate_knn_imputer(
            train_set, val_set_missing, y_val_true, n_neighbors=5
        )

        self.results["iterative_rf"] = self.evaluate_iterative_imputer(
            train_set, val_set_missing, y_val_true
        )

        # Select best method
        best_key = min(self.results, key=lambda k: self.results[k].rmse)
        self.best_method = best_key
        self.best_imputer = self.results[best_key].imputer

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("IMPUTATION METHODS COMPARISON - RESULTS")
        logger.info("=" * 70)
        for key, result in sorted(self.results.items(), key=lambda x: x[1].rmse):
            status = "BEST" if key == best_key else ""
            logger.info(f"  {result.method_name:30s} RMSE: {result.rmse:8.4f} {status}")
        logger.info("=" * 70)
        logger.info(f"Best method selected: {self.results[best_key].method_name}")
        logger.info("=" * 70)

        return self.results

    def create_comparison_plot(self) -> Figure:
        """
        Creates a bar plot comparing RMSE of different methods.

        Returns:
            Matplotlib figure
        """
        if not self.results:
            raise ValueError("No results to plot. Run compare_all_methods() first.")

        methods = [result.method_name for result in self.results.values()]
        rmses = [result.rmse for result in self.results.values()]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(
            methods,
            rmses,
            color=["green" if i == np.argmin(rmses) else "skyblue" for i in range(len(rmses))],
        )

        ax.set_xlabel("Imputation Method", fontsize=12, fontweight="bold")
        ax.set_ylabel("RMSE", fontsize=12, fontweight="bold")
        ax.set_title("Comparison of Imputation Methods", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, rmse in zip(bars, rmses):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{rmse:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        logger.info("Created comparison plot")

        return fig

    def apply_best_imputer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the best imputer to the full dataset.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with imputed values
        """
        if self.best_imputer is None:
            raise ValueError("No best imputer selected. Run compare_all_methods() first.")

        logger.info("\nApplying best imputer to full dataset...")

        df_out = df.copy()
        numeric_df = df_out.select_dtypes(include=[np.number])

        # Suppress RuntimeWarnings from KNN if applicable
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Check if imputer is a tuple (scaler, imputer) for KNN
            if isinstance(self.best_imputer, tuple):
                scaler, imputer = self.best_imputer
                # Scale, impute, then inverse transform
                numeric_scaled = scaler.transform(numeric_df)
                imputed_scaled = imputer.transform(numeric_scaled)
                imputed_array = scaler.inverse_transform(imputed_scaled)
            else:
                # Direct imputation for other methods
                imputed_array = self.best_imputer.transform(numeric_df)

        target_col_idx = numeric_df.columns.get_loc(self.target_column)
        df_out[self.target_column] = imputed_array[:, target_col_idx]

        missing_after = df_out[self.target_column].isnull().sum()
        logger.info(f"  Missing values after imputation: {missing_after}")

        return df_out

    def get_metrics_dict(self) -> Dict[str, float]:
        """
        Returns metrics dictionary for logging.

        Returns:
            Dictionary with method names and RMSE values
        """
        return {f"imputation_rmse_{key}": result.rmse for key, result in self.results.items()}
