"""
Unit tests for ImputationAnalyzer class
Author: Carlos Daniel Jiménez
Date: 2025-01-13
"""

import numpy as np
import pandas as pd
import pytest
from imputation_analyzer import ImputationAnalyzer, ImputationResult
from matplotlib.figure import Figure


class TestImputationAnalyzer:
    """Test suite for ImputationAnalyzer class."""

    def test_init(self, sample_housing_data):
        """Test initialization of ImputationAnalyzer."""
        analyzer = ImputationAnalyzer(
            df=sample_housing_data, target_column="total_bedrooms", test_size=0.2, random_state=42
        )

        assert analyzer.df.equals(sample_housing_data)
        assert analyzer.target_column == "total_bedrooms"
        assert analyzer.test_size == 0.2
        assert analyzer.random_state == 42
        assert analyzer.results == {}
        assert analyzer.best_method is None
        assert analyzer.best_imputer is None

    def test_analyze_missing_values(self, sample_housing_data):
        """Test analyzing missing values."""
        analyzer = ImputationAnalyzer(df=sample_housing_data, target_column="total_bedrooms")

        stats = analyzer.analyze_missing_values()

        assert "missing_count" in stats
        assert "missing_percentage" in stats
        assert "total_rows" in stats
        assert stats["missing_count"] > 0
        assert stats["total_rows"] == len(sample_housing_data)

    def test_compute_correlation_matrix(self, sample_housing_data):
        """Test computing correlation matrix."""
        analyzer = ImputationAnalyzer(df=sample_housing_data, target_column="total_bedrooms")

        corr_matrix = analyzer.compute_correlation_matrix()

        assert "total_bedrooms" in corr_matrix.columns
        assert "total_bedrooms" in corr_matrix.index
        assert corr_matrix.shape[0] == corr_matrix.shape[1]

    def test_create_correlation_heatmap(self, sample_housing_data):
        """Test creating correlation heatmap."""
        analyzer = ImputationAnalyzer(df=sample_housing_data, target_column="total_bedrooms")

        corr_matrix = analyzer.compute_correlation_matrix()
        fig = analyzer.create_correlation_heatmap(corr_matrix)

        assert isinstance(fig, Figure)

    def test_prepare_validation_set(self, sample_housing_data):
        """Test preparing validation sets."""
        analyzer = ImputationAnalyzer(
            df=sample_housing_data, target_column="total_bedrooms", test_size=0.2, random_state=42
        )

        train_set, val_set_missing, y_val_true = analyzer.prepare_validation_set()

        assert len(train_set) > 0
        assert len(val_set_missing) > 0
        assert len(y_val_true) > 0
        assert len(val_set_missing) == len(y_val_true)
        assert val_set_missing["total_bedrooms"].isnull().all()

    def test_evaluate_simple_imputer_median(self, sample_housing_data):
        """Test evaluating simple imputer with median strategy."""
        analyzer = ImputationAnalyzer(
            df=sample_housing_data, target_column="total_bedrooms", random_state=42
        )

        train_set, val_set_missing, y_val_true = analyzer.prepare_validation_set()

        result = analyzer.evaluate_simple_imputer(
            train_set, val_set_missing, y_val_true, strategy="median"
        )

        assert isinstance(result, ImputationResult)
        assert result.method_name == "Simple Imputer (median)"
        assert result.rmse > 0
        assert len(result.imputed_values) == len(y_val_true)
        assert result.imputer is not None

    def test_evaluate_simple_imputer_mean(self, sample_housing_data):
        """Test evaluating simple imputer with mean strategy."""
        analyzer = ImputationAnalyzer(
            df=sample_housing_data, target_column="total_bedrooms", random_state=42
        )

        train_set, val_set_missing, y_val_true = analyzer.prepare_validation_set()

        result = analyzer.evaluate_simple_imputer(
            train_set, val_set_missing, y_val_true, strategy="mean"
        )

        assert isinstance(result, ImputationResult)
        assert result.method_name == "Simple Imputer (mean)"
        assert result.rmse > 0

    def test_evaluate_knn_imputer(self, sample_housing_data):
        """Test evaluating KNN imputer."""
        analyzer = ImputationAnalyzer(
            df=sample_housing_data, target_column="total_bedrooms", random_state=42
        )

        train_set, val_set_missing, y_val_true = analyzer.prepare_validation_set()

        result = analyzer.evaluate_knn_imputer(
            train_set, val_set_missing, y_val_true, n_neighbors=5
        )

        assert isinstance(result, ImputationResult)
        assert result.method_name == "KNN Imputer (k=5)"
        assert result.rmse > 0
        assert isinstance(result.imputer, tuple)
        assert len(result.imputer) == 2

    def test_evaluate_iterative_imputer(self, sample_housing_data):
        """Test evaluating iterative imputer."""
        analyzer = ImputationAnalyzer(
            df=sample_housing_data, target_column="total_bedrooms", random_state=42
        )

        train_set, val_set_missing, y_val_true = analyzer.prepare_validation_set()

        result = analyzer.evaluate_iterative_imputer(train_set, val_set_missing, y_val_true)

        assert isinstance(result, ImputationResult)
        assert result.method_name == "Iterative Imputer (RF)"
        assert result.rmse > 0
        assert result.imputer is not None

    def test_compare_all_methods(self, sample_housing_data):
        """Test comparing all imputation methods."""
        analyzer = ImputationAnalyzer(
            df=sample_housing_data, target_column="total_bedrooms", random_state=42
        )

        results = analyzer.compare_all_methods()

        assert len(results) == 4
        assert "simple_median" in results
        assert "simple_mean" in results
        assert "knn" in results
        assert "iterative_rf" in results
        assert analyzer.best_method is not None
        assert analyzer.best_imputer is not None

        # Check that best method has lowest RMSE
        best_rmse = results[analyzer.best_method].rmse
        for key, result in results.items():
            assert result.rmse >= best_rmse

    def test_create_comparison_plot(self, sample_housing_data):
        """Test creating comparison plot."""
        analyzer = ImputationAnalyzer(
            df=sample_housing_data, target_column="total_bedrooms", random_state=42
        )

        analyzer.compare_all_methods()
        fig = analyzer.create_comparison_plot()

        assert isinstance(fig, Figure)

    def test_create_comparison_plot_no_results(self, sample_housing_data):
        """Test creating comparison plot without results raises error."""
        analyzer = ImputationAnalyzer(df=sample_housing_data, target_column="total_bedrooms")

        with pytest.raises(ValueError, match="No results to plot"):
            analyzer.create_comparison_plot()

    def test_apply_best_imputer_simple(self, sample_housing_data):
        """Test applying best imputer with Simple Imputer."""
        analyzer = ImputationAnalyzer(
            df=sample_housing_data, target_column="total_bedrooms", random_state=42
        )

        # Run comparison to select best method
        analyzer.compare_all_methods()

        # Apply best imputer
        df_imputed = analyzer.apply_best_imputer(sample_housing_data)

        assert len(df_imputed) == len(sample_housing_data)
        assert df_imputed["total_bedrooms"].isnull().sum() == 0

    def test_apply_best_imputer_no_selection(self, sample_housing_data):
        """Test applying best imputer without selection raises error."""
        analyzer = ImputationAnalyzer(df=sample_housing_data, target_column="total_bedrooms")

        with pytest.raises(ValueError, match="No best imputer selected"):
            analyzer.apply_best_imputer(sample_housing_data)

    def test_apply_best_imputer_knn(self, sample_housing_data):
        """Test applying best imputer when KNN is selected."""
        analyzer = ImputationAnalyzer(
            df=sample_housing_data, target_column="total_bedrooms", random_state=42
        )

        # Run comparison
        analyzer.compare_all_methods()

        # Force KNN as best method for testing
        if "knn" in analyzer.results:
            analyzer.best_method = "knn"
            analyzer.best_imputer = analyzer.results["knn"].imputer

            df_imputed = analyzer.apply_best_imputer(sample_housing_data)

            assert len(df_imputed) == len(sample_housing_data)
            assert df_imputed["total_bedrooms"].isnull().sum() == 0

    def test_get_metrics_dict(self, sample_housing_data):
        """Test getting metrics dictionary."""
        analyzer = ImputationAnalyzer(
            df=sample_housing_data, target_column="total_bedrooms", random_state=42
        )

        analyzer.compare_all_methods()
        metrics = analyzer.get_metrics_dict()

        assert isinstance(metrics, dict)
        assert len(metrics) == 4
        assert "imputation_rmse_simple_median" in metrics
        assert "imputation_rmse_simple_mean" in metrics
        assert "imputation_rmse_knn" in metrics
        assert "imputation_rmse_iterative_rf" in metrics

        # All values should be floats
        for value in metrics.values():
            assert isinstance(value, (int, float))

    def test_different_target_columns(self):
        """Test analyzer works with different target columns."""
        # Create data with different missing pattern
        np.random.seed(42)
        data = {
            "col1": np.random.randint(1, 100, 50),
            "col2": np.random.randint(1, 100, 50),
            "col3": np.random.randint(1, 100, 50),
        }
        df = pd.DataFrame(data)

        # Add missing values to col2
        missing_indices = np.random.choice(50, size=10, replace=False)
        df.loc[missing_indices, "col2"] = np.nan

        analyzer = ImputationAnalyzer(df=df, target_column="col2", random_state=42)

        results = analyzer.compare_all_methods()

        assert len(results) > 0
        assert analyzer.best_method is not None

    def test_high_missing_percentage(self):
        """Test analyzer handles high missing percentage."""
        np.random.seed(42)
        n_samples = 100

        data = {
            "col1": np.random.randint(1, 100, n_samples),
            "col2": np.random.randint(1, 100, n_samples),
            "col3": np.random.randint(1, 100, n_samples),
        }
        df = pd.DataFrame(data)

        # Add 50% missing values
        missing_indices = np.random.choice(n_samples, size=50, replace=False)
        df.loc[missing_indices, "col2"] = np.nan

        analyzer = ImputationAnalyzer(df=df, target_column="col2", random_state=42)

        stats = analyzer.analyze_missing_values()

        assert stats["missing_percentage"] == 50.0
        assert stats["missing_count"] == 50
