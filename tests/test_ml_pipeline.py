"""Unit tests for ml_pipeline.py.

Tests the _get_widget_value utility and the __init__ config-parsing logic of
BaseLayer, FeatureEngineering, and ModelTraining without a live Spark cluster
or MLflow server (conftest.py mocks Spark/Databricks/MLflow/sklearn deps).
"""

import logging
import os
from unittest.mock import patch

import pytest
import yaml

import ml_pipeline as mlp

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "dev.yaml")


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# --- _get_widget_value ---


class TestGetWidgetValue:
    """Test the _get_widget_value utility function."""

    @patch.object(mlp, "dbutils")
    def test_returns_widget_value_when_available(self, mock_dbutils) -> None:
        mock_dbutils.widgets.get.return_value = "/path/to/config.yaml"
        assert mlp._get_widget_value("config_file") == "/path/to/config.yaml"

    @patch.object(mlp, "dbutils")
    def test_returns_default_when_widget_missing(self, mock_dbutils) -> None:
        mock_dbutils.widgets.get.side_effect = Exception("No such widget")
        assert mlp._get_widget_value("config_file", "fallback.yaml") == "fallback.yaml"

    @patch.object(mlp, "dbutils")
    def test_returns_empty_string_default(self, mock_dbutils) -> None:
        mock_dbutils.widgets.get.side_effect = Exception("No such widget")
        assert mlp._get_widget_value("missing_widget") == ""

    @patch.object(mlp, "dbutils")
    def test_calls_widgets_get_with_correct_name(self, mock_dbutils) -> None:
        mock_dbutils.widgets.get.return_value = "value"
        mlp._get_widget_value("source_path")
        mock_dbutils.widgets.get.assert_called_once_with("source_path")


# --- BaseLayer ---


class TestBaseLayer:
    """Test the BaseLayer base class config loading and setup."""

    def test_config_path_stored(self) -> None:
        layer = mlp.BaseLayer(CONFIG_PATH, "test_stage")
        assert layer.config_path == CONFIG_PATH

    def test_layer_name_stored(self) -> None:
        layer = mlp.BaseLayer(CONFIG_PATH, "feature_engineering")
        assert layer.layer_name == "feature_engineering"

    def test_config_loaded(self, cfg: dict) -> None:
        layer = mlp.BaseLayer(CONFIG_PATH, "test_stage")
        assert layer.cfg["catalog"] == cfg["catalog"]
        assert layer.cfg["schemas"] == cfg["schemas"]

    def test_catalog_extracted(self) -> None:
        layer = mlp.BaseLayer(CONFIG_PATH, "test_stage")
        assert layer.catalog == "mle"

    def test_logger_is_logging_logger(self) -> None:
        layer = mlp.BaseLayer(CONFIG_PATH, "test_stage")
        assert isinstance(layer.logger, logging.Logger)
        assert layer.logger.name == "test_stage"

    def test_invalid_config_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            mlp.BaseLayer("/nonexistent/path.yaml", "test_stage")


# --- FeatureEngineering ---


class TestFeatureEngineering:
    """Test FeatureEngineering __init__ config parsing."""

    @pytest.fixture(scope="class")
    def fe(self) -> mlp.FeatureEngineering:
        return mlp.FeatureEngineering(config_path=CONFIG_PATH)

    def test_layer_cfg_is_fe_section(self, fe, cfg: dict) -> None:
        assert fe.layer_cfg == cfg["feature_engineering"]

    def test_gold_schema(self, fe, cfg: dict) -> None:
        assert fe.gold_schema == cfg["schemas"]["gold"]

    def test_features_schema(self, fe, cfg: dict) -> None:
        assert fe.features_schema == cfg["schemas"]["features"]

    def test_gold_table_name(self, fe) -> None:
        assert fe.gold_table == "product_city_demand"

    def test_feature_table_name(self, fe) -> None:
        assert fe.feature_table == "product_city_demand_features"

    def test_gold_full_qualified(self, fe) -> None:
        assert fe.gold_full == "mle.gold.product_city_demand"

    def test_feature_full_qualified(self, fe) -> None:
        assert fe.feature_full == "mle.features.product_city_demand_features"

    def test_primary_keys(self, fe) -> None:
        assert fe.primary_keys == [
            "product_name",
            "destination_city",
            "transaction_date",
        ]

    def test_timeseries_column(self, fe) -> None:
        assert fe.timeseries_column == "transaction_date"

    def test_layer_name(self, fe) -> None:
        assert fe.layer_name == "feature_engineering"

    def test_catalog_inherited(self, fe) -> None:
        assert fe.catalog == "mle"

    def test_lag_windows_from_config(self, fe) -> None:
        assert fe.layer_cfg["features"]["lag_windows"] == [1, 7]

    def test_rolling_window_from_config(self, fe) -> None:
        assert fe.layer_cfg["features"]["rolling_window"] == 7

    def test_time_features_from_config(self, fe) -> None:
        assert "day_of_week" in fe.layer_cfg["features"]["time_features"]
        assert "is_weekend" in fe.layer_cfg["features"]["time_features"]


# --- ModelTraining ---


class TestModelTraining:
    """Test ModelTraining __init__ config parsing."""

    @pytest.fixture(scope="class")
    def mt(self) -> mlp.ModelTraining:
        return mlp.ModelTraining(config_path=CONFIG_PATH)

    def test_fe_cfg_is_fe_section(self, mt, cfg: dict) -> None:
        assert mt.fe_cfg == cfg["feature_engineering"]

    def test_lifecycle_cfg_is_model_lifecycle_section(self, mt, cfg: dict) -> None:
        assert mt.lifecycle_cfg == cfg["model_lifecycle"]

    def test_features_schema(self, mt, cfg: dict) -> None:
        assert mt.features_schema == cfg["schemas"]["features"]

    def test_feature_table_name(self, mt) -> None:
        assert mt.feature_table == "product_city_demand_features"

    def test_feature_full_qualified(self, mt) -> None:
        assert mt.feature_full == "mle.features.product_city_demand_features"

    def test_gold_schema(self, mt, cfg: dict) -> None:
        assert mt.gold_schema == cfg["schemas"]["gold"]

    def test_gold_table_name(self, mt) -> None:
        assert mt.gold_table == "product_city_demand"

    def test_gold_full_qualified(self, mt) -> None:
        assert mt.gold_full == "mle.gold.product_city_demand"

    def test_experiment_name(self, mt) -> None:
        assert mt.experiment_name == mt.lifecycle_cfg["experiment_name"]
        assert mt.experiment_name.startswith("/Users/")

    def test_model_name_qualified(self, mt) -> None:
        assert mt.model_name == "mle.models.demand_forecast_lr"

    def test_prediction_table_qualified(self, mt) -> None:
        assert mt.prediction_table == "mle.models.demand_predictions"

    def test_target_column(self, mt) -> None:
        assert mt.target == "demand_quantity"

    def test_categorical_features(self, mt) -> None:
        assert "product_name" in mt.categorical_features
        assert "destination_city" in mt.categorical_features
        assert len(mt.categorical_features) == 4

    def test_min_r2_threshold(self, mt) -> None:
        assert mt.min_r2 == mt.lifecycle_cfg["min_r2"]
        assert mt.min_r2 >= 0

    def test_max_mae_threshold(self, mt) -> None:
        assert mt.max_mae == mt.lifecycle_cfg["max_mae"]
        assert mt.max_mae > 0

    def test_max_rmse_threshold(self, mt) -> None:
        assert mt.max_rmse == mt.lifecycle_cfg["max_rmse"]
        assert mt.max_rmse > 0

    def test_layer_name(self, mt) -> None:
        assert mt.layer_name == "model_training"

    def test_catalog_inherited(self, mt) -> None:
        assert mt.catalog == "mle"
