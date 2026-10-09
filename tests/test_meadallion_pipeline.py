"""Unit tests for medallion_pipeline.py.

Tests the _get_widget_value utility and the __init__ config-parsing logic of
BaseLayer, BronzeIngestion, SilverIngestion, and GoldIngestion without a
live Spark cluster (conftest.py mocks Spark/Databricks dependencies).
"""

import logging
import os
from unittest.mock import patch

import pytest
import yaml

import medallion_pipeline as mp

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "dev.yaml")


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# --- _get_widget_value ---


class TestGetWidgetValue:
    """Test the _get_widget_value utility function."""

    @patch.object(mp, "dbutils")
    def test_returns_widget_value_when_available(self, mock_dbutils) -> None:
        mock_dbutils.widgets.get.return_value = "/path/to/config.yaml"
        assert mp._get_widget_value("config_file") == "/path/to/config.yaml"

    @patch.object(mp, "dbutils")
    def test_returns_default_when_widget_missing(self, mock_dbutils) -> None:
        mock_dbutils.widgets.get.side_effect = Exception("No such widget")
        assert mp._get_widget_value("config_file", "fallback.yaml") == "fallback.yaml"

    @patch.object(mp, "dbutils")
    def test_returns_empty_string_default(self, mock_dbutils) -> None:
        mock_dbutils.widgets.get.side_effect = Exception("No such widget")
        assert mp._get_widget_value("missing_widget") == ""

    @patch.object(mp, "dbutils")
    def test_calls_widgets_get_with_correct_name(self, mock_dbutils) -> None:
        mock_dbutils.widgets.get.return_value = "value"
        mp._get_widget_value("source_path")
        mock_dbutils.widgets.get.assert_called_once_with("source_path")


# --- BaseLayer ---


class TestBaseLayer:
    """Test the BaseLayer base class config loading and setup."""

    def test_config_path_stored(self) -> None:
        layer = mp.BaseLayer(CONFIG_PATH, "test_layer")
        assert layer.config_path == CONFIG_PATH

    def test_layer_name_stored(self) -> None:
        layer = mp.BaseLayer(CONFIG_PATH, "bronze_ingestion")
        assert layer.layer_name == "bronze_ingestion"

    def test_config_loaded(self, cfg: dict) -> None:
        layer = mp.BaseLayer(CONFIG_PATH, "test_layer")
        assert layer.cfg["catalog"] == cfg["catalog"]
        assert layer.cfg["schemas"] == cfg["schemas"]

    def test_catalog_extracted(self) -> None:
        layer = mp.BaseLayer(CONFIG_PATH, "test_layer")
        assert layer.catalog == "mle"

    def test_logger_is_logging_logger(self) -> None:
        layer = mp.BaseLayer(CONFIG_PATH, "test_layer")
        assert isinstance(layer.logger, logging.Logger)
        assert layer.logger.name == "test_layer"

    def test_invalid_config_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            mp.BaseLayer("/nonexistent/path.yaml", "test_layer")


# --- BronzeIngestion ---


class TestBronzeIngestion:
    """Test BronzeIngestion __init__ config parsing."""

    @pytest.fixture(scope="class")
    def bronze(self) -> mp.BronzeIngestion:
        with patch.object(mp, "dbutils") as mock_dbutils:
            mock_dbutils.widgets.get.return_value = ""
            return mp.BronzeIngestion(config_path=CONFIG_PATH)

    def test_layer_cfg_is_bronze_section(self, bronze, cfg: dict) -> None:
        assert bronze.layer_cfg == cfg["bronze"]

    def test_schema_name(self, bronze, cfg: dict) -> None:
        assert bronze.schema == cfg["schemas"]["bronze"]

    def test_source_path(self, bronze, cfg: dict) -> None:
        assert bronze.source_path == cfg["source_path"]

    def test_table_name(self, bronze) -> None:
        assert bronze.table_name == "supply_chain_dataset_bronze"

    def test_full_table_name(self, bronze) -> None:
        assert bronze.full_table == "mle.bronze.supply_chain_dataset_bronze"

    def test_layer_name(self, bronze) -> None:
        assert bronze.layer_name == "bronze_ingestion"

    def test_catalog_inherited(self, bronze) -> None:
        assert bronze.catalog == "mle"


# --- SilverIngestion ---


class TestSilverIngestion:
    """Test SilverIngestion __init__ config parsing."""

    @pytest.fixture(scope="class")
    def silver(self) -> mp.SilverIngestion:
        with patch.object(mp, "dbutils") as mock_dbutils:
            mock_dbutils.widgets.get.return_value = ""
            return mp.SilverIngestion(config_path=CONFIG_PATH)

    def test_layer_cfg_is_silver_section(self, silver, cfg: dict) -> None:
        assert silver.layer_cfg == cfg["silver"]

    def test_bronze_schema(self, silver, cfg: dict) -> None:
        assert silver.bronze_schema == cfg["schemas"]["bronze"]

    def test_silver_schema(self, silver, cfg: dict) -> None:
        assert silver.silver_schema == cfg["schemas"]["silver"]

    def test_bronze_table_name(self, silver) -> None:
        assert silver.bronze_table == "supply_chain_dataset_bronze"

    def test_silver_table_name(self, silver) -> None:
        assert silver.silver_table == "supply_chain_dataset_silver"

    def test_quarantine_table_name(self, silver) -> None:
        assert silver.quarantine_table == "supply_chain_dataset_quarantine"

    def test_bronze_full_qualified(self, silver) -> None:
        assert silver.bronze_full == "mle.bronze.supply_chain_dataset_bronze"

    def test_silver_full_qualified(self, silver) -> None:
        assert silver.silver_full == "mle.silver.supply_chain_dataset_silver"

    def test_quarantine_full_qualified(self, silver) -> None:
        assert silver.quarantine_full == "mle.silver.supply_chain_dataset_quarantine"

    def test_dedup_key_column(self, silver) -> None:
        assert silver.layer_cfg["dedup"]["key_column"] == "transaction_id"

    def test_dedup_order_column(self, silver) -> None:
        assert silver.layer_cfg["dedup"]["order_column"] == "bronze_ingestion_timestamp"

    def test_dq_critical_columns(self, silver) -> None:
        assert "transaction_id" in silver.layer_cfg["data_quality"]["critical_columns"]

    def test_layer_name(self, silver) -> None:
        assert silver.layer_name == "silver_ingestion"


# --- GoldIngestion ---


class TestGoldIngestion:
    """Test GoldIngestion __init__ config parsing."""

    @pytest.fixture(scope="class")
    def gold(self) -> mp.GoldIngestion:
        with patch.object(mp, "dbutils") as mock_dbutils:
            mock_dbutils.widgets.get.return_value = ""
            return mp.GoldIngestion(config_path=CONFIG_PATH)

    def test_layer_cfg_is_gold_section(self, gold, cfg: dict) -> None:
        assert gold.layer_cfg == cfg["gold"]

    def test_silver_schema(self, gold, cfg: dict) -> None:
        assert gold.silver_schema == cfg["schemas"]["silver"]

    def test_gold_schema(self, gold, cfg: dict) -> None:
        assert gold.gold_schema == cfg["schemas"]["gold"]

    def test_silver_table_name(self, gold) -> None:
        assert gold.silver_table == "supply_chain_dataset_silver"

    def test_gold_table_name(self, gold) -> None:
        assert gold.gold_table == "product_city_demand"

    def test_silver_full_qualified(self, gold) -> None:
        assert gold.silver_full == "mle.silver.supply_chain_dataset_silver"

    def test_gold_full_qualified(self, gold) -> None:
        assert gold.gold_full == "mle.gold.product_city_demand"

    def test_grain_columns(self, gold) -> None:
        assert gold.grain_cols == [
            "product_name",
            "destination_city",
            "transaction_date",
        ]

    def test_layer_name(self, gold) -> None:
        assert gold.layer_name == "gold_ingestion"
