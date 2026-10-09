"""Shared pytest configuration for MLE pipeline unit tests.

Mocks Spark, Databricks SDK, MLflow, sklearn, and xgboost so the pipeline
source modules can be imported in CI without a Spark cluster.
"""

import os
import sys
from unittest.mock import MagicMock

# --- Add src/ to Python path so test files can import pipeline modules ---
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC_DIR)

# --- Mock Databricks runtime (dbutils) ---
_dbutils_mock = MagicMock()
sys.modules["databricks"] = MagicMock()
sys.modules["databricks.sdk"] = MagicMock()
sys.modules["databricks.sdk.runtime"] = _dbutils_mock
sys.modules["databricks.feature_engineering"] = MagicMock()

# --- Mock PySpark (no Spark cluster in CI) ---
_pyspark_sql_mock = MagicMock()
sys.modules["pyspark"] = MagicMock()
sys.modules["pyspark.sql"] = _pyspark_sql_mock
sys.modules["pyspark.sql.functions"] = MagicMock()
sys.modules["pyspark.sql.window"] = MagicMock()

# --- Mock MLflow ---
sys.modules["mlflow"] = MagicMock()
sys.modules["mlflow.sklearn"] = MagicMock()
sys.modules["mlflow.models"] = MagicMock()
sys.modules["mlflow.tracking"] = MagicMock()
sys.modules["mlflow.pyfunc"] = MagicMock()

# --- Mock sklearn sub-modules and xgboost (heavy deps not needed for config tests) ---
for _mod in [
    "sklearn.compose",
    "sklearn.impute",
    "sklearn.metrics",
    "sklearn.pipeline",
    "sklearn.preprocessing",
    "xgboost",
]:
    sys.modules[_mod] = MagicMock()

# --- Shared constants ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "dev.yaml")
