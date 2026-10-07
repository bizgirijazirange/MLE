#!/usr/bin/env python3
"""Pipeline Orchestrator — DE + ML Entry Point

Thin orchestration script that runs the data engineering (Medallion:
Bronze → Silver → Gold) and machine learning (Feature Engineering →
Train → Register → Predict) pipelines in sequence.

All class definitions live in importable ``.py`` modules
(``meadallion_pipeline.py``, ``ml_pipeline.py``) in the same directory.
This script imports them, calls their ``.run()`` methods directly, and
manages parameters, configuration, logging, and error handling.

Compatible with Declarative Automation Bundles (DAB) and Git-based CI/CD.

Usage (as a spark_python_task in a Databricks Job)::

    python main.py --config-file /path/to/config/dev.yaml

Execution order::

    1. Medallion Pipeline (Bronze → Silver → Gold)                     — data engineering
    2. ML Pipeline (Feature Engineering → Train → Register → Predict)  — machine learning
"""

import argparse
import logging
import os
import sys
import traceback

import yaml
from pyspark.sql import SparkSession

# ---------------------------------------------------------------------------
# Make sibling modules importable
# ---------------------------------------------------------------------------
# Try __file__ first (works for direct execution and spark_python_task with
# cluster compute).  On serverless spark_python_task, __file__ is not defined,
# so SRC_DIR is derived from the config file path inside main().
SRC_DIR = None
try:
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
except NameError:
    pass  # __file__ not defined — will derive from config_file in main()

# ---------------------------------------------------------------------------
# Spark session (available automatically in Databricks, but explicit is safe)
# ---------------------------------------------------------------------------
spark = SparkSession.builder.getOrCreate()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orchestrator")


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------
def run_de_pipeline(config_file: str, cfg: dict, catalog: str) -> None:
    """Stage 1: Data Engineering — Bronze → Silver → Gold."""
    logger.info("-" * 70)
    logger.info("Stage 1: Data Engineering — Bronze → Silver → Gold")
    logger.info("-" * 70)

    try:
        bronze = BronzeIngestion(config_path=config_file)
        bronze.run()

        silver = SilverIngestion(config_path=config_file)
        silver.run()

        gold = GoldIngestion(config_path=config_file)
        gold.run()

        logger.info("Stage 1 completed successfully")

        # Log record counts from the Delta tables
        bronze_table = (
            f"{catalog}.{cfg['schemas']['bronze']}"
            f".{cfg['bronze']['target']['table']}"
        )
        silver_table = (
            f"{catalog}.{cfg['schemas']['silver']}"
            f".{cfg['silver']['target']['silver_table']}"
        )
        gold_table = (
            f"{catalog}.{cfg['schemas']['gold']}"
            f".{cfg['gold']['target']['gold_table']}"
        )

        bronze_count = spark.table(bronze_table).count()
        silver_count = spark.table(silver_table).count()
        gold_count = spark.table(gold_table).count()

        logger.info(
            f"Record counts — Bronze: {bronze_count:,} "
            f"| Silver: {silver_count:,} | Gold: {gold_count:,}"
        )

    except Exception:
        logger.error("Stage 1 FAILED — Data Engineering pipeline error")
        logger.error(traceback.format_exc())
        raise


def run_ml_pipeline(config_file: str, cfg: dict, catalog: str) -> None:
    """Stage 2: Machine Learning — Feature Engineering → Train → Register → Predict."""
    logger.info("-" * 70)
    logger.info(
        "Stage 2: Machine Learning — Feature Engineering → Train → Register → Predict"
    )
    logger.info("-" * 70)

    try:
        fe = FeatureEngineering(config_path=config_file)
        fe.run()

        trainer = ModelTraining(config_path=config_file)
        trainer.run()

        logger.info("Stage 2 completed successfully")

        # Log record counts from the Feature Store and predictions tables
        feature_table = (
            f"{catalog}.{cfg['schemas']['features']}"
            f".{cfg['feature_engineering']['target']['feature_table']}"
        )
        prediction_table = f"{catalog}.models.demand_predictions"

        feature_count = spark.table(feature_table).count()
        prediction_count = spark.table(prediction_table).count()

        logger.info(
            f"Record counts — Features: {feature_count:,} "
            f"| Predictions: {prediction_count:,}"
        )

    except Exception:
        logger.error("Stage 2 FAILED — ML pipeline error")
        logger.error(traceback.format_exc())
        raise


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    # Read config file path: try widget first (notebook_task on serverless),
    # then fall back to argparse (spark_python_task or local CLI).
    config_file = None
    try:
        from databricks.sdk.runtime import dbutils
        config_file = dbutils.widgets.get("config_file")
    except Exception:  # noqa: S110, BLE001
        pass

    if not config_file:
        parser = argparse.ArgumentParser(description="MLE Pipeline Orchestrator")
        parser.add_argument(
            "--config-file",
            required=False,
            default=None,
            help="Path to the environment config YAML file",
        )
        args, _ = parser.parse_known_args()
        config_file = args.config_file

    # Derive default config path if not provided via widget or CLI argument
    if not config_file:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()
        config_file = os.path.join(os.path.dirname(script_dir), "config", "dev.yaml")

    # --- Ensure sibling modules are importable (deferred for serverless) ---
    global BronzeIngestion, SilverIngestion, GoldIngestion
    global FeatureEngineering, ModelTraining
    if SRC_DIR is None:
        # Derive from config file: .../config/dev.yaml → .../src/
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(config_file)))
        src_dir = os.path.join(project_root, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
    from meadallion_pipeline import BronzeIngestion, GoldIngestion, SilverIngestion
    from ml_pipeline import FeatureEngineering, ModelTraining

    # --- Load and log configuration ---
    logger.info("=" * 70)
    logger.info("Pipeline Orchestrator — Starting")
    logger.info("=" * 70)
    logger.info(f"Config file:       {config_file}")

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    catalog = cfg["catalog"]
    logger.info(f"Catalog:            {catalog}")
    logger.info(f"Schemas:            {cfg['schemas']}")
    logger.info(f"Source path:        {cfg.get('source_path', 'N/A')}")
    logger.info(
        f"Bronze table:      {catalog}.{cfg['schemas']['bronze']}"
        f".{cfg['bronze']['target']['table']}"
    )
    logger.info(
        f"Silver table:      {catalog}.{cfg['schemas']['silver']}"
        f".{cfg['silver']['target']['silver_table']}"
    )
    logger.info(
        f"Gold table:        {catalog}.{cfg['schemas']['gold']}"
        f".{cfg['gold']['target']['gold_table']}"
    )
    logger.info(
        f"Feature table:     {catalog}.{cfg['schemas']['features']}"
        f".{cfg['feature_engineering']['target']['feature_table']}"
    )
    logger.info(f"Prediction table:  {catalog}.models.demand_predictions")
    logger.info("Configuration loaded and validated successfully")

    # --- Run both stages ---
    run_de_pipeline(config_file, cfg, catalog)
    run_ml_pipeline(config_file, cfg, catalog)

    # --- Completion summary ---
    logger.info("=" * 70)
    logger.info("Pipeline Orchestrator — All stages completed successfully")
    logger.info("  Stage 1: DE  — Bronze → Silver → Gold")
    logger.info("  Stage 2: ML  — Feature Engineering → Train → Register → Predict")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()