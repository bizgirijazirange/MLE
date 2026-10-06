"""ML Pipeline - Feature Engineering -> Train -> Register -> Predict.

This file builds the machine learning part of our pipeline. It takes
clean data from the Gold layer (produced by the medallion pipeline) and
turns it into a trained, registered model that can make predictions.

The ML pipeline has two main stages:

    Feature Engineering  -> Read Gold data, create ML features (lag, rolling,
                           calendar), and save them to the Feature Store.
    Model Training       -> Train a Linear Regression model, validate it,
                           register it in Unity Catalog, and make predictions.

How to use this file:
    1. Import it:    from ml_pipeline import FeatureEngineering, ModelTraining
    2. Run it:       python ml_pipeline.py  (calls main())

Classes in this file:
    BaseLayer            -> Shared setup (config, logging, schema creation).
    FeatureEngineering   -> Creates ML features from Gold data and writes
                            them to the Databricks Feature Store.
    ModelTraining        -> Trains, validates, registers, and evaluates
                            the model. Also makes production predictions.
"""

# --- Imports ---
import logging
import yaml
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col           # For referencing columns by name
from pyspark.sql.window import Window           # For lag and rolling window calculations
from databricks.sdk.runtime import dbutils
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- Create the Spark session ---
# Spark is the engine that processes our data. This line connects to it.
spark = SparkSession.builder.getOrCreate()


def _get_widget_value(name: str, default: str = "") -> str:
    """Get a Databricks widget value, returning default if it doesn't exist.

    Widgets are like command-line arguments for Databricks notebooks.
    If the widget doesn't exist (e.g., when running as a script), we
    return the default value instead of crashing.
    """
    try:
        return dbutils.widgets.get(name)
    except Exception:
        return default


class BaseLayer:
    """Base class for ML pipeline stages with shared config and logging.

    Provides config loading from a YAML file, logger setup, and schema creation
    in Unity Catalog. Subclasses implement stage-specific logic.

    Attributes:
        config_path: Path to the YAML configuration file.
        layer_name: Name of the stage (used as the logger name).
        cfg: Parsed YAML configuration dictionary.
        catalog: Unity Catalog name from config.
    """

    def __init__(self, config_path: str, layer_name: str):
        self.config_path = config_path
        self.layer_name = layer_name
        self.logger = logging.getLogger(layer_name)
        self._load_config()

    def _load_config(self) -> None:
        """Load YAML configuration from the configured path."""
        with open(self.config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.catalog = self.cfg["catalog"]
        self.logger.info(f"Config loaded from {self.config_path}")

    def ensure_schema(self, schema_name: str) -> None:
        """Create the target schema in Unity Catalog if it doesn't exist."""
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.catalog}.{schema_name}")
        self.logger.info(f"Ensured schema exists: {self.catalog}.{schema_name}")


class FeatureEngineering(BaseLayer):
    """Feature Engineering layer: transform Gold data into ML-ready features for the Feature Store.

    Reads the Gold layer table, engineers lag features, rolling window features,
    and calendar features, then writes the result to the Databricks Feature Store
    as a time series feature table.

    Attributes:
        gold_full: Fully qualified Gold source table name.
        feature_full: Fully qualified Feature Store target table name.
        primary_keys: List of primary key columns for the feature table.
        timeseries_column: Name of the timeseries column.
    """

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = _get_widget_value(
                "config_file",
                "/Workspace/Users/girija.zirange@bizmetric.com/MLE/config/dev.yaml",
            )
        super().__init__(config_path, "feature_engineering")
        self.layer_cfg = self.cfg["feature_engineering"]
        self.gold_schema = self.cfg["schemas"]["gold"]
        self.features_schema = self.cfg["schemas"]["features"]
        self.gold_table = self.layer_cfg["source"]["gold_table"]
        self.feature_table = self.layer_cfg["target"]["feature_table"]
        self.gold_full = f"{self.catalog}.{self.gold_schema}.{self.gold_table}"
        self.feature_full = f"{self.catalog}.{self.features_schema}.{self.feature_table}"
        self.primary_keys = self.layer_cfg["primary_keys"]
        self.timeseries_column = self.layer_cfg["timeseries_column"]
        self.fe = FeatureEngineeringClient()
        self.logger.info(f"Gold source:      {self.gold_full}")
        self.logger.info(f"Feature target:   {self.feature_full}")

    def read_gold(self):
        """Read the Gold Delta table (aggregated product-city-date demand)."""
        self.gold_df = spark.table(self.gold_full)
        self.gold_count = self.gold_df.count()
        self.logger.info(f"Gold rows read: {self.gold_count:,}")
        return self.gold_df

    def create_time_features(self, df):
        """Add calendar/time features derived from transaction_date."""
        self.logger.info("Adding calendar/time features...")
        df = (
            df
            .withColumn("day_of_week", F.dayofweek("transaction_date"))
            .withColumn("day_of_month", F.dayofmonth("transaction_date"))
            .withColumn("is_weekend", F.dayofweek("transaction_date").isin(1, 7).cast("int"))
            .withColumn("quarter", F.quarter("transaction_date"))
        )
        self.logger.info(f"Columns after time features: {len(df.columns)}")
        return df

    def create_lag_features(self, df):
        """Add lag features partitioned by (product_name, destination_city), ordered by transaction_date."""
        self.logger.info("Adding lag features...")
        lag_windows = self.layer_cfg["features"]["lag_windows"]
        lag_columns = self.layer_cfg["features"]["lag_columns"]
        w = Window.partitionBy("product_name", "destination_city").orderBy("transaction_date")

        for col_name in lag_columns:
            for lag_n in lag_windows:
                if col_name == "demand_quantity":
                    feat_name = f"lag_{lag_n}_demand"
                else:
                    feat_name = f"lag_{lag_n}_{col_name}"
                df = df.withColumn(feat_name, F.lag(col_name, lag_n).over(w))
                self.logger.info(f"  Created {feat_name}")

        self.logger.info(f"Columns after lag features: {len(df.columns)}")
        return df

    def create_rolling_features(self, df):
        """Add rolling window features (7-row avg/sum/max) partitioned by product-city."""
        self.logger.info("Adding rolling window features...")
        rolling_window = self.layer_cfg["features"]["rolling_window"]
        rolling_columns = self.layer_cfg["features"]["rolling_columns"]
        # Trailing window that EXCLUDES the current row to prevent data
        # leakage — demand_quantity is the prediction target, so the
        # current row's value must not appear in its own rolling features.
        w = (
            Window
            .partitionBy("product_name", "destination_city")
            .orderBy("transaction_date")
            .rowsBetween(-rolling_window, -1)
        )

        for col_name in rolling_columns:
            if col_name == "demand_quantity":
                df = (
                    df
                    .withColumn(f"rolling_{rolling_window}_avg_demand", F.avg(col_name).over(w))
                    .withColumn(f"rolling_{rolling_window}_sum_demand", F.sum(col_name).over(w))
                    .withColumn(f"rolling_{rolling_window}_max_demand", F.max(col_name).over(w))
                )
                self.logger.info(f"  Created rolling_{rolling_window}_avg/sum/max_demand")
            else:
                df = df.withColumn(f"rolling_{rolling_window}_avg_{col_name}", F.avg(col_name).over(w))
                self.logger.info(f"  Created rolling_{rolling_window}_avg_{col_name}")

        self.logger.info(f"Columns after rolling features: {len(df.columns)}")
        return df

    def assemble(self):
        """Assemble the final feature DataFrame with a deterministic column order."""
        self.logger.info("Assembling final feature DataFrame...")

        pk_cols = ["product_name", "destination_city", "transaction_date"]
        # NOTE: demand_quantity (target) is intentionally excluded — it must
        # NOT be stored in the Feature Store.  The label comes from the Gold
        # table via labels_df in ModelTraining.read_features().
        original_cols = [
            "transaction_month", "product_category", "quantity_unit",
            "avg_unit_price_usd", "avg_available_inventory",
        ]
        engineered_cols = [
            "lag_1_demand", "lag_7_demand",
            "rolling_7_avg_demand", "rolling_7_sum_demand",
            "rolling_7_max_demand",
            "day_of_week", "day_of_month", "is_weekend", "quarter",
        ]
        final_cols = pk_cols + original_cols + engineered_cols
        self.feature_df = self.feature_df.select(*final_cols)
        self.feature_count = self.feature_df.count()
        self.logger.info(f"Final feature rows:    {self.feature_count:,}")
        self.logger.info(f"Final feature columns: {len(self.feature_df.columns)}")
        return self.feature_df

    def write_feature_store(self) -> None:
        """Write the feature DataFrame to the Databricks Feature Store as a time series feature table."""
        self.logger.info("Writing to Feature Store...")
        self.ensure_schema(self.features_schema)

        table_exists = spark.catalog.tableExists(self.feature_full)
        if table_exists:
            self.logger.info("Feature table exists — dropping and recreating...")
            spark.sql(f"DROP TABLE IF EXISTS {self.feature_full}")

        self.fe.create_table(
            name=self.feature_full,
            primary_keys=self.primary_keys,
            timeseries_column=self.timeseries_column,
            df=self.feature_df,
            schema=self.feature_df.schema,
            description=(
                "Product-city demand features for forecasting: lag, rolling window, "
                "and calendar features. Time series feature table keyed by "
                "(product_name, destination_city, transaction_date)."
            ),
        )
        self.logger.info(f"Created feature table: {self.feature_full}")
        self.logger.info(f"  Rows written: {self.feature_count:,}")

    def validate(self) -> None:
        """Log feature engineering summary."""
        self.logger.info(f"Features: {self.feature_count:,} rows, {len(self.feature_df.columns)} columns → {self.feature_full}")

    def run(self) -> None:
        """Execute the full feature engineering pipeline."""
        self.read_gold()
        self.feature_df = self.gold_df
        self.feature_df = self.create_time_features(self.feature_df)
        self.feature_df = self.create_lag_features(self.feature_df)
        self.feature_df = self.create_rolling_features(self.feature_df)
        self.assemble()
        self.write_feature_store()
        self.validate()


class ModelTraining(BaseLayer):
    """Model lifecycle: train -> validate -> register -> promote -> evaluate.

    Implements a full model lifecycle with configurable validation gates:

    1. Train candidate model and log to MLflow
    2. Evaluate candidate on validation data
    3. Track MAE, RMSE, and R-squared
    4. Technical validation (artifact, signature, NaN/inf, schema, non-negative)
    5. Data-quality validation (feature columns, nulls, no leakage)
    6. Register validated model with Challenger alias
    7. Compare Challenger vs Champion on held-out validation set
    8. Promote Challenger to Champion if criteria met
    9. Final test evaluation of Champion

    Uses MLflow aliases: Challenger, Champion.
    Does NOT use deprecated MLflow stages (Production, Staging).
    Test set is only evaluated once, after the final Champion is selected.
    """

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = _get_widget_value(
                "config_file",
                "/Workspace/Users/girija.zirange@bizmetric.com/MLE/config/dev.yaml",
            )
        super().__init__(config_path, "model_training")
        self.fe_cfg = self.cfg["feature_engineering"]
        self.lifecycle_cfg = self.cfg["model_lifecycle"]
        self.features_schema = self.cfg["schemas"]["features"]
        self.feature_table = self.fe_cfg["target"]["feature_table"]
        self.feature_full = f"{self.catalog}.{self.features_schema}.{self.feature_table}"
        self.gold_schema = self.cfg["schemas"]["gold"]
        self.gold_table = self.fe_cfg["source"]["gold_table"]
        self.gold_full = f"{self.catalog}.{self.gold_schema}.{self.gold_table}"
        self.experiment_name = self.lifecycle_cfg["experiment_name"]
        self.model_name = f"{self.catalog}.models.{self.lifecycle_cfg['model_name']}"
        self.prediction_table = f"{self.catalog}.models.{self.lifecycle_cfg['prediction_table']}"
        self.target = "demand_quantity"
        self.categorical_features = ["product_name", "destination_city", "product_category", "quantity_unit"]
        self.mlflow_client = MlflowClient()

        # Configurable thresholds (project-specific -- calibrate from validation results)
        self.min_r2 = self.lifecycle_cfg["min_r2"]
        self.max_mae = self.lifecycle_cfg["max_mae"]
        self.max_rmse = self.lifecycle_cfg["max_rmse"]

        self.logger.info(f"Feature table:    {self.feature_full}")
        self.logger.info(f"Experiment:       {self.experiment_name}")
        self.logger.info(f"Model name:       {self.model_name}")
        self.logger.info(f"Prediction table: {self.prediction_table}")
        self.logger.info(f"Thresholds -- min_r2: {self.min_r2}, max_mae: {self.max_mae}, max_rmse: {self.max_rmse}")

    def read_features(self):
        """Read features from the Feature Store and join them with labels.

        This uses the Feature Store's create_training_set API, which does a
        point-in-time join between the Gold labels (demand_quantity) and the
        features we stored earlier. This prevents data leakage.
        """
        fe = FeatureEngineeringClient()

        labels_df = spark.table(self.gold_full).select(
            "product_name", "destination_city",
            col("transaction_date").cast("date").alias("transaction_date"),
            "demand_quantity"
        )

        all_columns = spark.table(self.feature_full).columns
        pk_cols = {"product_name", "destination_city", "transaction_date"}
        feature_names = [c for c in all_columns if c not in pk_cols and c != self.target]

        training_set = fe.create_training_set(
            df=labels_df,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_full,
                    feature_names=feature_names,
                    lookup_key=["product_name", "destination_city"],
                    timestamp_lookup_key="transaction_date",
                )
            ],
            label="demand_quantity",
        )

        self.training_set = training_set
        self.df = training_set.load_df().toPandas()
        self.logger.info(f"Features: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")

        if "is_weekend" in self.df.columns:
            self.df["is_weekend"] = self.df["is_weekend"].astype("int32")

        self.df["transaction_date"] = pd.to_datetime(self.df["transaction_date"])
        self.df = self.df.sort_values("transaction_date").reset_index(drop=True)

        exclude = ["transaction_date", self.target]
        self.numerical_features = [
            c for c in self.df.columns
            if c not in exclude + self.categorical_features
        ]
        self.feature_cols = self.categorical_features + self.numerical_features
        # transaction_date is included as the Feature Store timestamp key for
        # fe.score_batch lookups, NOT as a predictive feature.  The
        # ColumnTransformer in _train_candidate does not include it in any
        # transformer, so remainder='drop' discards it before modelling.
        self.input_cols = ["transaction_date"] + self.feature_cols
        self.logger.info(f"Target: {self.target} | Features: {len(self.feature_cols)} ({len(self.categorical_features)} cat, {len(self.numerical_features)} num)")

    def split_data(self):
        """Split data chronologically (oldest data first).

        We split by time, not randomly, to avoid training on future data.
        - Train: first 64% of rows (oldest dates)
        - Validation: next 16% of rows
        - Test: last 20% of rows (newest dates)
        """
        n = len(self.df)
        idx_test_start = int(n * 0.80)
        idx_val_start = int(idx_test_start * 0.80)

        self.train_df = self.df.iloc[:idx_val_start].copy()
        self.val_df = self.df.iloc[idx_val_start:idx_test_start].copy()
        self.test_df = self.df.iloc[idx_test_start:].copy()

        self.X_train = self.train_df[self.input_cols].copy()
        self.y_train = self.train_df[self.target].copy()
        self.X_val = self.val_df[self.input_cols].copy()
        self.y_val = self.val_df[self.target].copy()
        self.X_test = self.test_df[self.input_cols].copy()
        self.y_test = self.test_df[self.target].copy()

        self.logger.info(f"Train: {len(self.X_train):,} | Val: {len(self.X_val):,} | Test: {len(self.X_test):,}")
        self.logger.info(f"Train dates: {self.train_df['transaction_date'].min()} -> {self.train_df['transaction_date'].max()}")
        self.logger.info(f"Val dates:   {self.val_df['transaction_date'].min()} -> {self.val_df['transaction_date'].max()}")
        self.logger.info(f"Test dates:  {self.test_df['transaction_date'].min()} -> {self.test_df['transaction_date'].max()}")

    def _compute_metrics(self, y_true, y_pred):
        """Compute MAE, RMSE, and R-squared.

        Returns a dictionary with three metrics:
        - mae: Mean Absolute Error (lower is better)
        - rmse: Root Mean Squared Error (lower is better)
        - r2: R-squared score (higher is better, 1.0 is perfect)
        """
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
        }

    def _train_candidate(self):
        """Train the XGBRegressor model and log it to MLflow.

        The model uses a preprocessing pipeline that:
        - Fills missing numeric values with the mean and scales them
        - Fills missing categorical values with the most frequent and one-hot encodes them
        Then trains an XGBoost regressor on the processed features.
        """
        self.logger.info("-" * 50)
        self.logger.info("Training Candidate: XGBRegressor")
        self.logger.info("-" * 50)

        # Build a preprocessing pipeline:
        # - Numeric columns: fill missing values with the mean, then scale
        # - Categorical columns: fill missing with most frequent, then one-hot encode
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]), self.numerical_features),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]), self.categorical_features),
            ]
        )

        # Full pipeline: preprocess the data, then train an XGBoost model
        self.model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
            )),
        ])

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name="xgboost_candidate") as run:
            self.candidate_run_id = run.info.run_id

            self.model.fit(self.X_train, self.y_train)

            val_pred = self.model.predict(self.X_val)
            self.candidate_val_metrics = self._compute_metrics(self.y_val.values, val_pred)

            mlflow.log_param("model_type", "XGBRegressor")
            mlflow.log_param("split_strategy", "chronological")
            mlflow.log_param("train_size", len(self.X_train))
            mlflow.log_param("val_size", len(self.X_val))
            mlflow.log_param("test_size", len(self.X_test))
            mlflow.log_param("n_features", len(self.feature_cols))

            mlflow.log_metric("val_mae", self.candidate_val_metrics["mae"])
            mlflow.log_metric("val_rmse", self.candidate_val_metrics["rmse"])
            mlflow.log_metric("val_r2", self.candidate_val_metrics["r2"])

            mlflow.set_tag("model_type", "XGBRegressor")
            mlflow.set_tag("framework", "sklearn")
            mlflow.set_tag("split_strategy", "chronological")
            mlflow.set_tag("feature_table", self.feature_full)
            mlflow.set_tag("target", self.target)
            mlflow.set_tag("stage", "candidate")

            # Create a model signature so MLflow knows what inputs the model expects
            signature = infer_signature(
                self.X_train.head(100),
                self.model.predict(self.X_train.head(100)),
            )

            fe = FeatureEngineeringClient()
            model_info = fe.log_model(
                model=self.model,
                artifact_path="model",
                flavor=mlflow.sklearn,
                training_set=self.training_set,
                signature=signature,
            )

            self.model_uri = model_info.model_uri

            self.logger.info(f"Candidate metrics -- MAE: {self.candidate_val_metrics['mae']:.2f}, "
                             f"RMSE: {self.candidate_val_metrics['rmse']:.2f}, R2: {self.candidate_val_metrics['r2']:.4f}")
            self.logger.info(f"Candidate MLflow run: {self.candidate_run_id}")

    def _technical_validation(self):
        """Run 5 technical checks on the candidate model.

        Gates:
        1. Model artifact can be loaded from MLflow
        2. Model signature exists
        3. Predictions contain no NaN or infinite values
        4. Prediction values are all numeric
        5. Predictions are not negative (warning only, does not block)
        """
        self.logger.info("-" * 50)
        self.logger.info("Technical Validation Gates")
        self.logger.info("-" * 50)

        all_passed = True
        results = {}

        # Gate 1: Model artifact can be loaded
        try:
            loaded_model = mlflow.pyfunc.load_model(self.model_uri)
            results["artifact_loadable"] = True
            self.logger.info("  PASS -- Model artifact loaded successfully")
        except Exception as e:
            results["artifact_loadable"] = False
            self.logger.error(f"  FAIL -- Cannot load model artifact: {e}")
            all_passed = False

        # Gate 2: Model signature exists
        try:
            from mlflow.models import Model
            model_dir = mlflow.artifacts.download_artifacts(
                run_id=self.candidate_run_id, artifact_path="model"
            )
            mlmodel = Model.load(model_dir)
            has_signature = mlmodel.signature is not None
            results["signature_exists"] = has_signature
            if has_signature:
                self.logger.info("  PASS -- Model signature exists")
            else:
                self.logger.error("  FAIL -- Model signature missing")
                all_passed = False
        except Exception as e:
            results["signature_exists"] = False
            self.logger.error(f"  FAIL -- Cannot verify model signature: {e}")
            all_passed = False

        # Gate 3: Predictions contain no NaN or infinite values
        val_pred = self.model.predict(self.X_val)
        has_nan = bool(np.isnan(val_pred).any())
        has_inf = bool(np.isinf(val_pred).any())
        results["no_nan_inf"] = not (has_nan or has_inf)
        if results["no_nan_inf"]:
            self.logger.info("  PASS -- No NaN or infinite predictions")
        else:
            self.logger.error(f"  FAIL -- Predictions contain NaN: {has_nan}, Inf: {has_inf}")
            all_passed = False

        # Gate 4: Prediction schema is correct (all numeric)
        is_numeric = all(isinstance(p, (int, float, np.floating, np.integer)) for p in val_pred[:100])
        results["schema_correct"] = bool(is_numeric)
        if results["schema_correct"]:
            self.logger.info("  PASS -- Prediction schema is correct (numeric)")
        else:
            self.logger.error("  FAIL -- Prediction schema incorrect (non-numeric values)")
            all_passed = False

        # Gate 5: Predictions are not negative
        has_negative = bool((val_pred < 0).any())
        if has_negative:
            neg_count = int((val_pred < 0).sum())
            self.logger.warning(f"  WARN -- {neg_count} negative predictions found (clipped to 0)")
            results["non_negative"] = True
        else:
            results["non_negative"] = True
            self.logger.info("  PASS -- No negative predictions")

        self.technical_validation_results = results
        self.logger.info(f"Technical validation: {'ALL PASSED' if all_passed else 'FAILED'}")
        return all_passed

    def _data_quality_validation(self):
        """Run data-quality checks on the training data.

        Gates:
        1. All required feature columns are present
        2. No unexpected null values (lag/rolling nulls are expected)
        3. No date overlap between train and validation sets (no leakage)
        4. No date overlap between validation and test sets
        """
        self.logger.info("-" * 50)
        self.logger.info("Data-Quality Validation Gates")
        self.logger.info("-" * 50)

        all_passed = True
        results = {}

        # Gate 1: Required feature columns exist
        required_features = set(self.feature_cols)
        train_cols = set(self.X_train.columns)
        missing_cols = required_features - train_cols
        results["required_columns_exist"] = len(missing_cols) == 0
        if results["required_columns_exist"]:
            self.logger.info(f"  PASS -- All {len(required_features)} required feature columns present")
        else:
            self.logger.error(f"  FAIL -- Missing columns: {missing_cols}")
            all_passed = False

        # Gate 2: No unexpected nulls in features
        null_counts = self.X_train.isnull().sum()
        expected_null_columns = {c for c in self.numerical_features
                                 if "lag_" in c or "rolling_" in c}
        unexpected_nulls = {c: int(null_counts[c]) for c in null_counts.index
                            if null_counts[c] > 0 and c not in expected_null_columns}
        results["no_unexpected_nulls"] = len(unexpected_nulls) == 0
        if results["no_unexpected_nulls"]:
            self.logger.info("  PASS -- No unexpected nulls in features")
        else:
            self.logger.error(f"  FAIL -- Unexpected nulls: {unexpected_nulls}")
            all_passed = False

        # Gate 3: Validation/test data is unseen during training (no date overlap)
        train_max = self.train_df["transaction_date"].max()
        val_min = self.val_df["transaction_date"].min()
        test_min = self.test_df["transaction_date"].min()
        no_overlap = (val_min >= train_max) and (test_min >= train_max)
        results["no_data_leakage"] = bool(no_overlap)
        if no_overlap:
            self.logger.info(f"  PASS -- No date overlap (train ends {train_max}, val starts {val_min})")
        else:
            self.logger.error(f"  FAIL -- Date overlap (train ends {train_max}, val starts {val_min})")
            all_passed = False

        # Additional: val and test don't overlap
        val_max = self.val_df["transaction_date"].max()
        no_val_test = test_min >= val_max
        results["no_val_test_overlap"] = bool(no_val_test)
        if no_val_test:
            self.logger.info(f"  PASS -- Val and test don't overlap (val ends {val_max}, test starts {test_min})")
        else:
            self.logger.error(f"  FAIL -- Val/test date overlap (val ends {val_max}, test starts {test_min})")
            all_passed = False

        self.dq_validation_results = results
        self.logger.info(f"Data-quality validation: {'ALL PASSED' if all_passed else 'FAILED'}")
        return all_passed

    def _register_and_challenger(self):
        """Register the validated model in Unity Catalog and assign the Challenger alias.

        This only runs AFTER all validation gates have passed.
        The model gets a new version number and the 'Challenger' alias.
        """
        self.logger.info("-" * 50)
        self.logger.info("Registering Validated Model (Challenger)")
        self.logger.info("-" * 50)

        self.ensure_schema("models")

        mv = mlflow.register_model(
            model_uri=self.model_uri,
            name=self.model_name,
        )
        self.candidate_version = mv.version
        self.logger.info(f"Registered: {self.model_name} (version {self.candidate_version})")

        self.mlflow_client.set_registered_model_alias(
            self.model_name, "Challenger", self.candidate_version
        )
        self.logger.info(f"Assigned 'Challenger' alias to version {self.candidate_version}")

        tags = {"domain": "supply_chain", "framework": "sklearn", "stage": "challenger"}
        for k, v in tags.items():
            self.mlflow_client.set_model_version_tag(self.model_name, self.candidate_version, k, v)

        with mlflow.start_run(run_id=self.candidate_run_id):
            mlflow.set_tag("registered_version", str(self.candidate_version))
            mlflow.set_tag("alias", "Challenger")
            mlflow.log_param("technical_validation", str(self.technical_validation_results))
            mlflow.log_param("dq_validation", str(self.dq_validation_results))

    def _get_champion_version(self):
        """Get the current Champion model version, or None if no Champion exists."""
        try:
            champion_mv = self.mlflow_client.get_model_version_by_alias(
                self.model_name, "Champion"
            )
            return champion_mv.version
        except Exception:
            return None

    def _evaluate_model_on_validation(self, model_uri):
        """Evaluate a registered model on the validation set via fe.score_batch.

        Uses the Feature Store score_batch API with the actual model_uri so
        that the model is fetched from the MLflow registry on each executor.
        This is the serverless-compatible path: only entity columns (primary
        keys + timestamp key) are passed; score_batch looks up features from
        the Feature Store and scores the model in a single call.
        """
        fe = FeatureEngineeringClient()
        entity_cols = ["product_name", "destination_city", "transaction_date"]
        entity_df = spark.createDataFrame(self.val_df[entity_cols])
        entity_df = entity_df.withColumn("transaction_date", col("transaction_date").cast("date"))

        self.logger.info(f"score_batch validation scoring -- model_uri: {model_uri}")
        scored_df = fe.score_batch(model_uri=model_uri, df=entity_df)
        scored_pd = scored_df.toPandas()

        # Merge predictions with actuals to guarantee row alignment
        result = scored_pd.merge(
            self.val_df[entity_cols + ["demand_quantity"]],
            on=entity_cols,
            how="inner",
        )
        val_pred = np.maximum(result["prediction"].values, 0)
        metrics = self._compute_metrics(result["demand_quantity"].values, val_pred)
        self.logger.info(
            f"  Validation metrics -- MAE: {metrics['mae']:.2f}, "
            f"RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.4f}"
        )
        return metrics

    def _compare_and_promote(self):
        """Compare the Challenger model against the current Champion.

        If there is no Champion yet, the candidate becomes Champion if it
        passes the absolute thresholds (min_r2, max_mae, max_rmse).

        If a Champion already exists, the Challenger must beat the Champion
        on at least 2 of 3 metrics to be promoted.
        The test set is NOT used here -- only the validation set.
        """
        self.logger.info("-" * 50)
        self.logger.info("Challenger vs Champion Comparison")
        self.logger.info("-" * 50)

        champion_version = self._get_champion_version()
        candidate_uri = f"models:/{self.model_name}/{self.candidate_version}"
        self.logger.info(f"Candidate model URI: {candidate_uri}")

        if champion_version is None:
            self.logger.info("No existing Champion -- evaluating candidate for direct promotion")
            self.logger.info(f"Candidate MAE: {self.candidate_val_metrics['mae']:.2f}")

            passes = (
                self.candidate_val_metrics["r2"] >= self.min_r2 and
                self.candidate_val_metrics["mae"] <= self.max_mae and
                self.candidate_val_metrics["rmse"] <= self.max_rmse
            )

            if passes:
                self.logger.info(f"  PASS -- Candidate meets absolute thresholds")
                self.logger.info(f"  Required: R2 >= {self.min_r2}, MAE <= {self.max_mae}, RMSE <= {self.max_rmse}")
                self.logger.info(f"  Got:      R2={self.candidate_val_metrics['r2']:.4f}, "
                                 f"MAE={self.candidate_val_metrics['mae']:.2f}, "
                                 f"RMSE={self.candidate_val_metrics['rmse']:.2f}")

                self.mlflow_client.set_registered_model_alias(
                    self.model_name, "Champion", self.candidate_version
                )
                self.mlflow_client.delete_registered_model_alias(
                    self.model_name, "Challenger"
                )
                self.logger.info(f"  Promoted version {self.candidate_version} to Champion")
                self.logger.info(f"  Removed 'Challenger' alias from version {self.candidate_version}")

                with mlflow.start_run(run_id=self.candidate_run_id):
                    mlflow.set_tag("promotion_decision", "Champion (no prior Champion)")
                    mlflow.set_tag("alias", "Champion")
                    mlflow.log_param("promotion_reason",
                                     "No prior Champion; candidate passed all gates and absolute thresholds")

                self.champion_version = self.candidate_version
                self.promotion_decision = "promoted_new_champion"
                self.logger.info(f"Promotion decision: PROMOTED (v{self.candidate_version} -> Champion, no prior Champion)")
            else:
                self.logger.error(f"  FAIL -- Candidate does not meet absolute thresholds")
                self.logger.error(f"  Required: R2 >= {self.min_r2}, MAE <= {self.max_mae}, RMSE <= {self.max_rmse}")
                self.logger.error(f"  Got:      R2={self.candidate_val_metrics['r2']:.4f}, "
                                  f"MAE={self.candidate_val_metrics['mae']:.2f}, "
                                  f"RMSE={self.candidate_val_metrics['rmse']:.2f}")

                with mlflow.start_run(run_id=self.candidate_run_id):
                    mlflow.set_tag("promotion_decision", "REJECTED -- failed absolute thresholds")
                    mlflow.log_param("promotion_reason", "Failed absolute thresholds for new Champion")

                self.champion_version = None
                self.promotion_decision = "rejected"
                self.logger.info(f"Promotion decision: REJECTED (candidate failed absolute thresholds)")
            return

        self.logger.info(f"Existing Champion: version {champion_version}")

        champion_uri = f"models:/{self.model_name}/{champion_version}"
        candidate_uri = f"models:/{self.model_name}/{self.candidate_version}"
        self.logger.info(f"Champion model URI:  {champion_uri}")
        self.logger.info(f"Candidate model URI: {candidate_uri}")

        # Evaluate the Champion on the validation set.  If this fails the
        # pipeline MUST fail — we never silently promote the candidate when
        # the Champion cannot be evaluated.
        champion_metrics = self._evaluate_model_on_validation(champion_uri)
        self.logger.info(f"Champion MAE:  {champion_metrics['mae']:.2f}")
        self.logger.info(f"Candidate MAE: {self.candidate_val_metrics['mae']:.2f}")

        cand = self.candidate_val_metrics
        champ = champion_metrics

        self.logger.info(f"  Champion (v{champion_version})  -- MAE: {champ['mae']:.2f}, "
                         f"RMSE: {champ['rmse']:.2f}, R2: {champ['r2']:.4f}")
        self.logger.info(f"  Challenger (v{self.candidate_version}) -- MAE: {cand['mae']:.2f}, "
                         f"RMSE: {cand['rmse']:.2f}, R2: {cand['r2']:.4f}")

        # Check each metric: is the challenger better than the champion?
        # R2: higher is better. MAE and RMSE: lower is better.
        wins = 0
        if cand["r2"] > champ["r2"]:
            wins += 1
        if cand["mae"] < champ["mae"]:
            wins += 1
        if cand["rmse"] < champ["rmse"]:
            wins += 1

        self.logger.info(f"  Challenger beats Champion on {wins}/3 metrics (need >=2)")

        if wins >= 2:
            self.logger.info("  PASS -- Challenger outperforms Champion on >=2/3 metrics")

            self.mlflow_client.set_registered_model_alias(
                self.model_name, "Champion", self.candidate_version
            )
            self.mlflow_client.delete_registered_model_alias(
                self.model_name, "Challenger"
            )
            self.logger.info(f"  Promoted Challenger (v{self.candidate_version}) to Champion")
            self.logger.info(f"  Removed 'Challenger' alias from version {self.candidate_version}")

            with mlflow.start_run(run_id=self.candidate_run_id):
                mlflow.set_tag("promotion_decision", "Promoted to Champion")
                mlflow.set_tag("alias", "Champion")
                mlflow.log_param("promotion_reason",
                                 f"Outperformed Champion v{champion_version} on {wins}/3 metrics")
                mlflow.log_metric("champion_val_mae", champ["mae"])
                mlflow.log_metric("champion_val_rmse", champ["rmse"])
                mlflow.log_metric("champion_val_r2", champ["r2"])

            self.champion_version = self.candidate_version
            self.promotion_decision = "promoted"
            self.logger.info(f"Promotion decision: PROMOTED (v{self.candidate_version} -> Champion)")
        else:
            self.logger.info("  FAIL -- Challenger does not outperform Champion")
            self.logger.info(f"  Keeping existing Champion (version {champion_version})")

            with mlflow.start_run(run_id=self.candidate_run_id):
                mlflow.set_tag("promotion_decision", "Not promoted -- Champion retained")
                mlflow.log_param("promotion_reason",
                                 f"Only beat Champion on {wins}/3 metrics (need >=2)")
                mlflow.log_metric("champion_val_mae", champ["mae"])
                mlflow.log_metric("champion_val_rmse", champ["rmse"])
                mlflow.log_metric("champion_val_r2", champ["r2"])

            self.champion_version = champion_version
            self.promotion_decision = "champion_retained"
            self.logger.info(f"Promotion decision: CHAMPION RETAINED (v{champion_version} stays Champion)")

    def predict(self):
        """Make production predictions using the Champion model.

        Uses the Feature Store's score_batch API, which automatically looks
        up the right features for each product-city-date combination.
        Saves predictions to a Delta table in Unity Catalog.
        """
        self.logger.info("-" * 50)
        self.logger.info("Production Predictions (Champion)")
        self.logger.info("-" * 50)

        if self.champion_version is None:
            self.logger.error("  No Champion -- skipping production predictions")
            self.pred_df = None
            return

        champion_uri = f"models:/{self.model_name}/{self.champion_version}"
        self.logger.info(f"Champion model URI: {champion_uri}")

        fe = FeatureEngineeringClient()
        entity_cols = ["product_name", "destination_city", "transaction_date"]
        entity_df = spark.createDataFrame(self.test_df[entity_cols])
        entity_df = entity_df.withColumn("transaction_date", col("transaction_date").cast("date"))

        self.logger.info("Scoring production predictions with fe.score_batch ...")
        scored_df = fe.score_batch(model_uri=champion_uri, df=entity_df)
        scored_pd = scored_df.toPandas()

        # Merge predictions with actuals
        result = scored_pd.merge(
            self.test_df[entity_cols + ["demand_quantity"]],
            on=entity_cols,
            how="inner",
        )

        self.pred_df = pd.DataFrame({
            "product_name": result["product_name"],
            "destination_city": result["destination_city"],
            "transaction_date": result["transaction_date"],
            "actual_demand": result["demand_quantity"],
            "predicted_demand": np.maximum(result["prediction"].values, 0),
            "model_name": self.model_name,
            "model_version": str(self.champion_version),
            "mlflow_run_id": self.candidate_run_id,
            "alias": "Champion",
        })

        pred_spark_df = spark.createDataFrame(self.pred_df)
        self.ensure_schema("models")
        (pred_spark_df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(self.prediction_table))
        spark.sql(f"ALTER TABLE {self.prediction_table} SET TAGS ('domain' = 'supply_chain', 'type' = 'predictions', 'model' = 'demand_forecast_lr')")
        self.logger.info(f"Written {len(self.pred_df):,} predictions to {self.prediction_table}")

    def evaluate(self):
        """Final evaluation with plots logged to MLflow."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self.pred_df is None or len(self.pred_df) == 0:
            self.logger.warning("No predictions available for evaluation plots")
            return

        y_actual = self.pred_df["actual_demand"].values
        y_pred = self.pred_df["predicted_demand"].values
        residuals = y_actual - y_pred

        r2 = self.test_metrics["r2"] if hasattr(self, "test_metrics") else float(r2_score(y_actual, y_pred))
        rmse = self.test_metrics["rmse"] if hasattr(self, "test_metrics") else float(np.sqrt(mean_squared_error(y_actual, y_pred)))
        mae = self.test_metrics["mae"] if hasattr(self, "test_metrics") else float(mean_absolute_error(y_actual, y_pred))
        mape = float(np.mean(np.abs((y_actual - y_pred) / np.where(y_actual == 0, 1, y_actual))) * 100)

        with mlflow.start_run(run_id=self.candidate_run_id):
            mlflow.log_metric("test_r2", r2)
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_mape", mape)

            # 1. Actual vs Predicted
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            ax1.scatter(y_actual, y_pred, alpha=0.3, s=10)
            lims = [min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())]
            ax1.plot(lims, lims, "r--", alpha=0.7, label="Perfect prediction")
            ax1.set_xlabel("Actual Demand")
            ax1.set_ylabel("Predicted Demand")
            ax1.set_title(f"Actual vs Predicted (R2 = {r2:.4f})")
            ax1.legend()
            mlflow.log_figure(fig1, "plots/actual_vs_predicted.png")
            plt.close(fig1)

            # 2. Residuals vs Predicted
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.scatter(y_pred, residuals, alpha=0.3, s=10)
            ax2.axhline(0, color="r", linestyle="--", alpha=0.7)
            ax2.set_xlabel("Predicted Demand")
            ax2.set_ylabel("Residuals (Actual - Predicted)")
            ax2.set_title("Residuals vs Predicted")
            mlflow.log_figure(fig2, "plots/residuals_vs_predicted.png")
            plt.close(fig2)

            # 3. Residual histogram
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
            ax3.axvline(0, color="r", linestyle="--", alpha=0.7)
            ax3.set_xlabel("Residual")
            ax3.set_ylabel("Frequency")
            ax3.set_title(f"Residual Distribution (RMSE = {rmse:.2f})")
            mlflow.log_figure(fig3, "plots/residual_histogram.png")
            plt.close(fig3)

        self.logger.info(f"Test metrics -- R2: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
        self.logger.info("Evaluation plots logged as MLflow artifacts: actual_vs_predicted, residuals_vs_predicted, residual_histogram")

    def _final_test_evaluation(self):
        """Evaluate the Champion model on the held-out test set.

        The test set is only used ONCE -- after the Champion model is selected.
        This gives us an unbiased estimate of how the model will perform in
        production. We use the Feature Store's score_batch API to look up
        features automatically and make predictions on the test data.
        """
        self.logger.info("-" * 50)
        self.logger.info("Final Test Evaluation (Champion)")
        self.logger.info("-" * 50)

        if self.champion_version is None:
            self.logger.error("  No Champion -- skipping test evaluation")
            self.test_metrics = None
            return

        champion_uri = f"models:/{self.model_name}/{self.champion_version}"
        self.logger.info(f"Champion model URI: {champion_uri}")

        fe = FeatureEngineeringClient()
        entity_cols = ["product_name", "destination_city", "transaction_date"]
        entity_df = spark.createDataFrame(self.test_df[entity_cols])
        entity_df = entity_df.withColumn("transaction_date", col("transaction_date").cast("date"))

        self.logger.info("Scoring test set with fe.score_batch ...")
        scored_df = fe.score_batch(model_uri=champion_uri, df=entity_df)
        scored_pd = scored_df.toPandas()

        # Merge predictions with actuals to guarantee row alignment
        result = scored_pd.merge(
            self.test_df[entity_cols + ["demand_quantity"]],
            on=entity_cols,
            how="inner",
        )
        test_pred = np.maximum(result["prediction"].values, 0)

        self.test_metrics = self._compute_metrics(result["demand_quantity"].values, test_pred)

        self.logger.info(f"  Final test MAE: {self.test_metrics['mae']:.2f}")
        self.logger.info(f"  Test metrics -- MAE: {self.test_metrics['mae']:.2f}, "
                         f"RMSE: {self.test_metrics['rmse']:.2f}, R2: {self.test_metrics['r2']:.4f}")

    def run(self) -> None:
        """Run the full model lifecycle step by step.

        Steps:
        1. Read features from Feature Store
        2. Split data into train / validation / test sets
        3. Train the candidate (XGBRegressor) model
        4. Run validation gates (technical + data-quality)
        5. If all gates pass: register as Challenger
        6. Compare Challenger vs Champion and promote if better
        7. Evaluate Champion on the held-out test set
        8. Make production predictions and log evaluation plots
        """
        self.read_features()
        self.split_data()
        self._train_candidate()

        # --- Validation gates ---
        tech_passed = self._technical_validation()
        dq_passed = self._data_quality_validation()

        if not (tech_passed and dq_passed):
            self.logger.error("Candidate FAILED validation gates -- not registering")
            return

        # --- Register and promote ---
        self._register_and_challenger()
        self._compare_and_promote()

        # --- Final test evaluation (only after Champion is selected) ---
        self._final_test_evaluation()

        # --- Production predictions and evaluation ---
        self.predict()
        self.evaluate()


# ============================================================================
# Main function -- run this file as a script
# ============================================================================
def main():
    """Run the full ML pipeline: Feature Engineering -> Train -> Register -> Predict.

    This function creates and runs each stage in order:
    1. Feature Engineering: Read Gold data, create ML features, write to Feature Store
    2. Model Training:       Train model, validate, register, predict
    """
    # Step 1: Feature Engineering -- create ML features from Gold data
    print("Starting Feature Engineering...")
    fe = FeatureEngineering()
    fe.run()

    # Step 2: Model Training -- train, validate, register, and predict
    print("Starting Model Training...")
    model = ModelTraining()
    model.run()

    print("ML pipeline complete!")


# ============================================================================
# Entry point -- this runs when you execute the file directly:
#   python ml_pipeline.py
# When you import the file (from ml_pipeline import ...), main()
# does NOT run automatically. You must call it yourself.
# ============================================================================
if __name__ == "__main__":
    main()