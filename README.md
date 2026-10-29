# MLE Supply Chain Demand Forecasting Pipeline

## Architecture

 This project is a full-stack data + ML pipeline on Databricks that forecasts product-city-level demand. It follows two architectural principles:

* **OOP class-based design** — all logic lives in importable `.py` modules with `BaseLayer` as the shared parent class (config loading, logging, schema creation)
* **Single source of truth** — class definitions live in `.py` files; notebooks are thin wrappers that import and call `.run()` methods

## Key Files

| File | Role |
| --- | --- |
| `src/main.py` | Orchestrator entry point — runs DE then ML stages in sequence, used as the DAB job's `spark_python_task` |
| `src/meadallion_pipeline.py` | Data engineering: `BronzeIngestion`, `SilverIngestion`, `GoldIngestion` |
| `src/ml_pipeline.py` | ML pipeline: `FeatureEngineering`, `ModelTraining` |
| `config/dev.yaml` | Single YAML config file — catalog, schemas, table names, feature params, model lifecycle thresholds |
| `databricks.yaml` | Declarative Automation Bundle (DAB) — defines the job, compute, libraries, dev/prod targets |
| `.github/workflows/cd.yaml` | GitHub Actions CI/CD — validate on PR, deploy+run on push to main |

## End-to-End Flow

### Stage 1: Data Engineering (Medallion Architecture)

* **Bronze** (`BronzeIngestion`): Reads raw CSV from a Unity Catalog volume, writes 102,081 rows to `mle.bronze.supply_chain_dataset_bronze` with an audit timestamp column.

* **Silver** (`SilverIngestion`): Reads Bronze, deduplicates on `transaction_id` (keeping latest by ingestion timestamp), applies type conversions, runs DQ validation (null checks on critical columns, `transaction_id` regex format validation, cross-field `delay_days_mismatch` rule). Good records → `mle.silver.supply_chain_dataset_silver` (100,000 rows), bad records → quarantine table. Adds audit timestamp.

* **Gold** (`GoldIngestion`): Reads Silver, aggregates to the `(product_name, destination_city, transaction_date)` grain via SUM/AVG/COUNT DISTINCT. Excludes post-event columns to prevent target leakage. Writes 74,900 rows to `mle.gold.product_city_demand`.

### Stage 2: Machine Learning (`ml_pipeline.py` — Detailed Architecture)

This file is the ML engine of the pipeline. It reads clean Gold-layer data (produced by the medallion pipeline), engineers ML features, writes them to the Databricks Feature Store, trains an XGBoost model, validates it through a multi-gate framework, registers it in Unity Catalog with a Champion/Challenger lifecycle, makes production predictions, and logs evaluation plots — all in a single importable Python module.

#### Module-Level Structure

The file is organized top-to-bottom in five layers:

1. **Module docstring** (lines 1–24) — describes the two-stage ML pipeline, usage instructions, and class inventory.
2. **Imports** (lines 26–47) — pulls in logging, MLflow (`mlflow`, `mlflow.sklearn`, `MlflowClient`, `infer_signature`), data libraries (`numpy`, `pandas`, `yaml`), Databricks Feature Store (`FeatureEngineeringClient`, `FeatureLookup`), Databricks runtime (`dbutils`), PySpark (`SparkSession`, `functions as F`, `col`, `Window`), and scikit-learn / XGBoost (`ColumnTransformer`, `SimpleImputer`, `Pipeline`, `OneHotEncoder`, `StandardScaler`, `XGBRegressor`, metrics functions).
3. **Spark session** (line 51) — `spark = SparkSession.builder.getOrCreate()` connects to the active Spark session. This is used for all DataFrame operations throughout the file.
4. **Classes** (lines 67–1261) — `BaseLayer`, `FeatureEngineering`, `ModelTraining`.
5. **Entry point** (lines 1266–1293) — `main()` function and `if __name__ == "__main__"` guard.

A helper function `_get_widget_value(name, default)` sits between the Spark session and the classes. It safely retrieves a Databricks widget value, returning the default if the widget doesn't exist (e.g., when running as a standalone `.py` script rather than a notebook). Both `FeatureEngineering.__init__` and `ModelTraining.__init__` use it to locate the config file.

#### Class 1: `BaseLayer` (lines 67–96)

The shared parent class for both ML pipeline stages. It provides:

* **`__init__(config_path, layer_name)`** — stores the config path and layer name, creates a Python `logging.getLogger(layer_name)`, and immediately calls `_load_config()`.
* **`_load_config()`** — opens the YAML config file with `yaml.safe_load()`, extracts `self.catalog` (the Unity Catalog name), and logs that config was loaded.
* **`ensure_schema(schema_name)`** — runs `CREATE SCHEMA IF NOT EXISTS {catalog}.{schema_name}` to guarantee the target Unity Catalog schema exists before writing tables.

Both subclasses call `super().__init__()` which triggers config loading, and inherit `ensure_schema()` for creating the `features` and `models` schemas.

#### Class 2: `FeatureEngineering` (lines 99–289)

Transforms Gold-layer data into ML-ready features and writes them to the Databricks Feature Store.

**`__init__`** — Resolves the config path (via widget or default), calls `super().__init__()`, then extracts layer-specific settings from `cfg["feature_engineering"]`: Gold source table, feature target table, primary keys, and timeseries column. Builds fully-qualified table names (`{catalog}.{schema}.{table}`). Instantiates `FeatureEngineeringClient()`.

**`read_gold()`** — Reads the Gold Delta table via `spark.table(self.gold_full)` and counts rows. The Gold table is at grain `(product_name, destination_city, transaction_date)` with aggregated columns like `demand_quantity`, `avg_unit_price_usd`, `avg_available_inventory`, etc.

**`create_time_features(df)`** — Adds four calendar features derived from `transaction_date` using PySpark functions:

* `day_of_week` (1=Sunday, 7=Saturday)
* `day_of_month` (1–31)
* `is_weekend` (1 if day_of_week is 1 or 7, else 0 — cast to int)
* `quarter` (1–4)

**`create_lag_features(df)`** — Adds lag features using `F.lag()` over a `Window.partitionBy("product_name", "destination_city").orderBy("transaction_date")`. Configurable via `dev.yaml`: `lag_windows` (e.g., `[1, 7]`) and `lag_columns` (e.g., `["demand_quantity"]`). For `demand_quantity`, the feature is named `lag_{n}_demand`; for other columns, `lag_{n}_{col_name}`.

**`create_rolling_features(df)`** — Adds rolling window features using a trailing `Window.rowsBetween(-rolling_window, -1)` that **excludes the current row** to prevent target leakage. For `demand_quantity`, creates three features: `rolling_7_avg_demand`, `rolling_7_sum_demand`, `rolling_7_max_demand`. For other columns, creates `rolling_7_avg_{col_name}`.

**`assemble()`** — Selects the final column set in a deterministic order: 3 primary key columns + 5 original Gold columns + 9 engineered features = 17 columns total. The target column `demand_quantity` is **intentionally excluded** — it must not be stored in the Feature Store. The label comes from the Gold table during training via a point-in-time join.

**`write_feature_store()`** — Creates the `features` schema if needed, drops the existing feature table if it exists, then calls `fe.create_table()` with:

* `primary_keys` = `["product_name", "destination_city", "transaction_date"]`
* `timeseries_column` = `"transaction_date"`
* The assembled feature DataFrame
* A descriptive comment

This creates a time series feature table in Unity Catalog.

**`validate()`** — Logs a summary of the feature engineering output (row count, column count, target table).

**`run()`** — Orchestrates the full feature engineering pipeline in order: `read_gold()` → `create_time_features()` → `create_lag_features()` → `create_rolling_features()` → `assemble()` → `write_feature_store()` → `validate()`.

#### Class 3: `ModelTraining` (lines 292–1261)

Implements the full model lifecycle: train → validate → register → promote → evaluate → predict.

**`__init__`** — Resolves config path, calls `super().__init__()`, then extracts settings from `cfg["model_lifecycle"]` and `cfg["feature_engineering"]`: feature table name, Gold table name, experiment name, model name (`{catalog}.models.{model_name}`), prediction table, target column (`demand_quantity`), categorical features list, and configurable thresholds (`min_r2`, `max_mae`, `max_rmse`). Creates an `MlflowClient` instance for model registry operations.

**`read_features()`** — Reads features from the Feature Store and joins them with labels:

1. Builds `labels_df` from the Gold table, selecting the 3 primary keys + `demand_quantity` as the label.
2. Reads the feature table's column list and computes `feature_names` (all non-PK, non-target columns).
3. Calls `fe.create_training_set()` with a `FeatureLookup` that specifies the feature table, feature names, lookup keys (`product_name`, `destination_city`), and timestamp lookup key (`transaction_date`). This performs a **point-in-time join** — features are looked up as-of each label's timestamp, preventing future data from leaking into training.
4. Loads the result into pandas via `training_set.load_df().toPandas()`.
5. Casts `is_weekend` to `int32`, converts `transaction_date` to pandas datetime, sorts chronologically.
6. Separates features into `categorical_features` (4 columns) and `numerical_features` (remaining non-date, non-target columns). Builds `input_cols` = `transaction_date` + feature columns (date is needed for `score_batch` lookups but is excluded from the model by the `ColumnTransformer`'s `remainder='drop'`).

**`split_data()`** — Performs a **chronological split** (not random) to avoid training on future data:

* Test set: last 20% of rows (newest dates)
* Validation set: next 16% of the remaining 80%
* Train set: remaining 64% (oldest dates)

Creates `X_train/val/test` (features) and `y_train/val/test` (target) DataFrames, and logs the date ranges for each split.

**`_compute_metrics(y_true, y_pred)`** — Utility that returns a dict with `mae`, `rmse`, and `r2` as floats.

**`_train_candidate()`** — Trains the XGBoost model and logs it to MLflow:

1. Builds a `ColumnTransformer` with two branches:
   * **Numeric**: `SimpleImputer(strategy="mean")` → `StandardScaler()`
   * **Categorical**: `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`
2. Wraps the preprocessor and an `XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)` inside a sklearn `Pipeline`.
3. Starts an MLflow run (`run_name="xgboost_candidate"`), fits the model on the train set, and predicts on the validation set.
4. Logs params (model type, split strategy, sizes, feature count), metrics (val MAE, RMSE, R²), and tags.
5. Creates a model signature via `infer_signature()` using 100 sample rows.
6. Logs the model with `fe.log_model(flavor=mlflow.sklearn, training_set=...)` — this preserves Feature Store lineage metadata so that `score_batch` can automatically look up features at scoring time.
7. Stores `self.model_uri` and `self.candidate_run_id` for later use.

**`_technical_validation()`** — Runs 5 technical gates on the candidate model:

| Gate | Check | Blocks? |
| --- | --- | --- |
| 1 | Model artifact loadable from MLflow (`mlflow.pyfunc.load_model`) | Yes |
| 2 | Model signature exists (downloads artifacts, loads `MLmodel` file) | Yes |
| 3 | Predictions contain no NaN or infinite values | Yes |
| 4 | All prediction values are numeric (checks first 100 predictions) | Yes |
| 5 | No negative predictions (warning only, clipped to 0 at scoring) | No |

Returns `True` only if all blocking gates pass.

**`_data_quality_validation()`** — Runs 4 data-quality gates:

| Gate | Check | Blocks? |
| --- | --- | --- |
| 1 | All required feature columns are present in `X_train` | Yes |
| 2 | No unexpected null values (nulls in `lag_`/`rolling_` columns are expected for early rows) | Yes |
| 3 | No date overlap between train and validation sets (no leakage) | Yes |
| 4 | No date overlap between validation and test sets | Yes |

Returns `True` only if all gates pass.

**`_register_and_challenger()`** — Registers the validated model:

1. Ensures the `models` schema exists.
2. Calls `mlflow.register_model()` to create a new model version in Unity Catalog.
3. Assigns the `"Challenger"` alias via `set_registered_model_alias()`.
4. Sets version tags (`domain`, `framework`, `stage`).
5. Logs validation results as MLflow params on the candidate run.

**`_get_champion_version()`** — Returns the current Champion model version by alias lookup, or `None` if no Champion exists.

**`_evaluate_model_on_validation(model_uri)`** — Evaluates a registered model on the validation set using `fe.score_batch(model_uri=model_uri, df=entity_df)`. Only entity columns (primary keys + timestamp) are passed — `score_batch` fetches the model from the MLflow registry on each executor and automatically looks up features from the Feature Store. Predictions are merged with actuals via inner join, clipped to ≥0, and metrics are computed.

**`_compare_and_promote()`** — The Champion/Challenger comparison logic:

* **No existing Champion**: Candidate must meet absolute thresholds (`R² ≥ min_r2`, `MAE ≤ max_mae`, `RMSE ≤ max_rmse`). If it passes, it's promoted to Champion and the Challenger alias is removed. If it fails, it's rejected.
* **Existing Champion**: Evaluates the Champion on the validation set via `_evaluate_model_on_validation(champion_uri)`. Compares Challenger vs Champion on 3 metrics (R² higher = win, MAE lower = win, RMSE lower = win). Challenger must win **≥2 of 3** to be promoted. If the Champion evaluation fails (throws an exception), the pipeline **fails** — there is no silent auto-promotion.

All promotion decisions are logged to MLflow as tags, params, and metrics.

**`predict()`** — Makes production predictions using the Champion model:

1. If no Champion exists, skips predictions.
2. Builds an entity DataFrame from the test set (only `product_name`, `destination_city`, `transaction_date`).
3. Calls `fe.score_batch(model_uri=champion_uri, df=entity_df)` to generate predictions.
4. Merges with actuals, clips predictions to ≥0.
5. Builds a results DataFrame with: `product_name`, `destination_city`, `transaction_date`, `actual_demand`, `predicted_demand`, `model_name`, `model_version`, `mlflow_run_id`, `alias`.
6. Writes to `mle.models.demand_predictions` as a Delta table (overwrite mode).
7. Tags the table with `domain`, `type`, and `model` metadata.

**`evaluate()`** — Generates and logs three evaluation plots to MLflow:

1. **Actual vs Predicted** scatter plot with a perfect-prediction diagonal line
2. **Residuals vs Predicted** scatter plot with a zero residual line
3. **Residual histogram** with 50 bins

Also computes and logs `test_r2`, `test_rmse`, `test_mae`, and `test_mape` metrics. Uses `matplotlib` with the `Agg` backend (non-interactive) to avoid display issues on serverless compute.

**`_final_test_evaluation()`** — Scores the Champion on the held-out test set (runs **only once**, after model selection is complete). Uses `fe.score_batch` with the Champion model URI. Stores results in `self.test_metrics` for the `evaluate()` method to use.

**`run()`** — Orchestrates the full model lifecycle in strict order:

```
read_features() → split_data() → _train_candidate()
  → _technical_validation() → _data_quality_validation()
    → [if both pass]: _register_and_challenger() → _compare_and_promote()
      → _final_test_evaluation() → predict() → evaluate()
    → [if either fails]: log error and return early
```

#### Data Flow Summary

```
Gold table (mle.gold.product_city_demand)
    ↓
FeatureEngineering.read_gold()
    ↓
create_time_features() → create_lag_features() → create_rolling_features()
    ↓
assemble() → 17 columns (3 PKs + 5 original + 9 engineered, NO target)
    ↓
write_feature_store() → Feature Store table (mle.features.product_city_demand_features)
    ↓
ModelTraining.read_features() → create_training_set() (point-in-time join) → pandas DataFrame
    ↓
split_data() → train (64%) / val (16%) / test (20%)
    ↓
_train_candidate() → XGBRegressor in sklearn Pipeline → fe.log_model() to MLflow
    ↓
_technical_validation() (5 gates) + _data_quality_validation() (4 gates)
    ↓
_register_and_challenger() → Unity Catalog model registry (Challenger alias)
    ↓
_compare_and_promote() → Champion/Challenger comparison on validation set
    ↓
_final_test_evaluation() → Champion scored on test set (once)
    ↓
predict() → fe.score_batch() → mle.models.demand_predictions (Delta table)
    ↓
evaluate() → 3 plots + 4 metrics logged to MLflow
```

#### Key Design Decisions

* **OOP class-based design** — Each pipeline stage is a class inheriting from `BaseLayer`. Config loading, logging, and schema creation are shared; stage-specific logic lives in subclasses.
* **Config-driven** — All table names, feature parameters, model thresholds, and experiment names come from `dev.yaml`. No hardcoded values in the code.
* **Feature Store integration** — Full lifecycle: `create_table` for writing features, `create_training_set` for point-in-time correct training data, `log_model` for feature lineage metadata, `score_batch` for automatic feature lookup at scoring time.
* **No target in Feature Store** — `demand_quantity` is excluded from the feature table. The label comes from the Gold table via the training set join, preventing leakage.
* **Rolling features exclude current row** — `rowsBetween(-rolling_window, -1)` prevents the current row's target value from appearing in its own rolling features.
* **Chronological split** — Time-based, not random, so the model never trains on future data.
* **Multi-gate validation** — 5 technical gates + 4 data-quality gates must all pass before a model is registered. Failures abort the pipeline.
* **Champion/Challenger lifecycle** — Uses MLflow aliases (not deprecated stages). New models start as Challenger, must beat the current Champion on ≥2/3 metrics to be promoted.
* **No silent auto-promotion** — If Champion evaluation fails, the pipeline fails. No try/except swallowing.
* **Serverless-compatible scoring** — All scoring uses `fe.score_batch(model_uri=<actual_uri>, df=entity_df)` rather than `mlflow.pyfunc.load_model().predict()`, which fails on serverless compute.
* **Negative prediction clipping** — All predictions are clipped to ≥0 via `np.maximum(pred, 0)` since demand cannot be negative.
* **`transaction_date` as lookup key, not feature** — Included in `input_cols` for `score_batch` entity lookups but excluded from the model by the `ColumnTransformer`'s `remainder='drop'`.

#### Entry Point

`main()` instantiates `FeatureEngineering()` and `ModelTraining()`, calling `.run()` on each in sequence. The `if __name__ == "__main__"` guard allows the file to be imported as a module without auto-execution.

## Orchestration

`main.py` runs both stages in sequence: `run_de_pipeline()` → `run_ml_pipeline()`. It's deployed as a DAB job (`spark_python_task`) with libraries (`databricks-feature-engineering`, `scikit-learn`, `xgboost`). GitHub Actions CI/CD validates on PR and deploys+runs on push to main.

## Challenges & Obstacles Faced

### 1. Feature Store score_batch ASOF Join Bug

* **Problem**: `fe.score_batch()` did a point-in-time ASOF join to look up features, but returned features for the wrong `(product_name, destination_city)` combinations — e.g., querying (Kerosene, Ahmedabad) returned features for (Bitumen, Hyderabad). Predictions from `score_batch` had R²=-0.49 (worse than mean) while direct model predictions had R²=0.48.
* **Evidence**: Correlation between `score_batch` predictions and direct predictions was -0.013 (essentially random).
* **Initial fix (later superseded)**: Replaced `fe.score_batch()` with `mlflow.pyfunc.load_model().predict()` in three methods — this worked on classic compute but introduced a new problem on serverless (see challenge #2).

### 2. Serverless score_batch with local_uri Error

* **Problem**: When the pipeline moved to serverless compute (via DAB), `mlflow.pyfunc.load_model().predict()` started failing with `ValueError: score_batch with local_uri is not supported on serverless runtime. Use model_uri instead.`
* **Root cause**: `fe.log_model` wraps the model in `_FeatureStoreModelWrapper`. When `predict()` is called on the loaded pyfunc model, the wrapper internally calls `score_batch(model_uri=None)`. With `model_uri=None`, the library falls back to a driver-local download path. On serverless, executors cannot access the driver's local temp directory, so this fails.
* **Key discovery**: `fe.log_model(flavor=mlflow.sklearn)` does NOT produce a sklearn flavor — only a pyfunc (Feature Store wrapper) flavor. `mlflow.sklearn.load_model` raises "Model does not have the 'sklearn' flavor". So the fallback of loading with sklearn directly wasn't available.
* **Final fix**: Replaced all `mlflow.pyfunc.load_model().predict()` scoring calls with `fe.score_batch(model_uri=<actual_model_uri>, df=entity_df)` in three methods (`_evaluate_model_on_validation`, `_final_test_evaluation`, `predict`). Only entity columns (`product_name`, `destination_city`, `transaction_date`) are passed; `score_batch` fetches the model from the MLflow registry on each executor and looks up features from the Feature Store. `fe.log_model` is preserved for Feature Store lineage.

### 3. Silent Auto-Promotion Bug

* **Problem**: In `_compare_and_promote`, a try/except around the Champion evaluation caught any failure and silently promoted the candidate as Champion with `promotion_decision = "promoted_champion_unloadable"`. This meant a candidate could become Champion without ever being compared against the existing Champion — defeating the entire champion/challenger framework.
* **Fix**: Removed the try/except entirely. If `_evaluate_model_on_validation` raises an exception, it propagates and the pipeline fails. The candidate is never promoted when the Champion cannot be evaluated.

### 4. Rolling Feature Data Leakage

* **Problem**: The rolling window used `rowsBetween(-(rolling_window - 1), 0)` which included the current row in the rolling average/sum/max of `demand_quantity`. Since `demand_quantity` is the prediction target, this was target leakage — the model could "cheat" by seeing the current target value in its own features.
* **Fix**: Changed to `rowsBetween(-rolling_window, -1)` — a strictly trailing window that excludes the current row. Lag features (`F.lag()`) were already correct (always look backward).

### 5. XGBoost Not Pre-Installed on Serverless

* **Problem**: `xgboost` is not pre-installed on Databricks Serverless compute. The import `from xgboost import XGBRegressor` fails with `ModuleNotFoundError`.
* **Fix**: XGBoost needs to be installed via `%pip install xgboost` in a notebook cell or added to the DAB bundle's library list. The DAB bundle libraries need to include `xgboost` alongside `databricks-feature-engineering` and `scikit-learn`. (This is still an open item — the current `xgboost` import at the top of `ml_pipeline.py` fails when the package isn't installed.)

### 6. databricks Namespace Package Shadowing on Serverless

* **Problem**: On serverless, the `databricks` namespace package from the base runtime (`databricks-sdk 0.122.0`) shadows the virtualenv version (`0.145.0`). This caused import conflicts with `databricks.feature_engineering`.
* **Fix**: Added `databricks.__path__` reordering in `main.py` to put `.ephemeral_nfs` paths first, so the virtualenv version takes precedence.

### 7. %pip install Doesn't Work in .py Files

* **Problem**: When converting from notebooks to `.py` modules, `%pip install` (a notebook magic command) causes `SyntaxError` in plain Python files.
* **Fix**: Library installation moved to the orchestrator notebook (cell 2) or the DAB bundle's library list. The `.py` files assume libraries are already installed.

### 8. is_weekend Schema Mismatch

* **Problem**: `is_weekend` was stored as boolean in the Feature Store but expected as integer by the model, causing a schema mismatch with `fe.score_batch`.
* **Fix**: Cast `is_weekend` to `int` in Spark before writing to the Feature Store, and also cast to `int32` in pandas after loading from the training set.

### 9. Champion/Challenger Alias Cleanup

* **Problem**: After promoting a Challenger to Champion, the "Challenger" alias wasn't being removed from the old version, leaving stale aliases.
* **Fix**: Added `delete_registered_model_alias("Challenger")` to all three promotion paths in `_compare_and_promote`.

### 10. Model Evolution Journey

The model went through three iterations:

1. **LinearRegression**: R²=0.48 val, 0.49 test (but `score_batch` showed R²=-0.49 due to the ASOF join bug)
2. **GradientBoostingRegressor**: R²=0.59 val, 0.59 test
3. **XGBoost** (`XGBRegressor`): R²=0.59 val, 0.60 test — current model

### 11. EDA Revealed Fundamental Data Challenge

The Gold EDA notebook revealed that autocorrelation is near zero at all lags (|r|<0.02), there's no meaningful seasonality (monthly variation <3%, day-of-week <2%), and the strong signal is cross-sectional (15 products × 10 cities = 150 combos, demand ranges 453–46,305). This means lag and rolling features add minimal predictive signal — the model primarily relies on categorical identity and cross-sectional baselines. Previous high R² was driven by `ordered_quantity` leakage (r=0.99 with target).

## Current State

The pipeline now uses `fe.score_batch(model_uri=<actual_uri>, df=entity_df)` for all model scoring on serverless, with:

* `fe.log_model` preserved for Feature Store lineage
* No silent auto-promotion — evaluation failures raise and fail the pipeline
* Rolling features exclude the current row (no target leakage)
* Comprehensive logging of all model URIs, MAE values, and promotion decisions

The immediate blocker is the `xgboost` import error — it needs to be added to the DAB bundle libraries or installed before the module is imported.
