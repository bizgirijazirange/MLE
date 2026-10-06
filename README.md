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

### Stage 2: Machine Learning

**Feature Engineering** (`FeatureEngineering`): Reads Gold, engineers three types of features:

* **Calendar features**: `day_of_week`, `day_of_month`, `is_weekend`, `quarter` — derived from `transaction_date`
* **Lag features**: `lag_1_demand`, `lag_7_demand` — `F.lag()` partitioned by `(product_name, destination_city)`, ordered by `transaction_date`
* **Rolling features**: `rolling_7_avg_demand`, `rolling_7_sum_demand`, `rolling_7_max_demand` — trailing 7-row window (now excluding current row to prevent leakage)

Writes to Feature Store as a time series table (`mle.features.product_city_demand_features`) with primary keys `(product_name, destination_city, transaction_date)` and timeseries column (`transaction_date`).

* **Read Features** (`read_features`): Uses `FeatureEngineeringClient.create_training_set()` with a `FeatureLookup` to do a point-in-time join between Gold labels (`demand_quantity`) and Feature Store features. Loads 74,900 rows × 16 features (4 categorical, 12 numerical) into pandas.

* **Split Data** (`split_data`): Chronological split — 64% train (47,936 rows) / 16% validation (11,984 rows) / 20% test (14,980 rows). Time-based, not random, to prevent training on future data.

* **Train Candidate** (`_train_candidate`): Trains an XGBoost regressor (`XGBRegressor`, `n_estimators=200`, `max_depth=5`, `lr=0.1`) inside a sklearn `Pipeline` with a `ColumnTransformer` (imputation + scaling for numerical, imputation + one-hot encoding for categorical). Logs to MLflow with `fe.log_model(flavor=mlflow.sklearn, training_set=...)` for Feature Store lineage. Computes candidate validation metrics using the in-memory model.

* **Technical Validation** (`_technical_validation`): 5 gates — model artifact loadable, signature exists, no NaN/inf predictions, all-numeric schema, non-negative predictions (warning only).

* **Data-Quality Validation** (`_data_quality_validation`): 4 gates — required feature columns present, no unexpected nulls, no train/val date overlap, no val/test date overlap.

* **Register & Challenger** (`_register_and_challenger`): Registers the model in Unity Catalog (`mle.models.demand_forecast_lr`), assigns the "Challenger" alias.

* **Champion/Challenger Comparison** (`_compare_and_promote`):
  * If no Champion exists: candidate must meet absolute thresholds (R² ≥ 0.3, MAE ≤ 5000, RMSE ≤ 6000)
  * If Champion exists: evaluates Champion on validation set via `fe.score_batch(model_uri=champion_uri, df=entity_df)`, compares against candidate metrics. Candidate promoted only if it wins ≥2/3 metrics (R² higher, MAE/RMSE lower). Evaluation failure now raises and fails the pipeline — no silent auto-promotion.

* **Final Test Evaluation** (`_final_test_evaluation`): Scores the Champion on the held-out test set via `fe.score_batch(model_uri=champion_uri, df=entity_df)`. Runs only once, after model selection is complete.

* **Production Predictions** (`predict`): Uses `fe.score_batch` to generate predictions, merges with actuals, writes 14,980 predictions to `mle.models.demand_predictions`.

* **Evaluation Plots** (`evaluate`): Logs actual-vs-predicted scatter, residuals plot, and residual histogram to MLflow.

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
