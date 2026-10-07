"""Medallion Pipeline - Bronze -> Silver -> Gold.

This file builds a 3-layer data pipeline (also called a "medallion architecture"):

    Bronze layer  -> Read raw CSV data and save it as a Delta table.
    Silver layer  -> Clean the data: remove duplicates, fix data types,
                     check data quality, and separate good/bad records.
    Gold layer    -> Group the clean data by product, city, and date so
                     it is ready for analysis and forecasting.

How to use this file:
    1. Import it:    from meadallion_pipeline import BronzeIngestion
    2. Run it:       python meadallion_pipeline.py  (calls main())

Classes in this file:
    BaseLayer         -> Shared setup (config, logging, schema creation).
    BronzeIngestion   -> Reads raw data and writes the Bronze table.
    SilverIngestion   -> Cleans data and writes the Silver + quarantine tables.
    GoldIngestion     -> Aggregates data and writes the Gold table.
"""

# --- Imports ---
# logging:  For printing messages as the pipeline runs.
# yaml:     For reading the config file (dev.yaml).
import logging

import yaml
from databricks.sdk.runtime import (
    dbutils,  # For reading Databricks widgets (parameters)
)

# PySpark: For reading, transforming, and writing large datasets.
from pyspark.sql import SparkSession
from pyspark.sql import functions as F  # Spark SQL functions (sum, count, etc.)
from pyspark.sql.window import Window  # For ranking / deduplication operations

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
    except Exception:  # noqa: BLE001
        return default


class BaseLayer:
    """Base class for medallion layer ingestion with shared config and logging.

    Provides config loading from a YAML file, logger setup, and schema creation
    in Unity Catalog. Subclasses implement layer-specific read, transform,
    write, and validate logic.

    Attributes:
        config_path: Path to the YAML configuration file.
        layer_name: Name of the layer (used as the logger name).
        cfg: Parsed YAML configuration dictionary.
        catalog: Unity Catalog name from config.
    """

    def __init__(self, config_path: str, layer_name: str):
        self.config_path = config_path
        self.layer_name = layer_name
        self.logger = logging.getLogger(layer_name)
        self._load_config()

    def _load_config(self) -> None:
        """Load YAML configuration from the configured path.

        The config file (dev.yaml) contains all our settings:
        catalog name, schema names, table names, source file path, etc.
        We read it once and store it as self.cfg.
        """
        with open(self.config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.catalog = self.cfg["catalog"]
        self.logger.info(f"Config loaded from {self.config_path}")

    def ensure_schema(self, schema_name: str) -> None:
        """Create the target schema in Unity Catalog if it doesn't exist.

        A schema is like a folder in Unity Catalog that holds tables.
        This makes sure the folder exists before we try to write to it.
        """
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.catalog}.{schema_name}")
        self.logger.info(f"Ensured schema exists: {self.catalog}.{schema_name}")


class BronzeIngestion(BaseLayer):
    """Bronze layer: read raw CSV from a UC Volume and write to a Bronze Delta table as-is.

    Adds a single audit column (bronze_ingestion_timestamp) to track when the
    data was ingested. All source format and write options come from the
    'bronze' section of dev.yaml.

    Attributes:
        full_table: Fully qualified Bronze table name (catalog.schema.table).
        source_path: Path to the source CSV file in a UC Volume.
    """

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = _get_widget_value(
                "config_file",
                "/Workspace/Users/girija.zirange@bizmetric.com/MLE/config/dev.yaml",
            )
        super().__init__(config_path, "bronze_ingestion")
        self.layer_cfg = self.cfg["bronze"]
        self.schema = self.cfg["schemas"]["bronze"]
        self.source_path = self.cfg["source_path"]
        source_override = _get_widget_value("source_path")
        if source_override:
            self.source_path = source_override
        self.table_name = self.layer_cfg["target"]["table"]
        self.full_table = f"{self.catalog}.{self.schema}.{self.table_name}"
        self.logger.info(f"Source:  {self.source_path}")
        self.logger.info(f"Target:  {self.full_table}")

    def read_source(self):
        """Read the source file as-is using format and options from config.

        We read the raw CSV file exactly as it is, without any changes.
        The file format (CSV) and options (like header=True) come from
        the 'bronze' section of dev.yaml.
        """
        # Build a reader for the configured file format (e.g., CSV)
        reader = spark.read.format(self.layer_cfg["source"]["format"])
        # Add each option from config (e.g., header=True, delimiter=,)
        for opt_key, opt_val in self.layer_cfg["source"]["options"].items():
            reader = reader.option(opt_key, opt_val)
        # Load the file and count how many rows we got
        self.df = reader.load(self.source_path)
        self.row_count = self.df.count()
        self.logger.info(f"Rows read: {self.row_count:,}")
        self.logger.info(f"Columns:  {len(self.df.columns)}")
        return self.df

    def write_bronze(self) -> None:
        """Add an ingestion-timestamp audit column and write to the Bronze Delta table."""
        self.ensure_schema(self.schema)
        audit_col = self.layer_cfg["audit_columns"]["ingestion_timestamp"]
        write_cfg = self.layer_cfg["write"]
        (
            self.df.withColumn(audit_col, F.current_timestamp())
            .write.format(write_cfg["format"])
            .mode(write_cfg["mode"])
            .option("overwriteSchema", str(write_cfg["overwrite_schema"]).lower())
            .saveAsTable(self.full_table)
        )
        self.logger.info(f"Written {self.row_count:,} rows to {self.full_table}")

    def validate(self) -> None:
        """Log Bronze table row and column counts."""
        bronze_df = spark.table(self.full_table)
        self.logger.info(
            f"Bronze table: {bronze_df.count():,} rows, {len(bronze_df.columns)} columns"
        )

    def run(self) -> None:
        """Execute the full Bronze ingestion pipeline."""
        self.read_source()
        self.write_bronze()
        self.validate()


class SilverIngestion(BaseLayer):
    """Silver layer: deduplicate, convert types, validate DQ, and split good/bad records.

    Reads from the Bronze table, removes duplicates on transaction_id (keeping
    earliest ingestion), applies type conversions, runs data-quality checks,
    and writes good records to the Silver table and bad records to quarantine.

    Attributes:
        bronze_full: Fully qualified Bronze source table name.
        silver_full: Fully qualified Silver target table name.
        quarantine_full: Fully qualified quarantine table name.
    """

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = _get_widget_value(
                "config_file",
                "/Workspace/Users/girija.zirange@bizmetric.com/MLE/config/dev.yaml",
            )
        super().__init__(config_path, "silver_ingestion")
        self.layer_cfg = self.cfg["silver"]
        self.bronze_schema = self.cfg["schemas"]["bronze"]
        self.silver_schema = self.cfg["schemas"]["silver"]
        self.bronze_table = self.layer_cfg["source"]["bronze_table"]
        self.silver_table = self.layer_cfg["target"]["silver_table"]
        self.quarantine_table = self.layer_cfg["target"]["quarantine_table"]
        self.bronze_full = f"{self.catalog}.{self.bronze_schema}.{self.bronze_table}"
        self.silver_full = f"{self.catalog}.{self.silver_schema}.{self.silver_table}"
        self.quarantine_full = (
            f"{self.catalog}.{self.silver_schema}.{self.quarantine_table}"
        )
        self.logger.info(f"Bronze source:     {self.bronze_full}")
        self.logger.info(f"Silver target:     {self.silver_full}")
        self.logger.info(f"Quarantine target: {self.quarantine_full}")

    def read_bronze(self):
        """Read the Bronze Delta table."""
        self.bronze_df = spark.table(self.bronze_full)
        self.bronze_count = self.bronze_df.count()
        self.logger.info(f"Bronze rows read: {self.bronze_count:,}")
        return self.bronze_df

    def deduplicate(self):
        """Remove duplicate rows, keeping the first one that was ingested.

        If the same transaction appears more than once, we keep only the
        earliest copy (based on the ingestion timestamp). We use a window
        function to rank rows within each group and keep rank = 1.
        """
        # Get the column names from config
        dedup_key = self.layer_cfg["dedup"]["key_column"]  # e.g., "transaction_id"
        dedup_order = self.layer_cfg["dedup"][
            "order_column"
        ]  # e.g., "bronze_ingestion_timestamp"
        # Window: group by the key column, sort by the order column
        window = Window.partitionBy(dedup_key).orderBy(dedup_order)
        # Rank each row within its group (1 = first/earliest), keep only rank 1
        self.df_deduped = (
            self.bronze_df.withColumn("_row_rank", F.row_number().over(window))
            .filter(F.col("_row_rank") == 1)
            .drop("_row_rank")
        )
        self.dedup_count = self.df_deduped.count()
        self.removed_count = self.bronze_count - self.dedup_count
        self.logger.info(
            f"After dedup: {self.dedup_count:,} rows (removed {self.removed_count:,} duplicates)"
        )
        return self.df_deduped

    def convert_types(self):
        """Apply type conversions from config and add the silver ingestion timestamp."""
        self.logger.info("Data type conversions applied:")
        df = self.df_deduped
        for conv in self.layer_cfg["type_conversions"]:
            col_name = conv["column"]
            to_type = conv["to_type"]
            if to_type == "date":
                df = df.withColumn(col_name, F.to_date(F.col(col_name)))
            elif to_type == "timestamp":
                df = df.withColumn(col_name, F.to_timestamp(F.col(col_name)))
            self.logger.info(f"  {col_name} -> {to_type}")
        audit_col = self.layer_cfg["audit_columns"]["silver_ingestion_timestamp"]
        df = df.withColumn(audit_col, F.current_timestamp())
        self.logger.info(f"  {audit_col} -> timestamp (new)")
        self.df_typed = df
        return self.df_typed

    def validate_dq(self):
        """Flag bad records based on data quality (DQ) rules from config.

        We check each record against several rules:
        1. Critical columns must not be null (e.g., transaction_id)
        2. Date columns must not be null
        3. Format validations (e.g., transaction_id must match a pattern)
        4. Cross-field validations (e.g., delay_days must match the date difference)

        Records that fail ANY rule are sent to the quarantine table.
        Records that pass ALL rules are sent to the Silver table.
        """
        dq_cfg = self.layer_cfg["data_quality"]
        bad_conditions = []  # List of True/False conditions: True means "this record is bad"

        # Rule 1: Critical columns must not be null
        for col in dq_cfg["critical_columns"]:
            bad_conditions.append(F.col(col).isNull())

        # Rule 2: Date columns must not be null
        for col in dq_cfg["date_validation_columns"]:
            bad_conditions.append(F.col(col).isNull())

        # Rule 3: Format validations (e.g., transaction_id must match a regex pattern)
        for rule in dq_cfg.get("format_validations", []):
            bad_conditions.append(
                F.col(rule["column"]).isNotNull()
                & ~F.col(rule["column"]).rlike(rule["pattern"])
            )

        # Rule 4: Cross-field validations (e.g., delay_days should match date difference)
        for rule in dq_cfg.get("cross_field_validations", []):
            expr = rule["expression"]
            rule_cols = rule["columns"]
            # If any column used in the expression is null, we can't validate -> mark as bad
            null_cond = F.lit(False)
            for c in rule_cols:
                null_cond = null_cond | F.col(c).isNull()
            bad_conditions.append(null_cond | ~F.expr(expr))

        # Combine all conditions with OR: a record is "bad" if ANY condition is true
        is_bad = bad_conditions[0]
        for cond in bad_conditions[1:]:
            is_bad = is_bad | cond

        # Add a True/False column to flag bad records
        self.df_typed = self.df_typed.withColumn("_is_bad_record", is_bad)

        # Split into good records (pass all rules) and bad records (fail any rule)
        self.df_good = self.df_typed.filter(~F.col("_is_bad_record")).drop(
            "_is_bad_record"
        )
        df_bad = self.df_typed.filter(F.col("_is_bad_record"))
        # Build quarantine reasons
        reason_whens = []
        for item in self.layer_cfg["quarantine"]["reason_labels"]:
            if "column" in item:
                reason_whens.append(
                    F.when(F.col(item["column"]).isNull(), F.lit(item["label"]))
                )
            elif "validation" in item:
                for rule in dq_cfg.get("cross_field_validations", []):
                    if rule["name"] == item["validation"]:
                        expr = rule["expression"]
                        rule_cols = rule["columns"]
                        null_cond = F.lit(False)
                        for c in rule_cols:
                            null_cond = null_cond | F.col(c).isNull()
                        reason_whens.append(
                            F.when(null_cond | ~F.expr(expr), F.lit(item["label"]))
                        )
            elif "format_validation" in item:
                for rule in dq_cfg.get("format_validations", []):
                    if rule["name"] == item["format_validation"]:
                        reason_whens.append(
                            F.when(
                                F.col(rule["column"]).isNotNull()
                                & ~F.col(rule["column"]).rlike(rule["pattern"]),
                                F.lit(item["label"]),
                            )
                        )
        bad_reason_expr = F.concat_ws("; ", *reason_whens)
        self.df_bad = df_bad.withColumn("_quarantine_reason", bad_reason_expr).drop(
            "_is_bad_record"
        )
        self.good_count = self.df_good.count()
        self.bad_count = self.df_bad.count()
        self.logger.info(f"Good records:  {self.good_count:,}")
        self.logger.info(f"Bad records:   {self.bad_count:,}")
        return self.df_good, self.df_bad

    def write_silver(self) -> None:
        """Write good records to the Silver table and bad records to the Quarantine table."""
        self.ensure_schema(self.silver_schema)
        write_cfg = self.layer_cfg["write"]
        (
            self.df_good.write.format(write_cfg["format"])
            .mode(write_cfg["mode"])
            .option("overwriteSchema", str(write_cfg["overwrite_schema"]).lower())
            .saveAsTable(self.silver_full)
        )
        self.logger.info(
            f"Written {self.good_count:,} good records to {self.silver_full}"
        )
        if self.bad_count > 0:
            (
                self.df_bad.write.format(write_cfg["format"])
                .mode(write_cfg["mode"])
                .option("overwriteSchema", str(write_cfg["overwrite_schema"]).lower())
                .saveAsTable(self.quarantine_full)
            )
            self.logger.info(
                f"Written {self.bad_count:,} bad records to {self.quarantine_full}"
            )
        else:
            self.logger.info("No bad records to quarantine.")

    def validate(self) -> None:
        """Log Silver ingestion summary."""
        self.logger.info(
            f"Silver: {self.good_count:,} good, {self.bad_count:,} bad, {self.removed_count:,} duplicates removed"
        )

    def run(self) -> None:
        """Execute the full Silver ingestion pipeline."""
        self.read_bronze()
        self.deduplicate()
        self.convert_types()
        self.validate_dq()
        self.write_silver()
        self.validate()


class GoldIngestion(BaseLayer):
    """Gold layer: aggregate Silver data to the product-city-date grain for demand analysis.

    Reads cleaned, validated transaction data from the Silver layer and
    aggregates to the grain of (product_name, destination_city, transaction_date).
    This Gold table serves as the business-ready foundation for downstream
    feature engineering and demand forecasting, without any ML-specific
    transformations.

    Attributes:
        silver_full: Fully qualified Silver source table name.
        gold_full: Fully qualified Gold target table name.
        grain_cols: List of columns defining the Gold grain.
    """

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = _get_widget_value(
                "config_file",
                "/Workspace/Users/girija.zirange@bizmetric.com/MLE/config/dev.yaml",
            )
        super().__init__(config_path, "gold_ingestion")
        self.layer_cfg = self.cfg["gold"]
        self.silver_schema = self.cfg["schemas"]["silver"]
        self.gold_schema = self.cfg["schemas"]["gold"]
        self.silver_table = self.layer_cfg["source"]["silver_table"]
        self.gold_table = self.layer_cfg["target"]["gold_table"]
        self.silver_full = f"{self.catalog}.{self.silver_schema}.{self.silver_table}"
        self.gold_full = f"{self.catalog}.{self.gold_schema}.{self.gold_table}"
        self.grain_cols = self.layer_cfg["grain"]["key_columns"]
        self.logger.info(f"Silver source:  {self.silver_full}")
        self.logger.info(f"Gold target:    {self.gold_full}")

    def read_silver(self):
        """Read the Silver Delta table (cleaned, typed, deduplicated, validated)."""
        self.silver_df = spark.table(self.silver_full)
        self.silver_count = self.silver_df.count()
        self.logger.info(f"Silver rows read: {self.silver_count:,}")
        return self.silver_df

    def aggregate(self):
        """Group Silver data by product, city, and date — and summarize each group.

        For each unique combination of (product_name, destination_city, transaction_date),
        we calculate:
        - SUM of demand and ordered quantities (how much was ordered/demanded)
        - COUNT of distinct transactions (how many transactions in this group)
        - AVG of prices, costs, inventory, and supplier scores
        - MIN of dimension columns (these are the same for all rows in a group)

        This reduces many rows of transaction-level data into one row per
        product-city-date combination, ready for analysis and forecasting.
        """
        self.logger.info("Aggregating Silver data to Gold grain...")
        self.gold_df = (
            self.silver_df.groupBy(
                "product_name", "destination_city", "transaction_date"
            )
            .agg(
                # Dimension columns: same for all rows in a group, so MIN just picks one value
                F.min("transaction_month").alias("transaction_month"),
                F.min("product_category").alias("product_category"),
                F.min("quantity_unit").alias("quantity_unit"),
                # Demand measures: add up quantities across all transactions in the group
                F.sum("demand_quantity").alias("demand_quantity"),
                F.sum("ordered_quantity").alias("ordered_quantity"),
                F.countDistinct("transaction_id").alias("transaction_count"),
                # Business measures: average prices and costs across the group
                F.avg("unit_price_usd").alias("avg_unit_price_usd"),
                F.avg("product_cost_usd").alias("avg_product_cost_usd"),
                # Business measures: total cost across all transactions
                F.sum("total_cost_usd").alias("total_cost_usd"),
                # Inventory: average available inventory
                F.avg("available_inventory").alias("avg_available_inventory"),
                # Supplier quality: average reliability score
                F.avg("supplier_reliability_score").alias(
                    "avg_supplier_reliability_score"
                ),
            )
            # Select columns in a clean, logical order
            .select(
                "transaction_date",
                "transaction_month",
                "product_name",
                "product_category",
                "quantity_unit",
                "destination_city",
                "demand_quantity",
                "ordered_quantity",
                "transaction_count",
                "avg_unit_price_usd",
                "avg_product_cost_usd",
                "total_cost_usd",
                "avg_available_inventory",
                "avg_supplier_reliability_score",
            )
        )
        self.gold_count = self.gold_df.count()
        self.logger.info(
            f"Silver: {self.silver_count:,} -> Gold: {self.gold_count:,} ({self.silver_count / self.gold_count:.2f}x compression)"
        )
        return self.gold_df

    def write_gold(self) -> None:
        """Write the Gold table as a Delta table (idempotent — overwrite mode)."""
        self.ensure_schema(self.gold_schema)
        write_cfg = self.layer_cfg["write"]
        (
            self.gold_df.write.format(write_cfg["format"])
            .mode(write_cfg["mode"])
            .option("overwriteSchema", str(write_cfg["overwrite_schema"]).lower())
            .saveAsTable(self.gold_full)
        )
        self.logger.info(f"Written {self.gold_count:,} rows to {self.gold_full}")

    def validate(self) -> None:
        """Log Gold ingestion summary."""
        self.logger.info(
            f"Gold: {self.gold_count:,} rows from {self.silver_count:,} silver rows ({self.silver_count / self.gold_count:.2f}x compression)"
        )

    def run(self) -> None:
        """Execute the full Gold ingestion pipeline."""
        self.read_silver()
        self.aggregate()
        self.write_gold()
        self.validate()


# ============================================================================
# Main function — run this file as a script
# ============================================================================
def main():
    """Run the full Medallion pipeline: Bronze -> Silver -> Gold.

    This function creates and runs each layer in order:
    1. Bronze: Read raw CSV data and save it as a Delta table
    2. Silver: Clean, deduplicate, and validate the data
    3. Gold:   Aggregate the clean data for analysis
    """
    # Step 1: Bronze layer — read raw data from the source file
    print("Starting Bronze layer...")
    bronze = BronzeIngestion()
    bronze.run()

    # Step 2: Silver layer — clean and validate the data
    print("Starting Silver layer...")
    silver = SilverIngestion()
    silver.run()

    # Step 3: Gold layer — aggregate for analysis
    print("Starting Gold layer...")
    gold = GoldIngestion()
    gold.run()

    print("Medallion pipeline complete!")


# ============================================================================
# Entry point — this runs when you execute the file directly:
#   python meadallion_pipeline.py
# When you import the file (from meadallion_pipeline import ...), main()
# does NOT run automatically. You must call it yourself.
# ============================================================================
if __name__ == "__main__":
    main()
