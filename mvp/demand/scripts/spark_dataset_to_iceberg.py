"""Generic Spark job to write dataset CSV (dimension or fact) to Iceberg."""

import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, current_timestamp, to_date


def build_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("demand-dataset-to-iceberg")
        .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.iceberg.type", "rest")
        .config("spark.sql.catalog.iceberg.uri", "http://iceberg-rest:8181")
        .config("spark.sql.catalog.iceberg.warehouse", "s3://warehouse/")
        .getOrCreate()
    )


def run_item(spark: SparkSession) -> None:
    csv_path = "/opt/mvp/data/itemdata_clean.csv"
    df = spark.read.option("header", "true").csv(csv_path)

    df2 = (
        df.select(
            col("item_no").alias("item_ck"),
            col("item_no"),
            col("item_desc"),
            col("item_status"),
            col("brand_name"),
            col("category"),
            col("class"),
            col("sub_class"),
            col("country"),
            col("scm_rtd_flag"),
            col("size"),
            col("case_weight").cast("double"),
            col("cpl").cast("int"),
            col("cpp").cast("int"),
            col("lpp").cast("int"),
            col("case_weight_uom"),
            col("bpc").cast("int"),
            col("bottle_pack").cast("int"),
            col("pack_case").cast("int"),
            col("item_proof").cast("double"),
            col("upc"),
            col("national_service_model"),
            col("supplier_no"),
            col("supplier_name"),
            col("item_is_deleted"),
            col("producer_name"),
        )
        .withColumn("modified_ts", current_timestamp())
    )

    spark.sql("CREATE NAMESPACE IF NOT EXISTS iceberg.silver")
    df2.writeTo("iceberg.silver.dim_item").using("iceberg").createOrReplace()
    print("Wrote iceberg.silver.dim_item")


def run_location(spark: SparkSession) -> None:
    csv_path = "/opt/mvp/data/locationdata_clean.csv"
    df = spark.read.option("header", "true").csv(csv_path)

    df2 = (
        df.select(
            col("location_id").alias("location_ck"),
            col("location_id"),
            col("site_id"),
            col("site_desc"),
            col("state_id"),
            col("primary_demand_location"),
        )
        .withColumn("modified_ts", current_timestamp())
    )

    spark.sql("CREATE NAMESPACE IF NOT EXISTS iceberg.silver")
    df2.writeTo("iceberg.silver.dim_location").using("iceberg").createOrReplace()
    print("Wrote iceberg.silver.dim_location")


def run_customer(spark: SparkSession) -> None:
    csv_path = "/opt/mvp/data/customerdata_clean.csv"
    df = spark.read.option("header", "true").csv(csv_path)

    df2 = (
        df.select(
            concat_ws("-", col("site"), col("customer_no")).alias("customer_ck"),
            col("site"),
            col("customer_no"),
            col("customer_name"),
            col("city"),
            col("state"),
            col("zip"),
            col("premise_code"),
            col("status"),
            col("license_name"),
            col("store_type_desc"),
            col("chain_type_desc"),
            col("state_chain_name"),
            col("corp_chain_name"),
            col("rpt_channel_desc"),
            col("rpt_sub_channel_desc"),
            col("rpt_ship_type_desc"),
            col("customer_acct_type_desc"),
            col("delivery_freq_code"),
        )
        .withColumn("modified_ts", current_timestamp())
    )

    spark.sql("CREATE NAMESPACE IF NOT EXISTS iceberg.silver")
    df2.writeTo("iceberg.silver.dim_customer").using("iceberg").createOrReplace()
    print("Wrote iceberg.silver.dim_customer")


def run_time(spark: SparkSession) -> None:
    csv_path = "/opt/mvp/data/timedata_clean.csv"
    df = spark.read.option("header", "true").csv(csv_path)

    df2 = (
        df.select(
            col("date_key").alias("time_ck"),
            to_date(col("date_key")).alias("date_key"),
            col("day_name"),
            col("day_of_week").cast("int"),
            col("day_of_month").cast("int"),
            col("day_of_year").cast("int"),
            col("iso_week_year").cast("int"),
            col("iso_week").cast("int"),
            to_date(col("week_start_date")).alias("week_start_date"),
            to_date(col("week_end_date")).alias("week_end_date"),
            col("month_number").cast("int"),
            col("month_name"),
            to_date(col("month_start_date")).alias("month_start_date"),
            to_date(col("month_end_date")).alias("month_end_date"),
            col("quarter_number").cast("int"),
            col("quarter_label"),
            to_date(col("quarter_start_date")).alias("quarter_start_date"),
            to_date(col("quarter_end_date")).alias("quarter_end_date"),
            col("year_number").cast("int"),
            to_date(col("year_start_date")).alias("year_start_date"),
            to_date(col("year_end_date")).alias("year_end_date"),
            col("week_bucket"),
            col("month_bucket"),
            col("quarter_bucket"),
            col("year_bucket"),
        )
        .withColumn("modified_ts", current_timestamp())
    )

    spark.sql("CREATE NAMESPACE IF NOT EXISTS iceberg.silver")
    df2.writeTo("iceberg.silver.dim_time").using("iceberg").createOrReplace()
    print("Wrote iceberg.silver.dim_time")


def run_dfu(spark: SparkSession) -> None:
    csv_path = "/opt/mvp/data/dfu_clean.csv"
    df = spark.read.option("header", "true").csv(csv_path)

    df2 = (
        df.select(
            concat_ws("_", col("dmdunit"), col("dmdgroup"), col("loc")).alias("dfu_ck"),
            col("dmdunit"),
            col("dmdgroup"),
            col("loc"),
            col("brand"),
            col("abc_vol"),
            col("brand_desc"),
            col("ded_div_sw").cast("int"),
            col("execution_lag").cast("int"),
            col("otc_status"),
            col("premise"),
            col("prod_subgrp_desc"),
            col("region"),
            col("service_lvl_grp"),
            col("size"),
            col("state_plan"),
            col("supergroup"),
            col("supplier_desc"),
            col("total_lt").cast("int"),
            col("vintage").cast("int"),
            col("sales_div"),
            col("purge_sw").cast("int"),
            col("alcoh_pct").cast("double"),
            col("bot_type_desc"),
            col("brand_size"),
            col("cnty"),
            col("dom_imp_opt"),
            col("grape_vrty_desc"),
            col("material"),
            col("prod_cat_desc"),
            col("producer_desc"),
            col("proof").cast("double"),
            col("subclass_desc"),
            col("prod_class_desc"),
            col("file_dt"),
            col("histstart"),
            col("cluster_assignment"),
            col("sop_ref"),
        )
        .withColumn("modified_ts", current_timestamp())
    )

    spark.sql("CREATE NAMESPACE IF NOT EXISTS iceberg.silver")
    df2.writeTo("iceberg.silver.dim_dfu").using("iceberg").createOrReplace()
    print("Wrote iceberg.silver.dim_dfu")


def run_sales(spark: SparkSession) -> None:
    csv_path = "/opt/mvp/data/dfu_lvl2_hist_clean.csv"
    df = spark.read.option("header", "true").csv(csv_path)

    df2 = (
        df.select(
            concat_ws("_", col("dmdunit"), col("dmdgroup"), col("loc"), col("startdate"), col("type")).alias(
                "sales_ck"
            ),
            col("dmdunit"),
            col("dmdgroup"),
            col("loc"),
            to_date(col("startdate")).alias("startdate"),
            col("type").cast("int"),
            col("qty_shipped").cast("double"),
            col("qty_ordered").cast("double"),
            col("qty").cast("double"),
            to_date(col("file_dt")).alias("file_dt"),
        )
        .withColumn("modified_ts", current_timestamp())
    )

    spark.sql("CREATE NAMESPACE IF NOT EXISTS iceberg.silver")
    df2.writeTo("iceberg.silver.fact_sales_monthly").using("iceberg").createOrReplace()
    print("Wrote iceberg.silver.fact_sales_monthly")


def run_forecast(spark: SparkSession) -> None:
    csv_path = "/opt/mvp/data/dfu_stat_fcst_clean.csv"
    df = spark.read.option("header", "true").csv(csv_path)

    df2 = (
        df.select(
            concat_ws("_", col("dmdunit"), col("dmdgroup"), col("loc"), col("fcstdate"), col("startdate")).alias(
                "forecast_ck"
            ),
            col("dmdunit"),
            col("dmdgroup"),
            col("loc"),
            to_date(col("fcstdate")).alias("fcstdate"),
            to_date(col("startdate")).alias("startdate"),
            col("lag").cast("int"),
            col("execution_lag").cast("int"),
            col("basefcst_pref").cast("double"),
            col("tothist_dmd").cast("double"),
        )
        .withColumn("modified_ts", current_timestamp())
    )

    spark.sql("CREATE NAMESPACE IF NOT EXISTS iceberg.silver")
    df2.writeTo("iceberg.silver.fact_external_forecast_monthly").using("iceberg").createOrReplace()
    print("Wrote iceberg.silver.fact_external_forecast_monthly")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write dataset data to Iceberg")
    parser.add_argument("--dataset", required=True, choices=["item", "location", "customer", "time", "dfu", "sales", "forecast"])
    args = parser.parse_args()
    dataset = args.dataset

    spark = build_spark()
    try:
        if dataset == "item":
            run_item(spark)
        elif dataset == "location":
            run_location(spark)
        elif dataset == "customer":
            run_customer(spark)
        elif dataset == "time":
            run_time(spark)
        elif dataset == "dfu":
            run_dfu(spark)
        elif dataset == "sales":
            run_sales(spark)
        else:
            run_forecast(spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
