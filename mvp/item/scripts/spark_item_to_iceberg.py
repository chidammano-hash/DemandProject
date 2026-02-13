"""Item-only Spark job: read normalized CSV and write Iceberg silver.dim_item."""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp

spark = (
    SparkSession.builder.appName("item-mvp-to-iceberg")
    .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.iceberg.type", "rest")
    .config("spark.sql.catalog.iceberg.uri", "http://iceberg-rest:8181")
    .config("spark.sql.catalog.iceberg.warehouse", "s3://warehouse/")
    .getOrCreate()
)

csv_path = "/opt/mvp/data/itemdata_clean.csv"

df = spark.read.option("header", "true").csv(csv_path)

df2 = (
    df.withColumn("item_ck", col("item_no"))
    .withColumn("load_ts", current_timestamp())
    .withColumn("modified_ts", current_timestamp())
)

spark.sql("CREATE NAMESPACE IF NOT EXISTS iceberg.silver")

df2.writeTo("iceberg.silver.dim_item").using("iceberg").createOrReplace()

print("Wrote iceberg.silver.dim_item")
spark.stop()
