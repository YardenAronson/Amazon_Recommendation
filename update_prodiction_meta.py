from pyspark.sql import SparkSession
from pyspark.sql.functions import col, struct, collect_list, when
from pyspark.sql.types import DoubleType, StringType

# Initialize Spark session with MongoDB connector
spark = SparkSession.builder \
    .appName("myApp") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1:27017/") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1:27017/") \
    .config('spark.sql.caseSensitive', True) \
    .getOrCreate()

# Load the collections
recommendations_df = spark.read.format("mongo").option("uri", "mongodb://127.0.0.1:27017/Amazon.product_recommendations").load()
meta_df = spark.read.format("mongo").option("uri", "mongodb://127.0.0.1:27017/Amazon.meta").load()

# Rename parent_asin column in recommendations_df for clarity
recommendations_df = recommendations_df.withColumnRenamed("parent_asin", "product_parent_asin")

# Join recommendations with metadata on parent_asin
joined_df = recommendations_df.alias("recommendations").join(
    meta_df.alias("meta"),
    col("recommendations.num_product") == col("meta.parent_asin"),
    "left"
)

# Function to convert columns to DoubleType and handle non-numeric values
def convert_to_double(column):
    return when(col(column).cast(DoubleType()).isNotNull(), col(column).cast(DoubleType())).otherwise(None).alias(column)

# Function to convert columns to StringType
def convert_to_string(column):
    return col(column).cast(StringType()).alias(column)

# Dynamically create the metadata struct with type conversion
metadata_columns = [convert_to_double("meta." + c) if dict(meta_df.dtypes)[c] == 'string' else col("meta." + c) for c in meta_df.columns if c != "parent_asin"]
metadata_columns += [convert_to_string("meta." + c) for c in meta_df.columns if dict(meta_df.dtypes)[c] != 'string' and c != "parent_asin"]
metadata_struct = struct(*metadata_columns).alias("metadata")

# Select necessary columns and structure the recommendations
updated_recommendations_df = joined_df.select(
    "recommendations.num_product",
    "recommendations.num_id",
    "recommendations.rating",
    "meta.parent_asin",
    metadata_struct
)

# Group by num_product to collect the updated recommendations into a list
final_recommendations_df = updated_recommendations_df.groupBy("num_product").agg(
    collect_list(struct(
        "num_product",
        "parent_asin",
        "num_id",
        "rating",
        "metadata"
    )).alias("recommendations")
)

# Save the updated recommendations back to MongoDB
final_recommendations_df.write.format("mongo").option("uri", "mongodb://127.0.0.1:27017/Amazon.updated_user_recommendations").mode("overwrite").save()

print("Updated product recommendations with metadata successfully.")
