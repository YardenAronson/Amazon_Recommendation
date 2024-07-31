from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize Spark session with MongoDB connector
spark = SparkSession.builder \
    .appName("myApp") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1:27017") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017") \
    .getOrCreate()

print("_____________Config__________________")

# Use MongoDB aggregation pipeline to load only the required columns
meta_pipeline = """[
    { "$project": { "parent_asin": 1, "average_rating": 1, "rating_number": 1 } }
]"""

products_df = spark.read.format("mongo") \
    .option("uri", "mongodb://127.0.0.1:27017/Amazon.meta") \
    .option("pipeline", meta_pipeline) \
    .load()

print("_____________Done reading__________________")

# Show the schema of the loaded DataFrame to identify column conflicts
products_df.printSchema()

# Show the first few rows of the DataFrame
products_df.show()

# Filter products with at least 50 ratings
filtered_products_df = products_df.filter(col("rating_number") >= 50)

# Calculate the top 10 products based on rating and rating count
top_products_df = filtered_products_df \
    .orderBy(col("average_rating").desc(), col("rating_number").desc()) \
    .limit(10)

# Save the top 10 products to MongoDB
top_products_df.write \
    .format("mongo") \
    .mode("overwrite") \
    .option("database", "Amazon") \
    .option("collection", "top_products") \
    .save()

# Stop the Spark session
spark.stop()
