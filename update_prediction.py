from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import explode, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

spark = SparkSession.builder \
    .appName("Amazon Recommendation System") \
    .config("spark.executor.memory", "16g") \
    .config("spark.driver.memory", "16g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.shuffle.file.buffer", "32k") \
    .config("spark.shuffle.memoryFraction", "0.3") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
    .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/Amazon.predictions") \
    .config("spark.mongodb.write.connection.uri", "mongodb://127.0.0.1/Amazon.predictions") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .getOrCreate()





products_schema = StructType([
    StructField("parent_asin", StringType(), True),
    StructField("num", IntegerType(), True)
])

users_schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("num", IntegerType(), True)
])

# Load the collections from JSON files
products_df = spark.read.csv('Amazon.products.csv', header=True)
users_df = spark.read.csv('Amazon.users.csv', header=True)
reviews_df = spark.read.json('Musical_Instruments.json')


# Show the dataframes
#products_df.show()
#users_df.show()
#reviews_df.show()


# Rename columns for easier joining
products_df = products_df.withColumnRenamed("parent_asin", "asin") \
                        .withColumnRenamed("num", "num_product")

reviews_df = reviews_df.withColumnRenamed("asin", "product_asin") \
                    .withColumnRenamed("user_id", "review_user_id")

users_df = users_df.withColumnRenamed("num", "user_num") \
                .withColumnRenamed("user_id", "user_id")

# Join products and reviews on asin (now product_asin) and parent_asin (now asin)
merged_df = reviews_df.join(products_df, reviews_df.product_asin == products_df.asin, "inner")

# Join the result with users on user_id
final_df = merged_df.join(users_df, merged_df.review_user_id == users_df.user_id, "inner")

# Select and rename final columns
result_df = final_df.select(
    col("user_num").alias("num_id"),
    col("num_product").alias("num_product"),
    col("rating")
)
result_df = result_df.withColumn("num_id", result_df["num_id"].cast("int"))
result_df = result_df.withColumn("num_product", result_df["num_product"].cast("int"))

# Show the resulting DataFrame
result_df.show()



# Load the ALS model
als_model = ALSModel.load("als_model_best")

# Make predictions
pred = als_model.transform(result_df)

# Show predictions
pred.show()

# Save the predictions to MongoDB
pred.write \
    .format("mongodb") \
    .mode("overwrite") \
    .option("database", "Amazon") \
    .option("collection", "predictions") \
    .save()

# Stop the Spark session
spark.stop()

