from pyspark.sql import SparkSession

# Create a Spark session with the MongoDB connector package
spark = SparkSession.builder \
    .appName("myApp") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/Amazon.users") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/Amazon.test") \
    .getOrCreate()

# Read data from MongoDB
df = spark.read.format("mongo").load()

# Show the DataFrame schema and data
df.printSchema()
df.show()

# Example DataFrame to write
data = [
    {"name": "Alice", "age": 55},
    {"name": "Charlie", "age": 35}
]

# Create a DataFrame
df_to_write = spark.createDataFrame(data)

# Write DataFrame to MongoDB
df_to_write.write.format("mongo").mode("overwrite").save()

# Stop the Spark session
spark.stop()
