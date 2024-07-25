from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Amazon Recommendation System") \
    .config("spark.executor.memory", "16g") \
    .config("spark.driver.memory", "16g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.shuffle.file.buffer", "32k") \
    .config("spark.shuffle.memoryFraction", "0.3") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
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

# Show the resulting DataFrame
result_df.show()

result_df = result_df.withColumn("num_id", result_df["num_id"].cast("int"))
result_df = result_df.withColumn("num_product", result_df["num_product"].cast("int"))


# Split data into training and test sets (adjust ratio as needed)
(training_data, test_data) = result_df.randomSplit([0.8, 0.2], seed=42)
print("split data")
# Define parameter grid for ALS
param_grid = ParamGridBuilder() \
    .addGrid(ALS.rank, [10, 20, 30]) \
    .addGrid(ALS.regParam, [0.1, 0.01, 0.001]) \
    .build()
print("param_grid")
# Initialize ALS model
als = ALS(userCol="num_id", itemCol="num_product", ratingCol="rating", coldStartStrategy="drop")

# Train-Validation split with hyperparameter tuning
train_validation = TrainValidationSplit(
    estimator=als,
    estimatorParamMaps=param_grid,
    evaluator=RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction"),
    trainRatio=0.8  # 80% training, 20% validation within training
)
print("start training")
# Fit the model
model = train_validation.fit(training_data)

# Get the best model with tuned hyperparameters
best_model = model.bestModel

# Evaluate on test data
predictions = best_model.transform(test_data)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
test_rmse = evaluator.evaluate(predictions)
print("Test RMSE with best hyperparameters:", test_rmse)

# Generate top 10 product recommendations for each user
user_recommendations = best_model.recommendForAllUsers(10)

# Generate top 10 user recommendations for each product
product_recommendations = best_model.recommendForAllItems(10)

# Define paths for saving recommendations and the model
user_recommendations_path = "user_recommendations.parquet"
product_recommendations_path = "product_recommendations.parquet"
model_path = "als_model_best"

# Save recommendations
user_recommendations.write.parquet(user_recommendations_path)
product_recommendations.write.parquet(product_recommendations_path)

# Save the trained model
best_model.save(model_path)
print("--------!!!saved!!!----------")



spark.stop()