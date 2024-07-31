from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql.functions import col, udf, row_number, collect_list
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import numpy as np
from pyspark.sql.window import Window


spark = SparkSession.builder \
    .appName("myApp") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1:27017") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017") \
    .config("spark.executor.memory", "16g") \
    .config("spark.driver.memory", "16g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.shuffle.file.buffer", "32k") \
    .config("spark.shuffle.memoryFraction", "0.3") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
    .getOrCreate()

# Read from the first collection
reviews_df = spark.read.format("mongo") \
    .option("database", "Amazon") \
    .option("collection", "reviews") \
    .load()

# Read from the second collection
products_df = spark.read.format("mongo") \
    .option("database", "Amazon") \
    .option("collection", "products") \
    .load()

# Read from the third collection
users_df = spark.read.format("mongo") \
    .option("database", "Amazon") \
    .option("collection", "users") \
    .load()

# Show some data from each DataFrame to verify


# Rename columns for easier joining
products_df = products_df.withColumnRenamed("num", "num_product")

reviews_df = reviews_df.withColumnRenamed("user_id", "review_user_id")

users_df = users_df.withColumnRenamed("num", "user_num") \
                   .withColumnRenamed("user_id", "user_id")
# reviews_df.show()
# products_df.show()
# users_df.show()

# Join products and reviews on asin (now product_asin) and parent_asin (now asin)
merged_df = reviews_df.join(products_df, reviews_df.parent_asin == products_df.parent_asin, "inner")

# Join the result with users on user_id
final_df = merged_df.join(users_df, merged_df.review_user_id == users_df.user_id, "inner")

# Select and rename final columns
result_df = final_df.select(
    col("user_num").alias("num_id"),
    col("num_product").alias("num_product"),
    col("rating")
)

# Show the resulting DataFrame
# result_df.show()

result_df = result_df.withColumn("num_id", result_df["num_id"].cast("int"))
result_df = result_df.withColumn("num_product", result_df["num_product"].cast("int"))


# Split data into training and test sets (adjust ratio as needed)
(training_data, test_data) = result_df.randomSplit([0.8, 0.2], seed=42)
print("split data")
# Define parameter grid for ALS
param_grid = ParamGridBuilder() \
    .addGrid(ALS.rank, [5, 15, 25, 35, 50]) \
    .addGrid(ALS.regParam, [0.1, 0.01, 0.001, 0.0001]) \
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


# Step 2: Create a new ALS model with the best parameters
new_als = ALS(
    userCol="num_id",
    itemCol="num_product",
    ratingCol="rating",
    rank=best_model.rank,
    coldStartStrategy="drop"
)


# Step 3: Fit the new model on the full dataset
full_model = new_als.fit(result_df)

# Evaluate the new model on the test data
predictions = full_model.transform(test_data)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
test_rmse = evaluator.evaluate(predictions)
print("Test RMSE with new model:", test_rmse)
# Generate top 10 product recommendations for each user
user_recommendations = full_model.recommendForAllUsers(10)


# Join the result with users on user_id
user_recommendations = user_recommendations.join(users_df, user_recommendations.num_id == users_df.user_num, "inner")


# Write user recommendations to MongoDB
user_recommendations.write \
    .format("mongo") \
    .mode("overwrite") \
    .option("database", "Amazon") \
    .option("collection", "user_recommendations") \
    .save()

# print("--------!!!user_recommendations saved!!!----------")

# # Step 1: Extract item factors from the best model
# item_factors = best_model.itemFactors
# print("--------!!!step 1 done!!!----------")

# # Step 2: Define a UDF to calculate cosine similarity
# def cosine_similarity(v1, v2):
#     cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#     return float(cos_sim)

# cosine_similarity_udf = udf(cosine_similarity, FloatType())
# print("--------!!!step 2 done!!!----------")

# # Step 3: Self-join item factors to compare each product with every other product
# item_factors_renamed = item_factors.withColumnRenamed("id", "product_id")
# product_similarity_df = item_factors.alias("i1").crossJoin(item_factors_renamed.alias("i2")) \
#     .select(col("i1.id").alias("product1"), col("i2.product_id").alias("product2"),
#             cosine_similarity_udf("i1.features", "i2.features").alias("similarity"))
# print("--------!!!step 3 done!!!----------")

# # Step 4: Filter out self-similarity (product compared with itself)
# product_similarity_df = product_similarity_df.filter(col("product1") != col("product2"))
# print("--------!!!step 4 done!!!----------")

# # Step 5: For each product, find the top 10 most similar products
# windowSpec = Window.partitionBy("product1").orderBy(col("similarity").desc())
# product_recommendations_df = product_similarity_df.withColumn("rank", row_number().over(windowSpec)) \
#     .filter(col("rank") <= 10) \
#     .drop("rank")
# print("--------!!!step 5 done!!!----------")

# # Step 6: Aggregate the top 10 similar products for each product1 into a list
# recommendations_df = product_recommendations_df.groupBy("product1") \
#     .agg(collect_list("product2").alias("recommendations"))
# print("--------!!!step 6 done!!!----------")

# # Show the schema of the recommendations DataFrame
# recommendations_df.printSchema()


# # Write product-to-product recommendations to MongoDB
# recommendations_df.write \
#     .format("mongo") \
#     .mode("overwrite") \
#     .option("database", "Amazon") \
#     .option("collection", "product_recommendations") \
#     .save()

print("--------!!!product to product saved!!!----------")


spark.stop()