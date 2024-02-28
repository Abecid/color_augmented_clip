from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import random

laion_path = "/Users/adamlee/Downloads/Assets/Datasets/LAION_400M/*"

# Initialize Spark session
spark = SparkSession.builder.appName("ColorDatasetAugmentation").getOrCreate()

# Define your colors
colors = ["red", "blue", "green", "yellow", "orange", "pink", "brown", "black", "white", "purple"]  # Add all colors you're interested in

# UDF to check if the text contains any color and to create a negative prompt
def contains_color(text):
    for color in colors:
        if color in text:
            return True, text.replace(color, random.choice([c for c in colors if c != color]))
    return False, text

contains_color_udf = udf(contains_color, StringType())

# Load Parquet files
df = spark.read.parquet("/path/to/parquet/files/*")
df = spark.read.parquet(laion_path)

# Filter and augment dataset
augmented_df = df.withColumn("contains_color", contains_color_udf(col("text_prompt")))
filtered_augmented_df = augmented_df.filter(col("contains_color").isNotNull())
