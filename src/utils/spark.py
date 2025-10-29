from pyspark.sql import SparkSession
from typing import Optional, Dict, Any


def create_spark_session(
    app_name: str,
    db_name: str,
    mongo_uri: str = "mongodb://127.0.0.1:27017/",
    driver_memory: str = "8g",
    jar_files_path: str = "file:///C:/Users/llucp/spark_jars/",
    additional_configs: Optional[Dict[str, Any]] = None
) -> SparkSession:
    """
    Given an app name and a database name creates a Spark session with MongoDB connector 
    and standard configuration settings.
    """
    # Standard JAR files for MongoDB connector
    jar_files = [
        "mongo-spark-connector_2.12-10.1.1.jar",
        "mongodb-driver-core-4.10.1.jar",
        "mongodb-driver-sync-4.10.1.jar",
        "bson-4.10.1.jar"
    ]
    
    # Builds Spark session with base configuration settings
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars", ",".join([jar_files_path + jar for jar in jar_files]))
        .config("spark.mongodb.read.connection.uri", mongo_uri)
        .config("spark.mongodb.write.connection.uri", mongo_uri)
        .config("spark.mongodb.read.database", db_name)
        .config("spark.mongodb.write.database", db_name)
        .config("spark.mongodb.write.ordered", "false")
        .config("spark.mongodb.write.writeConcern.w", "1")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.session.timeZone", "UTC")
    )
    
    # Adds any additional configuration settings if provided
    if additional_configs:
        for key, value in additional_configs.items():
            builder = builder.config(key, value)
    
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    return spark