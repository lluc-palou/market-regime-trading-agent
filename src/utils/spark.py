from pyspark.sql import SparkSession
from typing import Optional, Dict, Any


def create_spark_session(
    app_name: str,
    db_name: str,
    mongo_uri: str = "mongodb://127.0.0.1:27017/",
    driver_memory: str = "6g",  # Maximum 8GB total RAM (6GB driver + 1.5GB executor)
    jar_files_path: str = "file:///C:/spark/spark-3.4.1-bin-hadoop3/jars/",
    additional_configs: Optional[Dict[str, Any]] = None
) -> SparkSession:
    """
    Given an app name and a database name creates a Spark session with MongoDB connector
    and standard configuration settings.

    Optimized for systems with 8GB RAM maximum.
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
        # Memory configurations (optimized for 8GB RAM maximum)
        .config("spark.driver.memory", driver_memory)  # 6GB driver memory
        .config("spark.executor.memory", "1536m")  # 1.5GB executor memory
        .config("spark.driver.maxResultSize", "1536m")  # Max result size before spilling
        .config("spark.memory.fraction", "0.8")  # 80% of heap for execution/storage
        .config("spark.memory.storageFraction", "0.3")  # 30% of memory.fraction for caching
        # Performance configurations
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "32")  # Increased from 16 for better parallelism
        .config("spark.default.parallelism", "8")  # Better parallelism for local mode
        .config("spark.sql.session.timeZone", "UTC")
        # Garbage collection tuning for better memory management
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35")
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35")
        # Timeout configurations to prevent socket timeouts
        .config("spark.network.timeout", "1800s")  # 30 minutes for network operations
        .config("spark.executor.heartbeatInterval", "60s")  # Heartbeat every 60s
        .config("spark.python.worker.reuse", "true")  # Reuse Python workers
        .config("spark.rpc.askTimeout", "1800s")  # 30 minutes for RPC calls
        .config("spark.rpc.lookupTimeout", "1800s")  # 30 minutes for RPC lookups
        .config("spark.core.connection.ack.wait.timeout", "1800s")  # Connection timeout
    )
    
    # Adds any additional configuration settings if provided
    if additional_configs:
        for key, value in additional_configs.items():
            builder = builder.config(key, value)
    
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    return spark