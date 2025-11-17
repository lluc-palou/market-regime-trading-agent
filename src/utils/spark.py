import os
import sys
from pyspark.sql import SparkSession
from typing import Optional, Dict, Any
import os


def create_spark_session(
    app_name: str,
    db_name: str,
    mongo_uri: str = "mongodb://127.0.0.1:27017/",
    driver_memory: str = "8g",  # Optimized for 16GB server (8GB driver + 3GB executor + 4GB MongoDB + 1GB OS)
    jar_files_path: str = "file:///C:/spark/spark-3.4.1-bin-hadoop3/jars/",
    additional_configs: Optional[Dict[str, Any]] = None
) -> SparkSession:
    """
    Given an app name and a database name creates a Spark session with MongoDB connector
    and standard configuration settings.

    Optimized for 16GB server running both Spark and MongoDB.
    Memory allocation: 8GB driver + 3GB executor + 4GB MongoDB + 1GB OS/overhead
    """
    # Fix Windows temp directory permission issues for Spark
    if sys.platform == 'win32':
        temp_dir = os.path.join(os.path.expanduser('~'), '.spark_temp')
        os.makedirs(temp_dir, exist_ok=True)
        os.environ['TMPDIR'] = temp_dir
        os.environ['TEMP'] = temp_dir
        os.environ['TMP'] = temp_dir

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
        # Memory configurations (optimized for 16GB server with MongoDB)
        .config("spark.driver.memory", driver_memory)  # 8GB driver memory
        .config("spark.executor.memory", "3072m")  # 3GB executor memory
        .config("spark.driver.maxResultSize", "2g")  # Max result size before spilling
        .config("spark.memory.fraction", "0.8")  # 80% of heap for execution/storage
        .config("spark.memory.storageFraction", "0.3")  # 30% of memory.fraction for caching
        .config("spark.executor.pyspark.memory", "3072m")  # Match executor memory
        # Performance configurations
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "32")  # Increased from 16 for better parallelism
        .config("spark.default.parallelism", "8")  # Better parallelism for local mode
        .config("spark.sql.session.timeZone", "UTC")
        # Garbage collection tuning for better memory management
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35")
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35")
        # Timeout configurations to prevent socket timeouts (increased for large datasets)
        .config("spark.network.timeout", "7200s")  # 2 hours for network operations
        .config("spark.executor.heartbeatInterval", "60s")  # Heartbeat every 60s
        .config("spark.python.worker.reuse", "false")  # Disable worker reuse to prevent stuck workers
        .config("spark.python.worker.timeout", "7200")  # 2 hours for Python worker (in seconds)
        .config("spark.rpc.askTimeout", "7200s")  # 2 hours for RPC calls
        .config("spark.rpc.lookupTimeout", "7200s")  # 2 hours for RPC lookups
        .config("spark.core.connection.ack.wait.timeout", "7200s")  # Connection timeout
    )
    
    # Adds any additional configuration settings if provided
    if additional_configs:
        for key, value in additional_configs.items():
            builder = builder.config(key, value)
    
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    return spark