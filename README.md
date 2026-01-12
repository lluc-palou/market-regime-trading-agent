# Deep Reinforcement Learning for Trading in Cryptocurrency Markets Using Limit Order Book Data and Synthetic Augmentation

This project explores whether exploitable patterns exist in the market microstructure of high-frequency limit order book data from cryptocurrency assets like Bitcoin. The research investigates if market inefficiencies during price formation can be identified and exploited through machine learning to develop profitable algorithmic trading strategies.

The framework implements a statistically robust validation methodology combining three core components:
- **VQ-VAE**: Extracts discrete latent representations of market regimes from LOB data
- **Transformer + PPO**: Learns optimal trading policies using reinforcement learning
- **Prior Model**: Captures temporal dynamics of latent representations for synthetic data generation

The system evaluates trading strategies across multiple feature representations (latent codes, hand-crafted LOB indicators, hybrid approaches) and transaction cost scenarios, demonstrating that exploitable patterns exist in market microstructure under realistic trading conditions.

## Project Structure

```
drl-lob/
│
├── scripts/                                    <- Pipeline orchestration and stage scripts
│   ├── run_pipeline.py                         <- Main pipeline orchestrator (train/test modes)
│   ├── pipeline_init.py                        <- Initialize MongoDB working collections
│   ├── 01_data_collection.py                   <- Stage 1: Collect raw LOB data from websocket
│   ├── 02_data_ingestion.py                    <- Stage 2: Ingest .parquet files to MongoDB
│   ├── 03_data_splitting.py                    <- Stage 3: Assign CPCV fold IDs to samples
│   ├── 04_feature_derivation.py                <- Stage 4: Derive LOB-based features (microprice, depth, etc.)
│   ├── 05_lob_standardization.py               <- Stage 5: LOB standardization
│   ├── 06_materialize_splits.py                <- Stage 6: Create split_X collections and test_data
│   ├── 07_feature_transform.py                 <- Stage 8: Select and fit feature transformations
│   ├── 08_apply_feature_transforms.py          <- Stage 9: Apply fitted transformers to data
│   ├── 09_test_stylized_facts.py               <- Stage 7-9: Analysis (raw, transformed, standardized)
│   ├── 10_feature_scale.py                     <- Stage 10: Select EWMA half-lives for standardization
│   ├── 11_apply_feature_standardization.py     <- Stage 11: Apply EWMA standardization to data
│   ├── 12_filter_nulls.py                      <- Stage 12: Filter null values from features
│   ├── 13_vqvae_hyperparameter_search.py       <- Stage 13: Hyperparameter search for VQ-VAE
│   ├── 14_vqvae_production.py                  <- Stage 14: Train production VQ-VAE models
│   ├── 15_prior_hyperparameter_search.py       <- Stage 15: Hyperparameter search for Prior model
│   ├── 16_prior_production.py                  <- Stage 16: Train production Prior models
│   ├── 17_synthetic_generation.py              <- Stage 17: Generate synthetic LOB sequences
│   ├── 18_ppo_training.py                      <- Stage 18: PPO agent training/evaluation
│   └── 19_generalization_validation.py         <- Stage 19: Quality assurance validation
│
├── src/                                        <- Source code modules
│   │
│   ├── feature_standardization/                <- EWMA-based feature standardization
│   │   ├── ewma_scaler.py                      <- Exponential moving average scaler implementation
│   │   ├── processor.py                        <- Main processing orchestrator
│   │   ├── aggregator.py                       <- Results aggregation across folds
│   │   ├── apply_scaler.py                     <- Apply fitted scalers to data
│   │   ├── data_loader.py                      <- Data loading utilities
│   │   ├── mlflow_logger.py                    <- MLflow experiment tracking integration
│   │   └── __init__.py
│   │
│   ├── feature_transformation/                 <- Feature transformation selection
│   │   ├── transforms.py                       <- Available transformations (log, sqrt, box-cox, etc.)
│   │   ├── processor.py                        <- Main orchestrator for transformation pipeline
│   │   ├── accumulator.py                      <- Accumulate transformation statistics
│   │   ├── transformation_application.py       <- Apply transformations to features
│   │   ├── data_loader.py                      <- Load data for transformation analysis
│   │   ├── aggregator.py                       <- Results aggregation
│   │   ├── mlflow_logger.py                    <- MLflow integration
│   │   └── __init__.py
│   │
│   ├── hand_crafted_features/                  <- LOB-based feature engineering
│   │   ├── depth_features.py                   <- Order book depth metrics (imbalance, pressure)
│   │   ├── price_features.py                   <- Price-based metrics (microprice, spread)
│   │   ├── volatility_features.py              <- Volatility calculations
│   │   ├── forward_returns.py                  <- Forward-looking return targets
│   │   ├── historical_returns.py               <- Historical return statistics
│   │   ├── batch_loader.py                     <- Batch data loading from MongoDB
│   │   ├── orchestrator.py                     <- Main feature derivation orchestrator
│   │   └── __init__.py
│   │
│   ├── lob_standardization/                    <- LOB standardization
│   │   ├── price_standardizer.py               <- Price standardization
│   │   ├── volume_quantizer.py                 <- Volume quantization to bins
│   │   ├── volume_coverage_analyzer.py         <- Analyze volume distribution coverage
│   │   ├── batch_processor.py                  <- Batch processing for large datasets
│   │   ├── orchestrator.py                     <- Main standardization orchestrator
│   │   └── __init__.py
│   │
│   ├── vqvae_representation/                   <- VQ-VAE for latent encoding
│   │   ├── model.py                            <- VQ-VAE architecture
│   │   ├── trainer.py                          <- Training loop with early stopping
│   │   ├── hyperparameter_search.py            <- Grid search for optimal config
│   │   ├── production_trainer.py               <- Production model training
│   │   ├── latent_generator.py                 <- Generate latent codes from trained model
│   │   ├── data_loader.py                      <- Load training data (NumPy arrays)
│   │   ├── data_loader_pymongo.py              <- PyMongo data loading (streaming)
│   │   ├── config.py                           <- Model configuration dataclass
│   │   ├── mlflow_logger.py                    <- MLflow experiment tracking
│   │   └── __init__.py
│   │
│   ├── prior/                                  <- Prior model for latent distribution
│   │   ├── prior_model.py                      <- Causal CNN for latent sequence prior
│   │   ├── prior_trainer.py                    <- Training loop with validation
│   │   ├── prior_production_trainer.py         <- Production training with best config
│   │   ├── prior_hyperparameter_search.py      <- Hyperparameter search over grid
│   │   ├── synthetic_generator.py              <- Generate synthetic sequences from prior
│   │   ├── prior_data_loader.py                <- Data loading for prior training
│   │   ├── prior_config.py                     <- Configuration dataclass
│   │   └── __init__.py
│   │
│   ├── ppo/                                    <- PPO reinforcement learning agent
│   │   ├── ppo.py                              <- PPO training algorithm implementation
│   │   ├── model.py                            <- Actor-Critic network architectures
│   │   ├── environment.py                      <- Trading environment with LOB data
│   │   ├── buffer.py                           <- Trajectory buffer for PPO updates
│   │   ├── reward.py                           <- Reward calculation functions
│   │   ├── config.py                           <- Hyperparameters and configuration
│   │   ├── utils.py                            <- Utility functions (checkpoints, metrics)
│   │   └── __init__.py
│   │
│   ├── generalization_validation/              <- Quality assurance metrics
│   │   ├── end_to_end_quality.py               <- Full pipeline quality metrics
│   │   ├── vqvae_reconstruction.py             <- VQ-VAE reconstruction quality tests
│   │   ├── prior_quality.py                    <- Prior model generation quality
│   │   ├── metrics.py                          <- Common evaluation metrics
│   │   ├── visualization.py                    <- Plotting utilities for QA reports
│   │   ├── data_loader.py                      <- Data loading for validation
│   │   └── __init__.py
│   │
│   ├── stylized_facts/                         <- Statistical property testing
│   │   ├── pipeline.py                         <- Test orchestrator for stylized facts
│   │   ├── statistical_tests.py                <- Statistical tests (ADF, KPSS, ACF, etc.)
│   │   ├── data_extractor.py                   <- Extract representative windows for testing
│   │   ├── enhanced_aggregator.py              <- Enhanced results aggregation
│   │   ├── results_aggregator.py               <- Aggregate test results across splits
│   │   ├── window.py                           <- Windowing operations for time series
│   │   └── __init__.py
│   │
│   ├── split_materialization/                  <- Split data materialization
│   │   ├── split_materializer.py               <- Create split collections in MongoDB
│   │   ├── per_split_cyclic_manager.py         <- Manage per-split collection cycles
│   │   ├── batch_processor.py                  <- Batch processing for split creation
│   │   └── __init__.py
│   │
│   ├── validation/                             <- Data validation and CPCV
│   │   ├── cpcv.py                             <- Combinatorial Purged Cross-Validation
│   │   ├── folds.py                            <- Fold generation from timeline
│   │   ├── metadata.py                         <- Metadata validation utilities
│   │   ├── stamper.py                          <- Timestamp stamping utilities
│   │   ├── timeline.py                         <- Timeline creation and analysis
│   │   └── __init__.py
│   │
│   ├── pipeline/                               <- Pipeline orchestration utilities
│   │   ├── cyclic_manager.py                   <- MongoDB collection cycling (input/output swap)
│   │   ├── stage_runner.py                     <- Stage execution as subprocesses
│   │   └── __init__.py
│   │
│   └── utils/                                  <- Common utilities
│       ├── logging.py                          <- Unified logging system with timestamps
│       ├── database.py                         <- MongoDB connection and utilities
│       ├── spark.py                            <- Spark session configuration
│       ├── s3_config.py                        <- AWS S3 configuration
│       ├── timestamp.py                        <- Timestamp parsing and formatting
│       └── __init__.py
│
├── ops/                                        <- Operations and utilities
│   ├── S3_download_dataset.py                  <- Download LOB data from S3 (train/test)
│   ├── start_mlflow.bat                        <- MLflow UI startup script (Windows)
│   └── start_mongodb.bat                       <- MongoDB startup script (Windows)
│
├── env/                                        <- Environment setup
│   ├── environment.yaml                        <- Conda environment specification
│   └── installation_guide.md                   <- Setup guide (Java, Scala, Spark, MongoDB)
│
└── .gitignore                                  <- Git ignore patterns
```

## Architecture Overview

### Data Pipeline Flow

```
Raw LOB Data (Parquet)
    ↓
[Stage 2-6] Ingestion → Splitting → Feature Derivation → LOB Standardization → Materialization
    ↓
[Stage 8-12] Feature Transformation → EWMA Standardization → Null Filtering
    ↓
[Stage 13-17] VQ-VAE Training → Prior Training → Synthetic Generation
    ↓
[Stage 18] PPO Training/Evaluation
    ↓
[Stage 19] Quality Assurance Validation
```

### Key Technologies

- **MongoDB**: Cyclic collection management for pipeline data flow
- **PySpark**: Distributed processing for large-scale LOB data
- **MLflow**: Experiment tracking and model versioning
- **PyTorch**: Deep learning (VQ-VAE, Prior, PPO)
- **AWS S3**: Cloud storage for datasets

### Pipeline Execution

```bash
# Run full pipeline (all stages)
python scripts/run_pipeline.py

# Run specific stages
python scripts/run_pipeline.py --start-stage 13 --end-stage 17

# Test mode (uses test_data collection)
python scripts/run_pipeline.py --mode test
```

### Configuration

- Pipeline configuration in each stage script (CONFIG dict)
- Model hyperparameters in `src/*/config.py` modules
- Environment variables for MongoDB, Spark, S3 in `src/utils/`

### Data Storage Pattern

MongoDB collections follow a cyclic input/output pattern:
- `working_input` → processing → `working_output` → swap → `working_input`
- Split collections: `split_0_input`, `split_1_input`, etc.
- Test collection: `test_data` for out-of-sample evaluation

## Development

### Prerequisites

- Python 3.9+
- MongoDB 4.4+
- Apache Spark 3.3+
- Java 11+ (for Spark)

### Setup

```bash
# Create conda environment
conda env create -f env/environment.yaml
conda activate drl-lob

# Start MongoDB
./ops/start_mongodb.bat  # Windows
mongod --dbpath /path/to/data  # Linux/Mac

# Start MLflow (optional)
./ops/start_mlflow.bat
```

### Code Quality

- Total lines: ~20,458 across 89 Python files
- Modular architecture with clear separation of concerns
- Comprehensive logging and experiment tracking
- Production-ready with error handling and validation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Lluc Palou Masmartí - paloumasmarti@gmail.com
