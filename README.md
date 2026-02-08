# Deep Reinforcement Learning for Trading in Cryptocurrency Markets Using Limit Order Book Data and Synthetic Augmentation

This project explores whether exploitable patterns exist in the market microstructure of near high-frequency limit order book data from cryptocurrency assets like Bitcoin. The research investigates pattern identification and exploitation during price formation through machine learning to develop algorithmic trading strategies.

The framework implements a statistically robust validation methodology combining three core components:
* VQ-VAE: Extracts discrete latent representations of market regimes from LOB data
* Transformer + PPO: Learns optimal trading policies using reinforcement learning
* Prior Model: Captures temporal dynamics of latent representations for synthetic data generation

The system evaluates trading strategies across multiple feature representations (latent codes, hand-crafted LOB indicators, hybrid approaches) and transaction cost scenarios, demonstrating that exploitable patterns exist in market microstructure under realistic trading conditions.

This work was developed as the final degree project for the BSc in Data Science and Engineering at Universitat Politècnica de Catalunya (UPC), Barcelona, and forms part of ongoing research.

## Project Structure
```
drl-lob/
│
├── scripts/                                    # Pipeline orchestration and execution
│   ├── run_pipeline.py                         # Main pipeline orchestrator (train/test modes)
│   ├── pipeline_init.py                        # Initialize MongoDB working collections
│   ├── 01_data_collection.py                   # Stage 1: Collect raw LOB data from websocket
│   ├── 02_data_ingestion.py                    # Stage 2: Ingest .parquet files to MongoDB
│   ├── 03_data_splitting.py                    # Stage 3: Assign CPCV fold IDs to samples
│   ├── 04_feature_derivation.py                # Stage 4: Derive LOB-based features
│   ├── 05_lob_standardization.py               # Stage 5: LOB standardization
│   ├── 06_materialize_splits.py                # Stage 6: Create split collections
│   ├── 07_feature_transform.py                 # Stage 7: Select feature transformations
│   ├── 08_apply_feature_transforms.py          # Stage 8: Apply transformations
│   ├── 09_test_stylized_facts.py               # Stage 9: Statistical analysis
│   ├── 10_feature_scale.py                     # Stage 10: EWMA half-life selection
│   ├── 11_apply_feature_standardization.py     # Stage 11: Apply EWMA standardization
│   ├── 12_filter_nulls.py                      # Stage 12: Filter null values
│   ├── 13_vqvae_hyperparameter_search.py       # Stage 13: VQ-VAE hyperparameter search
│   ├── 14_vqvae_production.py                  # Stage 14: Train production VQ-VAE
│   ├── 15_prior_hyperparameter_search.py       # Stage 15: Prior hyperparameter search
│   ├── 16_prior_production.py                  # Stage 16: Train production Prior
│   ├── 17_synthetic_generation.py              # Stage 17: Generate synthetic sequences
│   ├── 18_ppo_training.py                      # Stage 18: PPO agent training
│   └── 19_generalization_validation.py         # Stage 19: Quality assurance validation
│
├── src/                                        # Core implementation modules
│   ├── feature_standardization/                # EWMA-based feature standardization
│   ├── feature_transformation/                 # Feature transformation selection
│   ├── hand_crafted_features/                  # LOB-based feature engineering
│   ├── lob_standardization/                    # LOB price/volume standardization
│   ├── vqvae_representation/                   # VQ-VAE for discrete latent encoding
│   ├── prior/                                  # Prior model for latent temporal dynamics
│   ├── ppo/                                    # PPO reinforcement learning agent
│   ├── generalization_validation/              # Quality assurance metrics
│   ├── stylized_facts/                         # Statistical property testing
│   ├── split_materialization/                  # Split data materialization
│   ├── validation/                             # CPCV and data validation
│   ├── pipeline/                               # Pipeline orchestration utilities
│   └── utils/                                  # Common utilities (logging, database, etc.)
│
├── ops/                                        # Operations and utilities
│   ├── S3_download_dataset.py                  # Download LOB data from S3
│   ├── start_mlflow.bat                        # MLflow UI startup script
│   └── start_mongodb.bat                       # MongoDB startup script
│
└── env/                                        # Environment setup
    ├── environment.yaml                        # Conda environment specification
    └── installation_guide.md                   # Setup guide
```

## Key Technologies

### Machine Learning & Deep Learning
- **PyTorch** - Neural network architectures (VQ-VAE, Transformer-based Prior, PPO agent)

### Data Processing & Engineering
- **PySpark** - Distributed data processing for large-scale LOB datasets
- **MongoDB** - Document storage for LOB snapshots and derived features

### Experimentation & Infrastructure
- **MLflow** - Experiment tracking and model versioning
- **AWS (S3, EC2)** - Cloud storage and distributed GPU training
- **Conda** - Environment and dependency management

### Validation & Cross-Validation
- **Combinatorial Purged Cross-Validation (CPCV) with Embargo** - Rigorous time-series validation methodology preventing information leakage
- **Stylized Facts Analysis** - Statistical property testing for financial time series

## Setup

Clone the repository and create the conda environment:
```bash
# Clone repository
git clone https://github.com/yourusername/drl-lob.git
cd drl-lob

# Create and activate conda environment
conda env create -f env/environment.yaml
conda activate drl-lob
```

### Additional Configuration

- **Installation Guide**: See `env/installation_guide.md` for detailed setup instructions (Java, Scala, Spark, MongoDB)
- **MongoDB**: Start MongoDB instance for data storage (see `ops/start_mongodb.bat`)
- **MLflow**: Launch MLflow UI for experiment tracking: `mlflow ui` (or use `ops/start_mlflow.bat`)
- **AWS S3**: Optional configuration for additional dataset management

## Execution

### Pipeline Execution
```bash
# Initialize MongoDB collections
python scripts/pipeline_init.py

# Run full training pipeline
python scripts/run_pipeline.py --mode train

# Run test/evaluation pipeline
python scripts/run_pipeline.py --mode test

# Execute individual stages
python scripts/13_vqvae_hyperparameter_search.py
python scripts/14_vqvae_production.py
python scripts/18_ppo_training.py
```

Refer to individual script help messages for detailed parameter options: `python scripts/<script_name>.py --help`

## Contact

For questions or collaboration opportunities, please contact:

**Lluc Palou Masmatí**  
Email: paloumasmarti@gmail.com
