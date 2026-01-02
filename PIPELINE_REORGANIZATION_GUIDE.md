# Pipeline Reorganization Implementation Guide

## Overview

This document provides implementation guidance for reorganizing the pipeline into **Train Mode** (CPCV-based model selection) and **Test Mode** (final evaluation on held-out test data).

## Completed

✅ **`scripts/run_pipeline.py`** - Updated with train/test mode support
- Added `mode` configuration parameter ('train' or 'test')
- Reorganized stages 2-21 with proper train/test flags
- Added artifact validation for test mode
- Passes `--mode` and `--test-split` arguments to compatible stage scripts

## Remaining Implementation Tasks

### Stage 09: Apply Feature Transformations (`08_apply_feature_transforms.py`)

**Current Behavior (Train Mode):**
- Loads fitted transformers from `artifacts/feature_transformation/split_X/`
- Applies to `split_X_input` → `split_X_output`
- Swaps `split_X_output` → `split_X_input`

**Required Changes for Test Mode:**
1. Add CLI argument parser:
```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--test-split', type=int, default=0)
args = parser.parse_args()
```

2. Modify `main()` function:
```python
def main():
    mode = args.mode
    test_split = args.test_split

    if mode == 'train':
        # Existing logic: process all split_X collections
        split_ids = manager.discover_splits()
        for split_id in split_ids:
            fitted_params = load_split_fitted_params(split_id)
            apply_transformations_to_split(spark, split_id, final_transforms, fitted_params)
            manager.swap_split_to_input(split_id)

    else:  # test mode
        # Load fitted transformers from test_split
        fitted_params = load_split_fitted_params(test_split)

        # Apply to test_data collection
        apply_transformations_to_test_data(
            spark,
            final_transforms,
            fitted_params,
            input_coll='test_data',
            output_coll='test_data_transformed'
        )

        # Swap test_data_transformed → test_data
        swap_test_data_collection(spark, 'test_data_transformed', 'test_data')
```

3. Add new function `apply_transformations_to_test_data()`:
```python
def apply_transformations_to_test_data(spark, final_transforms, fitted_params,
                                       input_coll, output_coll):
    """Apply transformations to test_data collection."""
    logger(f"Applying transformations to {input_coll} using fitted params", "INFO")

    # Read test_data
    df = spark.read.format("mongodb") \
        .option("database", DB_NAME) \
        .option("collection", input_coll) \
        .load()

    # Apply transformations (same logic as train mode)
    actual_features = identify_feature_names(df)
    transformed_df = apply_transformations_direct(
        df, actual_features, final_transforms, fitted_params
    )

    # Write to output collection
    write_to_mongodb(transformed_df, DB_NAME, output_coll)
    logger(f"Transformed data written to {output_coll}", "INFO")
```

---

### Stage 12: Apply EWMA Standardization (`11_apply_feature_standardization.py`)

**Same pattern as Stage 09:**
1. Add CLI argument parser for `--mode` and `--test-split`
2. In test mode:
   - Load fitted scalers from `artifacts/feature_scale/split_{test_split}/`
   - Apply to `test_data` → `test_data_standardized`
   - Swap `test_data_standardized` → `test_data`

---

### Stage 14: Null Filtering (`12_filter_nulls.py`)

**Current Behavior (Train Mode):**
- Filters null values from `split_X_input` collections

**Required Changes for Test Mode:**
1. Add CLI argument parser:
```python
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
args = parser.parse_args()
```

2. Modify `main()` function:
```python
def main():
    mode = args.mode

    if mode == 'train':
        # Existing logic: filter all split_X collections
        split_ids = discover_splits()
        for split_id in split_ids:
            filter_split(split_id)

    else:  # test mode
        # Filter test_data collection
        filter_test_data()
```

3. Add `filter_test_data()` function:
```python
def filter_test_data():
    """Filter null values from test_data collection."""
    logger("Filtering null values from test_data", "INFO")

    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    # Count before filtering
    before_count = db['test_data'].count_documents({})

    # Filter documents with null/NaN/Inf in features array
    # (Same logic as train mode)
    result = db['test_data'].delete_many({
        '$or': [
            {'features': None},
            {'features': {'$exists': False}},
            # Add other null checks
        ]
    })

    after_count = db['test_data'].count_documents({})
    logger(f"Filtered {result.deleted_count} documents ({before_count} → {after_count})", "INFO")

    client.close()
```

---

### Stage 16: VQ-VAE Production (`14_vqvae_production.py`)

**Current Behavior (Train Mode):**
- Trains separate VQ-VAE models per split on `role='train'` data
- Encodes `split_X_input` → adds `codebook_index` field
- Saves models as `artifacts/vqvae_models/production/split_X_model.pth`

**Required Changes for Test Mode:**
1. Add CLI argument parser:
```python
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--test-split', type=int, default=0)
args = parser.parse_args()
```

2. Modify `main()` function:
```python
def main():
    mode = args.mode
    test_split = args.test_split

    if mode == 'train':
        # Existing logic: train per split
        for split_id in split_ids:
            train_vqvae_for_split(split_id, role_filter='train')
            encode_split_data(split_id, model_path=f'split_{split_id}_model.pth')

    else:  # test mode
        # Train on FULL split_0 (train + val combined)
        logger(f"Training VQ-VAE on full split_{test_split} (train+val combined)", "INFO")

        # Load best config from hyperparameter search
        best_config = load_best_vqvae_config()

        # Train on split_0 with role IN ('train', 'validation')
        train_vqvae_full_split(
            split_id=test_split,
            config=best_config,
            role_filter=None,  # Include all roles
            output_path=f'artifacts/vqvae_models/test/split_{test_split}_full_model.pth'
        )

        # Encode test_data collection
        encode_test_data(
            model_path=f'artifacts/vqvae_models/test/split_{test_split}_full_model.pth',
            input_coll='test_data',
            output_coll='test_data_encoded'
        )

        # Swap test_data_encoded → test_data
        swap_collection('test_data_encoded', 'test_data')
```

3. Add helper functions:
```python
def load_best_vqvae_config():
    """Load best VQ-VAE config from hyperparameter search."""
    config_path = Path(REPO_ROOT) / "artifacts" / "vqvae_models" / "hyperparameter_search" / "best_config.json"
    with open(config_path, 'r') as f:
        return json.load(f)

def train_vqvae_full_split(split_id, config, role_filter, output_path):
    """Train VQ-VAE on full split (train+val combined)."""
    # Query: role IN ('train', 'validation') if role_filter is None
    # Otherwise query: role = role_filter
    pass  # Implement using existing training logic

def encode_test_data(model_path, input_coll, output_coll):
    """Encode test_data collection using trained VQ-VAE model."""
    # Load model
    # Read test_data
    # Encode features → codebook_index
    # Write to output_coll
    pass  # Implement using existing encoding logic
```

---

### Stage 21: PPO Training/Evaluation (`18_ppo_training.py`)

**Current Behavior (Train Mode):**
- Trains on `split_X` with `role='train'`, validates on `role='validation'`
- Early stopping based on validation Sharpe ratio
- Logs train/val metrics to CSV and MLflow

**Required Changes for Test Mode:**
1. Add CLI argument parser:
```python
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--test-split', type=int, default=0)
parser.add_argument('--experiment', type=int, choices=[1,2,3,4])  # Existing
args = parser.parse_args()
```

2. Modify `main()` function:
```python
def main():
    mode = args.mode
    test_split = args.test_split
    experiment = args.experiment

    if mode == 'train':
        # Existing CPCV logic: train on split_X role='train', validate on role='validation'
        for split_id in split_ids:
            train_ppo_cpcv(split_id, experiment_type)

    else:  # test mode
        # Load best experiment config from train mode
        best_experiment = load_best_ppo_experiment()

        logger(f"Training PPO on full split_{test_split} (train+val combined)", "INFO")
        logger(f"Using best experiment config: {best_experiment}", "INFO")

        # Train on split_0 with role IN ('train', 'validation')
        train_ppo_full_split(
            split_id=test_split,
            experiment_type=best_experiment,
            role_filter=None,  # Include all roles for training
            max_epochs=config.max_epochs,  # Use same max_epochs from train config
            patience=config.patience,  # Use same patience from train config
        )

        # Evaluate on test_data
        test_metrics = evaluate_ppo_on_test(
            split_id=test_split,
            experiment_type=best_experiment,
            test_collection='test_data'
        )

        # Log test metrics
        log_test_metrics(test_metrics)
```

3. Add helper functions:
```python
def load_best_ppo_experiment():
    """Load best PPO experiment from train mode results."""
    # Read CSV logs from train mode
    # Find experiment with highest val_sharpe_taker
    # Return experiment number
    pass

def train_ppo_full_split(split_id, experiment_type, role_filter, max_epochs, patience):
    """Train PPO on full split (train+val combined)."""
    # Load episodes from split_0 with role IN ('train', 'validation')
    # Train for max_epochs with patience (same as train mode)
    # Log only train metrics (no validation during training)
    # Save final model to artifacts/ppo_training/test/split_0_full_model.pth
    pass

def evaluate_ppo_on_test(split_id, experiment_type, test_collection):
    """Evaluate trained PPO agent on test_data."""
    # Load trained model from artifacts/ppo_training/test/split_0_full_model.pth
    # Load test episodes from test_collection
    # Run episodes without training (inference mode)
    # Compute metrics: Sharpe ratios, activity, PnL, etc.
    # Return dict with all test metrics
    pass

def log_test_metrics(test_metrics):
    """Log test metrics to CSV and MLflow."""
    # Save to artifacts/ppo_training/test/test_results.csv
    # Log to MLflow under test run
    logger("Test Metrics:", "INFO")
    logger(f"  Sharpe (Buy-and-Hold): {test_metrics['sharpe_buyhold']:.4f}", "INFO")
    logger(f"  Sharpe (Taker 5 bps):  {test_metrics['sharpe_taker']:.4f}", "INFO")
    logger(f"  Sharpe (Maker 0 bps):  {test_metrics['sharpe_maker_neutral']:.4f}", "INFO")
    logger(f"  Sharpe (Maker -2.5 bps): {test_metrics['sharpe_maker_rebate']:.4f}", "INFO")
    logger(f"  Activity: {test_metrics['activity']:.2%}", "INFO")
    logger(f"  Avg PnL: {test_metrics['avg_pnl']:.6f}", "INFO")
```

---

## Artifact Naming Conventions

### Train Mode
- Transformers: `artifacts/feature_transformation/split_X/split_X_fitted_params.json`
- Scalers: `artifacts/feature_scale/split_X/split_X_fitted_params.json`
- VQ-VAE Models: `artifacts/vqvae_models/production/split_X_model.pth`
- PPO Models: `artifacts/ppo_training/experiment_X/split_Y/checkpoint_best.pth`

### Test Mode
- VQ-VAE Models: `artifacts/vqvae_models/test/split_0_full_model.pth`
- PPO Models: `artifacts/ppo_training/test/experiment_X/split_0_full_model.pth`
- Test Results: `artifacts/ppo_training/test/experiment_X/test_results.csv`

---

## Collection Naming Conventions

### Train Mode
- Input: `split_X_input` (materialized splits)
- Working: `split_X_output` (intermediate transformations)
- Final: `split_X_input` (after swap)
- Synthetic: `split_X_synthetic`

### Test Mode
- Input: `test_data` (from materialization stage 06)
- Working: `test_data_transformed`, `test_data_standardized`, `test_data_encoded` (intermediate)
- Final: `test_data` (after each swap, progressively enriched)

---

## Testing the Pipeline

### Train Mode Test
```bash
cd /home/user/drl-lob
# Edit scripts/run_pipeline.py: set mode='train', start_from=2, stop_at=21
python scripts/run_pipeline.py
```

Expected output:
- Stages 2-6: Data preparation (creates split_X + test_data)
- Stages 7-14: Feature preprocessing on split_X
- Stages 15-21: Model training/validation on split_X
- Artifacts saved to `artifacts/`

### Test Mode Test
```bash
cd /home/user/drl-lob
# Edit scripts/run_pipeline.py: set mode='test', start_from=9, stop_at=21
python scripts/run_pipeline.py
```

Expected output:
- Stages 9, 12, 14: Apply preprocessing to test_data
- Stage 16: Train VQ-VAE on full split_0, encode test_data
- Stage 21: Train PPO on full split_0, evaluate on test_data
- Test metrics logged to `artifacts/ppo_training/test/`

---

## Summary of Changes

**Completed:**
1. ✅ `run_pipeline.py` - Train/test mode orchestration

**Pending:**
2. ⏳ `08_apply_feature_transforms.py` - Apply fitted transformers to test_data
3. ⏳ `11_apply_feature_standardization.py` - Apply fitted scalers to test_data
4. ⏳ `12_filter_nulls.py` - Filter test_data
5. ⏳ `14_vqvae_production.py` - Train on full split, encode test_data
6. ⏳ `18_ppo_training.py` - Train on full split, evaluate on test_data
7. ⏳ Rename `19_generalization_validation.py` → indicate it's stage 20

---

## Key Implementation Principles

1. **No data leakage:** Test mode only APPLIES fitted artifacts from train mode, never fits new ones
2. **Same hyperparameters:** Test mode uses best configs from train mode (max_epochs, patience, architecture)
3. **Full training data:** Test mode trains on split_0 with train+val combined (no role filter)
4. **Same metrics:** Test mode logs same metrics as train mode (Sharpe for all fee scenarios)
5. **Artifact validation:** Test mode validates required artifacts exist before running
6. **Collection swaps:** Test mode uses same swap pattern as train mode for consistency

---

## Next Steps

1. Implement test mode for each pending stage script (2-6)
2. Test train mode end-to-end (stages 2-21)
3. Test test mode end-to-end (stages 9-21)
4. Validate final test metrics are reasonable
5. Document test results

---

Last updated: 2026-01-02
