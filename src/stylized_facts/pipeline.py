"""
Streaming Stylized Facts Testing Pipeline

Orchestrates testing per split with memory-efficient streaming approach.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List
from pyspark.sql import SparkSession

from src.utils.logging import logger
from .data_extractor import StreamingWindowExtractor
from .statistical_tests import FeaturePreprocessor, StylizedFactsTests


class StreamingStylizedFactsPipeline:
    """
    Tests windows one at a time, writing results immediately to disk.
    
    Memory-efficient approach:
    1. Load ONE window at a time (streaming, hour-by-hour)
    2. Run tests on that window
    3. Save results immediately
    4. Free memory
    5. Repeat for next window
    """
    
    def __init__(
        self, 
        spark: SparkSession,
        db_name: str,
        forecast_horizon: int,
        output_dir: Path,
        significance_level: float = 0.05
    ):
        """
        Initialize streaming pipeline.
        
        Args:
            spark: SparkSession instance
            db_name: MongoDB database name
            forecast_horizon: Forecast horizon for stride calculation
            output_dir: Directory to save results
            significance_level: Significance level for statistical tests
        """
        self.spark = spark
        self.db_name = db_name
        self.forecast_horizon = forecast_horizon
        self.output_dir = Path(output_dir)
        self.significance_level = significance_level
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.extractor = StreamingWindowExtractor(spark, db_name)
        self.preprocessor = FeaturePreprocessor(forecast_horizon)
        self.tester = StylizedFactsTests(significance_level)
    
    def _get_windows_for_split(self, metadata: Dict, split_id: int) -> List[Dict]:
        """
        Get windows that belong to a specific split.
        
        SIMPLIFIED: Windows now have explicit split_id, so just filter directly.
        
        Args:
            metadata: Metadata dictionary from YAML
            split_id: Split ID
            
        Returns:
            List of window metadata dicts for this split
        """
        all_windows = metadata.get('stylized_facts_windows', {}).get('windows', [])
        
        # Filter windows by split_id (windows are already tagged with split_id and are train only)
        # Handle both dictionary and object attribute access
        split_windows = []
        for w in all_windows:
            window_split_id = w.get('split_id') if isinstance(w, dict) else getattr(w, 'split_id', None)
            if window_split_id == split_id:
                split_windows.append(w)
        
        return split_windows
    
    def test_split(
        self, 
        split_id: int, 
        windows: List[Dict]
    ) -> Path:
        """
        Test all windows for one split, processing one window at a time.
        
        Args:
            split_id: Split ID
            windows: List of window metadata dicts
            
        Returns:
            Path to results CSV file
        """
        collection_name = f"split_{split_id}_input"
        
        logger("=" * 80, "INFO")
        logger(f"TESTING SPLIT {split_id}", "INFO")
        logger("=" * 80, "INFO")
        logger(f"Collection: {collection_name}", "INFO")
        logger(f"Windows: {len(windows)}", "INFO")
        logger("=" * 80, "INFO")
        
        # Results file for this split
        split_results_path = self.output_dir / f"split_{split_id}_results.csv"
        
        if not windows:
            logger(f"No windows for split {split_id}", "WARNING")
            return split_results_path
        
        # Process windows one at a time (streaming)
        window_count = 0
        
        for i, window in enumerate(windows, 1):
            fold_id = window['fold_id']
            fold_type = window['fold_type']
            
            logger(f"Window {i}/{len(windows)}: Fold {fold_id} ({fold_type})", "INFO")
            
            try:
                # 1. Extract THIS window only (streaming, hour-by-hour)
                window_df = self.extractor.extract_window_streaming(
                    collection_name, 
                    window
                )
                
                if window_df.empty:
                    logger(f"  No data found for window", "WARNING")
                    continue
                
                logger(f"  Loaded {len(window_df):,} samples", "INFO")
                
                # Check for null values
                null_counts = window_df.isnull().sum()
                total_nulls = null_counts.sum()
                if total_nulls > 0:
                    logger(f"  Found {total_nulls:,} null values across all features", "INFO")
                
                # 2. Classify features (only once, first window)
                if not self.preprocessor.return_features and not self.preprocessor.other_features:
                    self.preprocessor.classify_features(window_df)
                    logger(f"  Classified {len(self.preprocessor.return_features)} return features "
                          f"and {len(self.preprocessor.other_features)} other features", "INFO")
                
                # 3. Test return features (with per-feature stride)
                tested_features = 0
                
                if self.preprocessor.return_features:
                    logger(f"  Testing return features (with per-feature stride)...", "INFO")
                    
                    # Get per-feature strided DataFrames
                    feature_dfs = self.preprocessor.apply_stride_per_feature(
                        window_df, 
                        self.preprocessor.return_features
                    )
                    
                    # Test each feature individually
                    for feature, feature_df in feature_dfs.items():
                        if feature in feature_df.columns and len(feature_df) > 0:
                            self.tester.run_all_tests(
                                feature_df[feature], 
                                fold_id, 
                                fold_type, 
                                feature
                            )
                            tested_features += 1
                
                # 4. Test other features (no stride)
                if self.preprocessor.other_features:
                    logger(f"  Testing other features (no stride)...", "INFO")
                    
                    for feature in self.preprocessor.other_features:
                        if feature in window_df.columns:
                            self.tester.run_all_tests(
                                window_df[feature], 
                                fold_id, 
                                fold_type, 
                                feature
                            )
                            tested_features += 1
                
                logger(f"  Tested {tested_features} features", "INFO")
                
                # 5. Save results immediately
                if self.tester.test_results:
                    window_results = pd.DataFrame(self.tester.test_results)
                    window_results['split_id'] = split_id

                    # Append to file (or create if first window)
                    write_mode = 'w' if window_count == 0 else 'a'
                    write_header = (window_count == 0)

                    window_results.to_csv(
                        split_results_path,
                        mode=write_mode,
                        header=write_header,
                        quoting=1,
                        escapechar='\\',
                        index=False
                    )

                    logger(f"  Saved {len(window_results):,} test results (mode={write_mode})", "INFO")
                    
                    # Clear test results for next window
                    self.tester.test_results = []
                    window_count += 1
                
                # 6. Free memory
                del window_df
                if 'returns_df' in locals():
                    del returns_df
                if 'window_results' in locals():
                    del window_results
            
            except Exception as e:
                logger(f"  Error processing window: {e}", "ERROR")
                import traceback
                logger(traceback.format_exc(), "DEBUG")
                continue
        
        logger("=" * 80, "INFO")
        logger(f"Split {split_id} complete: {window_count} windows tested", "INFO")
        logger(f"Results saved to: {split_results_path}", "INFO")
        logger("=" * 80, "INFO")
        
        return split_results_path
    
    def test_all_splits(
        self, 
        metadata: Dict,
        split_ids: List[int]
    ) -> List[Path]:
        """
        Test all splits, one at a time.
        
        Args:
            metadata: Metadata dictionary from YAML
            split_ids: List of split IDs to test
            
        Returns:
            List of paths to result CSV files
        """
        logger("", "INFO")
        logger("=" * 80, "INFO")
        logger("STYLIZED FACTS TESTING - ALL SPLITS", "INFO")
        logger("=" * 80, "INFO")
        logger(f"Splits to test: {split_ids}", "INFO")
        logger(f"Forecast horizon: {self.forecast_horizon}", "INFO")
        logger(f"Significance level: {self.significance_level}", "INFO")
        logger(f"Output directory: {self.output_dir}", "INFO")
        logger("=" * 80, "INFO")
        logger("", "INFO")
        
        all_results_paths = []
        
        for split_id in split_ids:
            # Get windows for this split
            windows = self._get_windows_for_split(metadata, split_id)
            
            if not windows:
                logger(f"No windows found for split {split_id}, skipping", "WARNING")
                continue
            
            # Test split (streaming, one window at a time)
            results_path = self.test_split(split_id, windows)
            
            if results_path.exists():
                all_results_paths.append(results_path)
            
            logger("", "INFO")
        
        logger("=" * 80, "INFO")
        logger(f"ALL SPLITS COMPLETE", "INFO")
        logger(f"Tested {len(all_results_paths)} splits", "INFO")
        logger("=" * 80, "INFO")
        
        return all_results_paths