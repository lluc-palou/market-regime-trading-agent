"""
Results Aggregator

Combines and summarizes test results across all splits.
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from src.utils.logging import logger


class ResultsAggregator:
    """
    Aggregates stylized facts test results across splits.
    
    Creates summary statistics and identifies violations.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize aggregator.
        
        Args:
            output_dir: Directory containing split result files
        """
        self.output_dir = Path(output_dir)
    
    def aggregate_from_files(self, results_paths: List[Path]) -> pd.DataFrame:
        """
        Load and combine results from multiple CSV files.
        
        Args:
            results_paths: List of paths to split result CSVs
            
        Returns:
            Combined DataFrame with all results
        """
        logger("Aggregating results from all splits...", "INFO")
        
        chunks = []
        for path in results_paths:
            if path.exists():
                chunk = pd.read_csv(path)
                chunks.append(chunk)
                logger(f"  Loaded {len(chunk):,} results from {path.name}", "INFO")
        
        if not chunks:
            logger("No results to aggregate", "WARNING")
            return pd.DataFrame()
        
        # Combine all
        combined = pd.concat(chunks, ignore_index=True)
        
        logger(f"Combined {len(combined):,} total test results", "INFO")
        logger(f"  Features: {combined['feature_name'].nunique()}", "INFO")
        logger(f"  Folds: {combined['fold_id'].nunique()}", "INFO")
        logger(f"  Splits: {combined['split_id'].nunique()}", "INFO")
        
        return combined
    
    def compute_summary_statistics(self, combined_df: pd.DataFrame) -> Dict:
        """
        Compute summary statistics across all tests.
        
        Args:
            combined_df: Combined results DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        logger("Computing summary statistics...", "INFO")
        
        summary = {
            'metadata': {
                'total_tests': len(combined_df),
                'total_features': int(combined_df['feature_name'].nunique()),
                'total_folds': int(combined_df['fold_id'].nunique()),
                'total_splits': int(combined_df['split_id'].nunique()),
                'test_categories': list(combined_df['test_category'].unique()),
                'fold_types': list(combined_df['fold_type'].unique()),
                'timestamp': datetime.now().isoformat()
            },
            'by_test_category': {},
            'by_feature': {},
            'by_fold_type': {}
        }
        
        # Aggregate by test category
        for category in combined_df['test_category'].unique():
            cat_data = combined_df[combined_df['test_category'] == category]
            
            # Count tests with p-values
            pvalue_data = cat_data[cat_data['p_value'].notna()]
            
            summary['by_test_category'][category] = {
                'n_tests': int(len(cat_data)),
                'mean_p_value': float(pvalue_data['p_value'].mean()) if len(pvalue_data) > 0 else None,
                'median_p_value': float(pvalue_data['p_value'].median()) if len(pvalue_data) > 0 else None,
                'significant_pct': float((pvalue_data['p_value'] < 0.05).mean() * 100) if len(pvalue_data) > 0 else None
            }
        
        # Aggregate by feature
        for feature in combined_df['feature_name'].unique():
            feat_data = combined_df[combined_df['feature_name'] == feature]
            
            # Stationarity (ADF test)
            adf_data = feat_data[feat_data['test_name'] == 'ADF']
            stationary_pct = float((adf_data['p_value'] < 0.05).mean() * 100) if len(adf_data) > 0 else None
            
            # Normality (Jarque-Bera test)
            jb_data = feat_data[feat_data['test_name'] == 'Jarque-Bera']
            normal_pct = float((jb_data['p_value'] > 0.05).mean() * 100) if len(jb_data) > 0 else None
            
            # Autocorrelation (Ljung-Box test)
            lb_data = feat_data[feat_data['test_name'] == 'Ljung-Box']
            autocorr_pct = float((lb_data['p_value'] < 0.05).mean() * 100) if len(lb_data) > 0 else None
            
            # Kurtosis
            kurt_data = feat_data[feat_data['test_name'] == 'Moments']
            mean_kurtosis = float(kurt_data['kurtosis'].mean()) if 'kurtosis' in kurt_data.columns and len(kurt_data) > 0 else None
            
            summary['by_feature'][feature] = {
                'n_tests': int(len(feat_data)),
                'n_windows': int(feat_data['fold_id'].nunique()),
                'stationary_pct': stationary_pct,
                'normal_pct': normal_pct,
                'autocorr_pct': autocorr_pct,
                'mean_kurtosis': mean_kurtosis
            }
        
        # Aggregate by fold type
        for fold_type in combined_df['fold_type'].unique():
            ft_data = combined_df[combined_df['fold_type'] == fold_type]
            
            summary['by_fold_type'][fold_type] = {
                'n_tests': int(len(ft_data)),
                'n_features': int(ft_data['feature_name'].nunique()),
                'n_folds': int(ft_data['fold_id'].nunique())
            }
        
        return summary
    
    def identify_violations(
        self, 
        combined_df: pd.DataFrame,
        thresholds: Dict = None
    ) -> Dict:
        """
        Identify features that violate key assumptions.
        
        Args:
            combined_df: Combined results DataFrame
            thresholds: Custom thresholds (optional)
            
        Returns:
            Dictionary with violations
        """
        if thresholds is None:
            thresholds = {
                'stationary_threshold': 50,  # % of windows that should be stationary
                'normal_threshold': 30,      # % of windows that should be normal
                'kurtosis_threshold': 5,     # Kurtosis value for fat tails
                'autocorr_threshold': 30     # % of windows that can have autocorrelation
            }
        
        logger("Identifying violations...", "INFO")
        
        violations = {
            'non_stationary_features': [],
            'non_normal_features': [],
            'fat_tailed_features': [],
            'autocorrelated_features': []
        }
        
        for feature in combined_df['feature_name'].unique():
            feat_data = combined_df[combined_df['feature_name'] == feature]
            
            # Check stationarity
            adf_data = feat_data[feat_data['test_name'] == 'ADF']
            if len(adf_data) > 0:
                stationary_pct = (adf_data['p_value'] < 0.05).mean() * 100
                if stationary_pct < thresholds['stationary_threshold']:
                    violations['non_stationary_features'].append({
                        'feature': feature,
                        'stationary_pct': float(stationary_pct)
                    })
            
            # Check normality
            jb_data = feat_data[feat_data['test_name'] == 'Jarque-Bera']
            if len(jb_data) > 0:
                normal_pct = (jb_data['p_value'] > 0.05).mean() * 100
                if normal_pct < thresholds['normal_threshold']:
                    violations['non_normal_features'].append({
                        'feature': feature,
                        'normal_pct': float(normal_pct)
                    })
            
            # Check fat tails
            kurt_data = feat_data[feat_data['test_name'] == 'Moments']
            if 'kurtosis' in kurt_data.columns and len(kurt_data) > 0:
                mean_kurtosis = kurt_data['kurtosis'].mean()
                if mean_kurtosis > thresholds['kurtosis_threshold']:
                    violations['fat_tailed_features'].append({
                        'feature': feature,
                        'mean_kurtosis': float(mean_kurtosis)
                    })
            
            # Check autocorrelation
            lb_data = feat_data[feat_data['test_name'] == 'Ljung-Box']
            if len(lb_data) > 0:
                autocorr_pct = (lb_data['p_value'] < 0.05).mean() * 100
                if autocorr_pct > thresholds['autocorr_threshold']:
                    violations['autocorrelated_features'].append({
                        'feature': feature,
                        'autocorr_pct': float(autocorr_pct)
                    })
        
        return violations
    
    def create_summary_report(
        self, 
        combined_df: pd.DataFrame,
        summary_stats: Dict,
        violations: Dict
    ) -> str:
        """
        Create human-readable summary report.
        
        Args:
            combined_df: Combined results DataFrame
            summary_stats: Summary statistics dict
            violations: Violations dict
            
        Returns:
            Report text
        """
        lines = []
        lines.append("=" * 80)
        lines.append("STYLIZED FACTS TESTING SUMMARY")
        lines.append("=" * 80)
        lines.append("")
        
        # Metadata
        meta = summary_stats['metadata']
        lines.append(f"Date: {meta['timestamp']}")
        lines.append(f"Total tests: {meta['total_tests']:,}")
        lines.append(f"Features tested: {meta['total_features']}")
        lines.append(f"Folds tested: {meta['total_folds']}")
        lines.append(f"Splits tested: {meta['total_splits']}")
        lines.append("")
        
        # Test categories
        lines.append("=" * 80)
        lines.append("TESTS BY CATEGORY")
        lines.append("=" * 80)
        for category, stats in summary_stats['by_test_category'].items():
            lines.append(f"{category}:")
            lines.append(f"  Tests: {stats['n_tests']:,}")
            if stats['mean_p_value'] is not None:
                lines.append(f"  Mean p-value: {stats['mean_p_value']:.4f}")
                lines.append(f"  Significant (α=0.05): {stats['significant_pct']:.1f}%")
        lines.append("")
        
        # Violations
        lines.append("=" * 80)
        lines.append("KEY FINDINGS & VIOLATIONS")
        lines.append("=" * 80)
        
        if violations['non_stationary_features']:
            lines.append(f"\n⚠ Non-stationary features ({len(violations['non_stationary_features'])}):")
            for v in violations['non_stationary_features'][:10]:
                lines.append(f"  - {v['feature']}: {v['stationary_pct']:.1f}% stationary")
            if len(violations['non_stationary_features']) > 10:
                lines.append(f"  ... and {len(violations['non_stationary_features']) - 10} more")
        
        if violations['non_normal_features']:
            lines.append(f"\n⚠ Non-normal features ({len(violations['non_normal_features'])}):")
            for v in violations['non_normal_features'][:10]:
                lines.append(f"  - {v['feature']}: {v['normal_pct']:.1f}% normal")
            if len(violations['non_normal_features']) > 10:
                lines.append(f"  ... and {len(violations['non_normal_features']) - 10} more")
        
        if violations['fat_tailed_features']:
            lines.append(f"\n⚠ Fat-tailed features ({len(violations['fat_tailed_features'])}):")
            for v in violations['fat_tailed_features'][:10]:
                lines.append(f"  - {v['feature']}: kurtosis = {v['mean_kurtosis']:.2f}")
            if len(violations['fat_tailed_features']) > 10:
                lines.append(f"  ... and {len(violations['fat_tailed_features']) - 10} more")
        
        if violations['autocorrelated_features']:
            lines.append(f"\n⚠ Autocorrelated features ({len(violations['autocorrelated_features'])}):")
            for v in violations['autocorrelated_features'][:10]:
                lines.append(f"  - {v['feature']}: {v['autocorr_pct']:.1f}% with autocorrelation")
            if len(violations['autocorrelated_features']) > 10:
                lines.append(f"  ... and {len(violations['autocorrelated_features']) - 10} more")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def save_summary(
        self, 
        combined_df: pd.DataFrame,
        summary_stats: Dict,
        violations: Dict
    ):
        """
        Save all summary outputs.
        
        Args:
            combined_df: Combined results DataFrame
            summary_stats: Summary statistics dict
            violations: Violations dict
        """
        logger("Saving summary outputs...", "INFO")
        
        # Save combined results
        combined_path = self.output_dir / "summary_all_splits.csv"
        combined_df.to_csv(combined_path, index=False)
        logger(f"  Saved combined results: {combined_path}", "INFO")
        
        # Save summary statistics JSON
        stats_path = self.output_dir / "summary_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2)
        logger(f"  Saved summary statistics: {stats_path}", "INFO")
        
        # Save violations JSON
        violations_path = self.output_dir / "violations.json"
        with open(violations_path, 'w', encoding='utf-8') as f:
            json.dump(violations, f, indent=2)
        logger(f"  Saved violations: {violations_path}", "INFO")
        
        # Save text report
        report = self.create_summary_report(combined_df, summary_stats, violations)
        report_path = self.output_dir / "summary_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger(f"  Saved text report: {report_path}", "INFO")
        
        # Also print report
        print("\n" + report)