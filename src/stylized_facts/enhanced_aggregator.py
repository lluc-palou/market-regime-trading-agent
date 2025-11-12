"""
Enhanced Results Aggregator with Statistical Confidence Measures

Extends the base ResultsAggregator to provide:
- Mean and variance of pass rates (instead of just percentages)
- Confidence intervals for statistical tests
- Standard errors and uncertainty quantification
- Bootstrap confidence intervals for robust estimation

This provides better understanding of:
1. How consistent are the test results across different windows/splits?
2. What is the uncertainty in our pass rate estimates?
3. Which features have high variance in their test results (unstable)?

Outputs:
- enhanced_summary_statistics.json: Full statistical measures
- enhanced_summary_by_feature.csv: Per-feature statistics (easy to analyze)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from scipy import stats

from src.utils.logging import logger


class EnhancedResultsAggregator:
    """
    Enhanced aggregator with statistical confidence measures.
    
    Provides mean, variance, confidence intervals, and uncertainty quantification
    for all stylized facts test results.
    """
    
    def __init__(self, output_dir: Path, confidence_level: float = 0.95):
        """
        Initialize enhanced aggregator.
        
        Args:
            output_dir: Directory containing split result files
            confidence_level: Confidence level for intervals (default 0.95 = 95%)
        """
        self.output_dir = Path(output_dir)
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
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
    
    def _compute_proportion_confidence_interval(
        self, 
        successes: int, 
        n_trials: int
    ) -> Tuple[float, float, float, float]:
        """
        Compute confidence interval for a proportion using Wilson score method.
        
        Also returns standard error and margin of error.
        
        Args:
            successes: Number of successes
            n_trials: Total number of trials
            
        Returns:
            Tuple of (proportion, std_error, ci_lower, ci_upper)
        """
        if n_trials == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        p = successes / n_trials
        
        # Wilson score confidence interval (better for proportions near 0 or 1)
        z = stats.norm.ppf(1 - self.alpha / 2)  # z-score for confidence level
        
        denominator = 1 + z**2 / n_trials
        centre = (p + z**2 / (2 * n_trials)) / denominator
        margin = z * np.sqrt((p * (1 - p) / n_trials + z**2 / (4 * n_trials**2))) / denominator
        
        ci_lower = max(0.0, centre - margin)
        ci_upper = min(1.0, centre + margin)
        
        # Standard error for the proportion
        std_error = np.sqrt(p * (1 - p) / n_trials) if n_trials > 0 else 0.0
        
        return p, std_error, ci_lower, ci_upper
    
    def _compute_mean_confidence_interval(
        self,
        values: np.ndarray
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute mean and confidence interval for continuous values.
        
        Args:
            values: Array of values
            
        Returns:
            Tuple of (mean, std, std_error, ci_lower, ci_upper)
        """
        if len(values) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # Sample standard deviation
        n = len(values)
        std_error = std / np.sqrt(n)
        
        # t-distribution for confidence interval (more appropriate for small samples)
        t_crit = stats.t.ppf(1 - self.alpha / 2, df=n-1) if n > 1 else 0
        margin = t_crit * std_error
        
        ci_lower = mean - margin
        ci_upper = mean + margin
        
        return mean, std, std_error, ci_lower, ci_upper
    
    def compute_enhanced_statistics(self, combined_df: pd.DataFrame) -> Dict:
        """
        Compute enhanced summary statistics with confidence intervals.
        
        Args:
            combined_df: Combined results DataFrame
            
        Returns:
            Dictionary with enhanced summary statistics
        """
        logger("Computing enhanced summary statistics with confidence intervals...", "INFO")
        
        summary = {
            'metadata': {
                'total_tests': len(combined_df),
                'total_features': int(combined_df['feature_name'].nunique()),
                'total_folds': int(combined_df['fold_id'].nunique()),
                'total_splits': int(combined_df['split_id'].nunique()),
                'test_categories': list(combined_df['test_category'].unique()),
                'fold_types': list(combined_df['fold_type'].unique()),
                'confidence_level': self.confidence_level,
                'timestamp': datetime.now().isoformat()
            },
            'by_test_category': {},
            'by_feature': {}
        }
        
        # =====================================================================
        # AGGREGATE BY TEST CATEGORY
        # =====================================================================
        
        logger("  Computing statistics by test category...", "INFO")
        
        for category in combined_df['test_category'].unique():
            cat_data = combined_df[combined_df['test_category'] == category]
            pvalue_data = cat_data[cat_data['p_value'].notna()]
            
            if len(pvalue_data) > 0:
                # Mean p-value statistics
                p_values = pvalue_data['p_value'].values
                mean_p, std_p, se_p, ci_lower_p, ci_upper_p = self._compute_mean_confidence_interval(p_values)
                
                # Significance statistics (proportion of significant tests)
                significant_count = (p_values < 0.05).sum()
                n_tests = len(p_values)
                sig_prop, sig_se, sig_ci_lower, sig_ci_upper = self._compute_proportion_confidence_interval(
                    significant_count, n_tests
                )
                
                summary['by_test_category'][category] = {
                    'n_tests': n_tests,
                    'p_value_stats': {
                        'mean': float(mean_p),
                        'std': float(std_p),
                        'std_error': float(se_p),
                        'ci_lower': float(ci_lower_p),
                        'ci_upper': float(ci_upper_p),
                        'median': float(np.median(p_values))
                    },
                    'significance_stats': {
                        'proportion_significant': float(sig_prop),
                        'percentage_significant': float(sig_prop * 100),
                        'std_error': float(sig_se),
                        'ci_lower': float(sig_ci_lower),
                        'ci_upper': float(sig_ci_upper),
                        'ci_lower_pct': float(sig_ci_lower * 100),
                        'ci_upper_pct': float(sig_ci_upper * 100)
                    }
                }
        
        # =====================================================================
        # AGGREGATE BY FEATURE (MOST IMPORTANT)
        # =====================================================================
        
        logger("  Computing statistics by feature...", "INFO")
        
        for feature in combined_df['feature_name'].unique():
            feat_data = combined_df[combined_df['feature_name'] == feature]
            
            feature_stats = {
                'n_tests': int(len(feat_data)),
                'n_windows': int(feat_data['fold_id'].nunique()),
                'n_splits': int(feat_data['split_id'].nunique())
            }
            
            # -----------------------------------------------------------------
            # STATIONARITY (ADF Test) - Reject null = stationary
            # -----------------------------------------------------------------
            adf_data = feat_data[feat_data['test_name'] == 'ADF']
            if len(adf_data) > 0:
                stationary_count = (adf_data['p_value'] < 0.05).sum()
                n_adf = len(adf_data)
                stat_prop, stat_se, stat_ci_lower, stat_ci_upper = self._compute_proportion_confidence_interval(
                    stationary_count, n_adf
                )
                
                # Also compute statistics on the ADF p-values themselves
                adf_pvals = adf_data['p_value'].values
                mean_adf, std_adf, se_adf, ci_lower_adf, ci_upper_adf = self._compute_mean_confidence_interval(adf_pvals)
                
                feature_stats['stationarity'] = {
                    'n_tests': n_adf,
                    'proportion_stationary': float(stat_prop),
                    'percentage_stationary': float(stat_prop * 100),
                    'std_error': float(stat_se),
                    'ci_lower': float(stat_ci_lower),
                    'ci_upper': float(stat_ci_upper),
                    'ci_lower_pct': float(stat_ci_lower * 100),
                    'ci_upper_pct': float(stat_ci_upper * 100),
                    'p_value_stats': {
                        'mean': float(mean_adf),
                        'std': float(std_adf),
                        'std_error': float(se_adf),
                        'ci_lower': float(ci_lower_adf),
                        'ci_upper': float(ci_upper_adf)
                    }
                }
            
            # -----------------------------------------------------------------
            # NORMALITY (Jarque-Bera Test) - Fail to reject null = normal
            # -----------------------------------------------------------------
            jb_data = feat_data[feat_data['test_name'] == 'Jarque-Bera']
            if len(jb_data) > 0:
                normal_count = (jb_data['p_value'] > 0.05).sum()
                n_jb = len(jb_data)
                norm_prop, norm_se, norm_ci_lower, norm_ci_upper = self._compute_proportion_confidence_interval(
                    normal_count, n_jb
                )
                
                jb_pvals = jb_data['p_value'].values
                mean_jb, std_jb, se_jb, ci_lower_jb, ci_upper_jb = self._compute_mean_confidence_interval(jb_pvals)
                
                feature_stats['normality'] = {
                    'n_tests': n_jb,
                    'proportion_normal': float(norm_prop),
                    'percentage_normal': float(norm_prop * 100),
                    'std_error': float(norm_se),
                    'ci_lower': float(norm_ci_lower),
                    'ci_upper': float(norm_ci_upper),
                    'ci_lower_pct': float(norm_ci_lower * 100),
                    'ci_upper_pct': float(norm_ci_upper * 100),
                    'p_value_stats': {
                        'mean': float(mean_jb),
                        'std': float(std_jb),
                        'std_error': float(se_jb),
                        'ci_lower': float(ci_lower_jb),
                        'ci_upper': float(ci_upper_jb)
                    }
                }
            
            # -----------------------------------------------------------------
            # AUTOCORRELATION (Ljung-Box Test) - Reject null = has autocorr
            # -----------------------------------------------------------------
            lb_data = feat_data[feat_data['test_name'] == 'Ljung-Box']
            if len(lb_data) > 0:
                autocorr_count = (lb_data['p_value'] < 0.05).sum()
                n_lb = len(lb_data)
                ac_prop, ac_se, ac_ci_lower, ac_ci_upper = self._compute_proportion_confidence_interval(
                    autocorr_count, n_lb
                )
                
                lb_pvals = lb_data['p_value'].values
                mean_lb, std_lb, se_lb, ci_lower_lb, ci_upper_lb = self._compute_mean_confidence_interval(lb_pvals)
                
                feature_stats['autocorrelation'] = {
                    'n_tests': n_lb,
                    'proportion_autocorrelated': float(ac_prop),
                    'percentage_autocorrelated': float(ac_prop * 100),
                    'std_error': float(ac_se),
                    'ci_lower': float(ac_ci_lower),
                    'ci_upper': float(ac_ci_upper),
                    'ci_lower_pct': float(ac_ci_lower * 100),
                    'ci_upper_pct': float(ac_ci_upper * 100),
                    'p_value_stats': {
                        'mean': float(mean_lb),
                        'std': float(std_lb),
                        'std_error': float(se_lb),
                        'ci_lower': float(ci_lower_lb),
                        'ci_upper': float(ci_upper_lb)
                    }
                }
            
            # -----------------------------------------------------------------
            # KURTOSIS (from Moments test)
            # -----------------------------------------------------------------
            kurt_data = feat_data[feat_data['test_name'] == 'Moments']
            if 'kurtosis' in kurt_data.columns and len(kurt_data) > 0:
                kurt_values = kurt_data['kurtosis'].dropna().values
                if len(kurt_values) > 0:
                    mean_kurt, std_kurt, se_kurt, ci_lower_kurt, ci_upper_kurt = self._compute_mean_confidence_interval(kurt_values)
                    
                    feature_stats['kurtosis'] = {
                        'n_measurements': len(kurt_values),
                        'mean': float(mean_kurt),
                        'std': float(std_kurt),
                        'std_error': float(se_kurt),
                        'ci_lower': float(ci_lower_kurt),
                        'ci_upper': float(ci_upper_kurt),
                        'median': float(np.median(kurt_values)),
                        'min': float(np.min(kurt_values)),
                        'max': float(np.max(kurt_values))
                    }
            
            # -----------------------------------------------------------------
            # SKEWNESS (from Moments test)
            # -----------------------------------------------------------------
            if 'skewness' in kurt_data.columns and len(kurt_data) > 0:
                skew_values = kurt_data['skewness'].dropna().values
                if len(skew_values) > 0:
                    mean_skew, std_skew, se_skew, ci_lower_skew, ci_upper_skew = self._compute_mean_confidence_interval(skew_values)
                    
                    feature_stats['skewness'] = {
                        'n_measurements': len(skew_values),
                        'mean': float(mean_skew),
                        'std': float(std_skew),
                        'std_error': float(se_skew),
                        'ci_lower': float(ci_lower_skew),
                        'ci_upper': float(ci_upper_skew),
                        'median': float(np.median(skew_values))
                    }
            
            summary['by_feature'][feature] = feature_stats
        
        return summary
    
    def create_feature_summary_dataframe(self, summary_stats: Dict) -> pd.DataFrame:
        """
        Convert the by_feature statistics into a clean DataFrame for easy analysis.
        
        Args:
            summary_stats: Enhanced summary statistics dictionary
            
        Returns:
            DataFrame with one row per feature, columns for all statistics
        """
        logger("Creating feature summary DataFrame...", "INFO")
        
        rows = []
        
        for feature, stats in summary_stats['by_feature'].items():
            row = {
                'feature_name': feature,
                'n_tests': stats['n_tests'],
                'n_windows': stats['n_windows'],
                'n_splits': stats['n_splits']
            }
            
            # Stationarity
            if 'stationarity' in stats:
                st = stats['stationarity']
                row['stationary_pct'] = st['percentage_stationary']
                row['stationary_pct_ci_lower'] = st['ci_lower_pct']
                row['stationary_pct_ci_upper'] = st['ci_upper_pct']
                row['stationary_pct_stderr'] = st['std_error'] * 100
                row['stationary_pvalue_mean'] = st['p_value_stats']['mean']
                row['stationary_pvalue_std'] = st['p_value_stats']['std']
            
            # Normality
            if 'normality' in stats:
                nm = stats['normality']
                row['normal_pct'] = nm['percentage_normal']
                row['normal_pct_ci_lower'] = nm['ci_lower_pct']
                row['normal_pct_ci_upper'] = nm['ci_upper_pct']
                row['normal_pct_stderr'] = nm['std_error'] * 100
                row['normal_pvalue_mean'] = nm['p_value_stats']['mean']
                row['normal_pvalue_std'] = nm['p_value_stats']['std']
            
            # Autocorrelation
            if 'autocorrelation' in stats:
                ac = stats['autocorrelation']
                row['autocorr_pct'] = ac['percentage_autocorrelated']
                row['autocorr_pct_ci_lower'] = ac['ci_lower_pct']
                row['autocorr_pct_ci_upper'] = ac['ci_upper_pct']
                row['autocorr_pct_stderr'] = ac['std_error'] * 100
                row['autocorr_pvalue_mean'] = ac['p_value_stats']['mean']
                row['autocorr_pvalue_std'] = ac['p_value_stats']['std']
            
            # Kurtosis
            if 'kurtosis' in stats:
                kt = stats['kurtosis']
                row['kurtosis_mean'] = kt['mean']
                row['kurtosis_std'] = kt['std']
                row['kurtosis_ci_lower'] = kt['ci_lower']
                row['kurtosis_ci_upper'] = kt['ci_upper']
                row['kurtosis_stderr'] = kt['std_error']
            
            # Skewness
            if 'skewness' in stats:
                sk = stats['skewness']
                row['skewness_mean'] = sk['mean']
                row['skewness_std'] = sk['std']
                row['skewness_ci_lower'] = sk['ci_lower']
                row['skewness_ci_upper'] = sk['ci_upper']
                row['skewness_stderr'] = sk['std_error']
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by feature name
        df = df.sort_values('feature_name').reset_index(drop=True)
        
        return df
    
    def save_enhanced_summary(
        self,
        summary_stats: Dict,
        feature_df: pd.DataFrame
    ):
        """
        Save enhanced summary outputs.
        
        Args:
            summary_stats: Enhanced summary statistics dictionary
            feature_df: Feature summary DataFrame
        """
        logger("Saving enhanced summary outputs...", "INFO")
        
        # Save full JSON with all statistics
        json_path = self.output_dir / "enhanced_summary_statistics.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2)
        logger(f"  Saved enhanced JSON: {json_path}", "INFO")
        
        # Save feature DataFrame (easy to analyze in Excel/Python)
        csv_path = self.output_dir / "enhanced_summary_by_feature.csv"
        feature_df.to_csv(csv_path, index=False)
        logger(f"  Saved feature CSV: {csv_path}", "INFO")
        
        logger("Enhanced summary outputs saved successfully", "INFO")
    
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
        
        logger("Identifying violations with statistical confidence...", "INFO")
        
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
                stationary_count = (adf_data['p_value'] < 0.05).sum()
                n_adf = len(adf_data)
                stat_prop, stat_se, stat_ci_lower, stat_ci_upper = self._compute_proportion_confidence_interval(
                    stationary_count, n_adf
                )
                stationary_pct = stat_prop * 100
                
                if stationary_pct < thresholds['stationary_threshold']:
                    violations['non_stationary_features'].append({
                        'feature': feature,
                        'stationary_pct': float(stationary_pct),
                        'ci_lower_pct': float(stat_ci_lower * 100),
                        'ci_upper_pct': float(stat_ci_upper * 100),
                        'std_error_pct': float(stat_se * 100),
                        'n_tests': n_adf
                    })
            
            # Check normality
            jb_data = feat_data[feat_data['test_name'] == 'Jarque-Bera']
            if len(jb_data) > 0:
                normal_count = (jb_data['p_value'] > 0.05).sum()
                n_jb = len(jb_data)
                norm_prop, norm_se, norm_ci_lower, norm_ci_upper = self._compute_proportion_confidence_interval(
                    normal_count, n_jb
                )
                normal_pct = norm_prop * 100
                
                if normal_pct < thresholds['normal_threshold']:
                    violations['non_normal_features'].append({
                        'feature': feature,
                        'normal_pct': float(normal_pct),
                        'ci_lower_pct': float(norm_ci_lower * 100),
                        'ci_upper_pct': float(norm_ci_upper * 100),
                        'std_error_pct': float(norm_se * 100),
                        'n_tests': n_jb
                    })
            
            # Check fat tails
            kurt_data = feat_data[feat_data['test_name'] == 'Moments']
            if 'kurtosis' in kurt_data.columns and len(kurt_data) > 0:
                kurt_values = kurt_data['kurtosis'].dropna().values
                if len(kurt_values) > 0:
                    mean_kurt, std_kurt, se_kurt, ci_lower_kurt, ci_upper_kurt = self._compute_mean_confidence_interval(kurt_values)
                    
                    if mean_kurt > thresholds['kurtosis_threshold']:
                        violations['fat_tailed_features'].append({
                            'feature': feature,
                            'mean_kurtosis': float(mean_kurt),
                            'std_kurtosis': float(std_kurt),
                            'ci_lower': float(ci_lower_kurt),
                            'ci_upper': float(ci_upper_kurt),
                            'std_error': float(se_kurt),
                            'n_measurements': len(kurt_values)
                        })
            
            # Check autocorrelation
            lb_data = feat_data[feat_data['test_name'] == 'Ljung-Box']
            if len(lb_data) > 0:
                autocorr_count = (lb_data['p_value'] < 0.05).sum()
                n_lb = len(lb_data)
                ac_prop, ac_se, ac_ci_lower, ac_ci_upper = self._compute_proportion_confidence_interval(
                    autocorr_count, n_lb
                )
                autocorr_pct = ac_prop * 100
                
                if autocorr_pct > thresholds['autocorr_threshold']:
                    violations['autocorrelated_features'].append({
                        'feature': feature,
                        'autocorr_pct': float(autocorr_pct),
                        'ci_lower_pct': float(ac_ci_lower * 100),
                        'ci_upper_pct': float(ac_ci_upper * 100),
                        'std_error_pct': float(ac_se * 100),
                        'n_tests': n_lb
                    })
        
        return violations