"""
Stylized Facts Testing Module
Applies statistical tests to representative windows from each fold.

CORRECTED: Properly excludes metadata fields and only tests actual features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch.unitroot import PhillipsPerron
import warnings
warnings.filterwarnings('ignore')

from src.utils.logging import logger

# =================================================================================================
# Feature Classification & Preprocessing
# =================================================================================================

class FeaturePreprocessor:
    """
    Identifies feature types and applies stride to forward returns.
    
    CORRECTED: Properly excludes metadata columns from feature classification.
    Only processes actual features extracted from the 'features' array.
    """
    
    # Define metadata columns that should NEVER be tested
    METADATA_COLUMNS = {
        '_id',
        'timestamp',
        'timestamp_str',
        'fold_id',
        'fold_type',
        'split_roles',
        'role',
        'bins',
        'feature_names',  # This is just the list of names, not data
        'features'  # This is the raw array, already expanded to columns
    }
    
    # Define feature categories to EXCLUDE from testing
    EXCLUDED_FEATURE_PATTERNS = [
        'fwd_logret',  # Forward returns - don't test these
        'fwd_ret',
        'forward_return'
    ]
    
    # Minimum samples required for reliable statistical tests
    MIN_SAMPLES_FOR_TESTING = 30  # Standard minimum for most statistical tests
    
    def __init__(self, forecast_horizon_steps: int):
        """
        Args:
            forecast_horizon_steps: The forecast horizon (from metadata) used for stride
        """
        self.forecast_horizon = forecast_horizon_steps
        self.return_features = []
        self.other_features = []
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Extract actual feature columns, excluding all metadata and forward returns.
        
        FOCUS: Only past_logret (with stride) and lob features (no stride)
        EXCLUDE: fwd_logret features
        
        Args:
            df: DataFrame with features as columns
            
        Returns:
            List of feature column names (metadata and fwd_logret excluded)
        """
        # Get all columns except metadata
        candidate_cols = [
            col for col in df.columns 
            if col not in self.METADATA_COLUMNS
        ]
        
        # Exclude forward return features
        feature_cols = []
        excluded_fwd = []
        
        for col in candidate_cols:
            col_lower = col.lower()
            
            # Check if this is a forward return feature (to exclude)
            is_fwd_return = any(pattern in col_lower for pattern in self.EXCLUDED_FEATURE_PATTERNS)
            
            if is_fwd_return:
                excluded_fwd.append(col)
            else:
                feature_cols.append(col)
        
        if not feature_cols:
            logger("WARNING: No feature columns found after filtering metadata and forward returns!", "WARNING")
            logger(f"Available columns: {df.columns.tolist()}", "DEBUG")
        else:
            # Count how many metadata columns were actually present and excluded
            excluded_metadata_count = len(self.METADATA_COLUMNS & set(df.columns))
            logger(f"Found {len(feature_cols)} feature columns to test", "DEBUG")
            logger(f"  Excluded {excluded_metadata_count} metadata columns", "DEBUG")
            logger(f"  Excluded {len(excluded_fwd)} forward return features", "DEBUG")
        
        return feature_cols
    
    def _filter_low_sample_features(
        self, 
        df: pd.DataFrame, 
        features: List[str],
        min_samples: int = None
    ) -> Tuple[List[str], List[str]]:
        """
        Filter out features with insufficient non-null samples for testing.
        
        Args:
            df: DataFrame with feature data
            features: List of feature names to check
            min_samples: Minimum required samples (uses class default if None)
            
        Returns:
            Tuple of (valid_features, excluded_features)
        """
        if min_samples is None:
            min_samples = self.MIN_SAMPLES_FOR_TESTING
        
        valid_features = []
        excluded_features = []
        
        for feature in features:
            if feature not in df.columns:
                excluded_features.append((feature, 0, "not in dataframe"))
                continue
            
            # Count non-null samples
            non_null_count = df[feature].notna().sum()
            
            if non_null_count < min_samples:
                excluded_features.append((feature, non_null_count, f"< {min_samples} samples"))
            else:
                valid_features.append(feature)
        
        # Log exclusions
        if excluded_features:
            logger(f"  ⚠️  Excluded {len(excluded_features)} features with insufficient samples:", "WARNING")
            for feat, count, reason in excluded_features[:5]:  # Show first 5
                logger(f"      - {feat}: {count} samples ({reason})", "WARNING")
            if len(excluded_features) > 5:
                logger(f"      ... and {len(excluded_features) - 5} more", "WARNING")
        
        return valid_features, excluded_features
    
    def classify_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Classifies features into two categories:
        1. past_logret features (need stride to avoid overlapping lookbacks)
        2. lob features (no stride needed)
        
        Also filters out features with insufficient samples for testing.
        
        LOGIC:
        - past_logret features with lookback > forecast_horizon need stride
        - lob features are instantaneous snapshots, no stride needed
        - fwd_logret features are excluded entirely
        - Features with < MIN_SAMPLES_FOR_TESTING samples are excluded
        
        Args:
            df: DataFrame with features as columns
            
        Returns:
            Tuple of (past_logret_features, lob_features)
        """
        # Get only actual feature columns (no metadata, no forward returns)
        all_features = self._get_feature_columns(df)
        
        # First pass: categorize by feature type
        past_logret_candidates = []
        lob_candidates = []
        
        for feature in all_features:
            feature_lower = feature.lower()
            
            # Check if this is a past_logret feature
            is_past_logret = any(kw in feature_lower for kw in ['past_logret', 'past_ret', 'logret'])
            
            # Check if this is a lob feature
            is_lob = 'lob' in feature_lower
            
            if is_past_logret:
                # past_logret features need stride
                past_logret_candidates.append(feature)
            elif is_lob:
                # lob features are instantaneous, no stride needed
                lob_candidates.append(feature)
            else:
                # Any other features that aren't fwd_logret (already filtered)
                # Treat as lob-like (no stride)
                lob_candidates.append(feature)
        
        # Second pass: filter features with insufficient samples
        # Note: We check BEFORE applying stride for past_logret features
        self.return_features, excluded_past = self._filter_low_sample_features(
            df, past_logret_candidates
        )
        
        self.other_features, excluded_lob = self._filter_low_sample_features(
            df, lob_candidates
        )
        
        # Summary logging
        logger("", "INFO")
        logger(f"Feature Classification Summary:", "INFO")
        logger(f"  past_logret features (with stride={self.forecast_horizon}): {len(self.return_features)}", "INFO")
        logger(f"  lob features (no stride): {len(self.other_features)}", "INFO")
        
        total_excluded = len(excluded_past) + len(excluded_lob)
        if total_excluded > 0:
            logger(f"  Excluded features (low samples): {total_excluded}", "WARNING")
        
        if self.return_features:
            sample_past = self.return_features[:3]
            logger(f"  Sample past_logret: {sample_past}{'...' if len(self.return_features) > 3 else ''}", "DEBUG")
        
        if self.other_features:
            sample_lob = self.other_features[:3]
            logger(f"  Sample lob: {sample_lob}{'...' if len(self.other_features) > 3 else ''}", "DEBUG")
        
        return self.return_features, self.other_features
    
    def apply_stride_per_feature(
        self, 
        df: pd.DataFrame, 
        features: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply feature-specific stride based on lookback period.
        
        CORRECTED LOGIC:
        - past_logret_1: lookback=1, stride=1 → all samples independent
        - past_logret_10: lookback=10, stride=10 → no overlap
        - past_logret_240: lookback=240, stride=240 → no overlap
        
        Returns one DataFrame per feature with appropriate stride.
        
        Args:
            df: DataFrame with all samples
            features: List of past_logret feature names
            
        Returns:
            Dictionary mapping feature_name -> strided_dataframe
        """
        import re
        
        feature_dfs = {}
        
        for feature in features:
            # Extract lookback from feature name
            # Pattern: past_logret_10, past_ret_240, logret_5
            lookback_match = re.search(r'_(\d+)', feature)
            
            if lookback_match:
                lookback = int(lookback_match.group(1))
            else:
                # Default to forecast_horizon if can't extract
                logger(f"    ⚠️  Cannot extract lookback from '{feature}', using stride={self.forecast_horizon}", "WARNING")
                lookback = self.forecast_horizon
            
            # Stride = lookback to ensure no overlap
            stride = lookback
            
            # Apply stride
            if stride > 0:
                strided = df.iloc[::stride].copy()
                
                # Keep timestamp and this feature
                keep_cols = ['timestamp', feature] if feature in df.columns else ['timestamp']
                feature_df = strided[keep_cols].reset_index(drop=True)
                
                samples_before = len(df)
                samples_after = len(feature_df)
                
                logger(f"    {feature}: stride={stride}, {samples_before} → {samples_after} samples", "DEBUG")
                
                # Warn if still too few samples
                if samples_after < self.MIN_SAMPLES_FOR_TESTING:
                    logger(f"      ⚠️  Only {samples_after} samples after stride, less than {self.MIN_SAMPLES_FOR_TESTING}", "WARNING")
                
                feature_dfs[feature] = feature_df
            else:
                logger(f"    ⚠️  Invalid stride {stride} for {feature}", "WARNING")
        
        return feature_dfs
    
    def apply_stride(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        DEPRECATED: Use apply_stride_per_feature instead.
        
        This method applies a single stride to all features, which is incorrect
        for past_logret features with different lookback periods.
        
        Kept for backward compatibility but logs a warning.
        """
        logger("⚠️  WARNING: apply_stride() applies same stride to all features", "WARNING")
        logger("   Use apply_stride_per_feature() for correct per-feature stride", "WARNING")
        
        # Apply uniform stride (original behavior)
        strided_df = df.iloc[::self.forecast_horizon].copy()
        keep_cols = ['timestamp'] + [f for f in features if f in df.columns]
        result_df = strided_df[keep_cols].reset_index(drop=True)
        
        return result_df
    
    def prepare_windows(self, windows_dict: Dict[int, pd.DataFrame]) -> Tuple[Dict[int, Dict[str, pd.DataFrame]], Dict[int, pd.DataFrame]]:
        """
        Prepares windows by applying feature-specific stride to past_logret features.
        
        CORRECTED: Each past_logret feature gets its own stride based on lookback.
        
        Args:
            windows_dict: Dictionary mapping fold_id to window DataFrame
            
        Returns:
            returns_dict: Dict[fold_id -> Dict[feature_name -> strided_df]]
            features_dict: Dict[fold_id -> df with all lob features]
        """
        logger("="*80, "INFO")
        logger("PREPARING FEATURES FOR TESTING", "INFO")
        logger("="*80, "INFO")
        
        returns_dict = {}
        features_dict = {}
        
        for fold_id, df in windows_dict.items():
            # Classify features (only need to do once)
            if not self.return_features and not self.other_features:
                self.classify_features(df)
            
            # Apply per-feature stride to past_logret features
            if self.return_features:
                feature_dfs = self.apply_stride_per_feature(df, self.return_features)
                returns_dict[fold_id] = feature_dfs
                
                # Log summary
                total_samples = sum(len(feat_df) for feat_df in feature_dfs.values())
                logger(f"Fold {fold_id}: {len(self.return_features)} past_logret features with per-feature stride", "INFO")
                logger(f"  Total strided samples across all features: {total_samples}", "DEBUG")
            
            # Keep all samples for lob features
            if self.other_features:
                keep_cols = ['timestamp'] + [f for f in self.other_features if f in df.columns]
                features_dict[fold_id] = df[keep_cols].copy()
                logger(f"Fold {fold_id}: {len(features_dict[fold_id])} full samples for {len(self.other_features)} lob features", "INFO")
        
        logger("="*80, "INFO")
        return returns_dict, features_dict


# =================================================================================================
# Statistical Tests Implementation
# =================================================================================================

class StylizedFactsTests:
    """
    Implements battery of statistical tests for time series features.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Args:
            significance_level: Significance level for tests
        """
        self.alpha = significance_level
        self.test_results = []
    
    # =============================================================================================
    # Stationarity Tests
    # =============================================================================================
    
    def test_adf(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """Augmented Dickey-Fuller test for unit root."""
        try:
            result = adfuller(series.dropna(), autolag='AIC')
            
            self.test_results.append({
                'fold_id': fold_id,
                'fold_type': fold_type,
                'feature_name': feature_name,
                'test_name': 'ADF',
                'test_category': 'stationarity',
                'statistic': result[0],
                'p_value': result[1],
                'critical_value_1%': result[4]['1%'],
                'critical_value_5%': result[4]['5%'],
                'critical_value_10%': result[4]['10%'],
                'n_lags_used': result[2],
                'n_obs': result[3]
            })
        except Exception as e:
            logger(f"  ADF test failed for fold {fold_id}, {feature_name}: {str(e)}", "WARNING")
    
    def test_kpss(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """KPSS test for stationarity."""
        try:
            clean_series = series.dropna()
            
            # Need sufficient samples
            if len(clean_series) < 20:
                return
            
            # Check for constant series (zero variance)
            if clean_series.std() == 0 or np.isnan(clean_series.std()):
                logger(f"  KPSS test skipped for {feature_name}: constant or all-NaN series", "DEBUG")
                return
            
            result = kpss(clean_series, regression='c', nlags='auto')
            
            self.test_results.append({
                'fold_id': fold_id,
                'fold_type': fold_type,
                'feature_name': feature_name,
                'test_name': 'KPSS',
                'test_category': 'stationarity',
                'statistic': result[0],
                'p_value': result[1],
                'critical_value_1%': result[3]['1%'],
                'critical_value_5%': result[3]['5%'],
                'critical_value_10%': result[3]['10%'],
                'n_lags_used': result[2],
                'n_obs': len(clean_series)
            })
        except Exception as e:
            logger(f"  KPSS test failed for fold {fold_id}, {feature_name}: {str(e)}", "WARNING")
    
    def test_phillips_perron(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """Phillips-Perron test for unit root."""
        try:
            clean_series = series.dropna()
            
            # Need sufficient samples
            if len(clean_series) < 20:
                return
            
            # Check for constant series (zero variance)
            if clean_series.std() == 0 or np.isnan(clean_series.std()):
                logger(f"  Phillips-Perron test skipped for {feature_name}: constant or all-NaN series", "DEBUG")
                return
            
            pp = PhillipsPerron(clean_series)
            
            self.test_results.append({
                'fold_id': fold_id,
                'fold_type': fold_type,
                'feature_name': feature_name,
                'test_name': 'Phillips-Perron',
                'test_category': 'stationarity',
                'statistic': pp.stat,
                'p_value': pp.pvalue,
                'critical_value_1%': pp.critical_values['1%'],
                'critical_value_5%': pp.critical_values['5%'],
                'critical_value_10%': pp.critical_values['10%'],
                'n_lags_used': pp.lags,
                'n_obs': pp.nobs
            })
        except Exception as e:
            logger(f"  Phillips-Perron test failed for fold {fold_id}, {feature_name}: {str(e)}", "WARNING")
    
    # =============================================================================================
    # Normality Tests
    # =============================================================================================
    
    def test_jarque_bera(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """Jarque-Bera test for normality."""
        try:
            clean_series = series.dropna()
            statistic, p_value = stats.jarque_bera(clean_series)
            
            self.test_results.append({
                'fold_id': fold_id,
                'fold_type': fold_type,
                'feature_name': feature_name,
                'test_name': 'Jarque-Bera',
                'test_category': 'normality',
                'statistic': statistic,
                'p_value': p_value,
                'n_obs': len(clean_series)
            })
        except Exception as e:
            logger(f"  Jarque-Bera test failed for fold {fold_id}, {feature_name}: {str(e)}", "WARNING")
    
    def test_shapiro_wilk(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """Shapiro-Wilk test for normality."""
        try:
            clean_series = series.dropna()
            # Shapiro-Wilk has sample size limits
            if len(clean_series) > 5000:
                clean_series = clean_series.sample(n=5000, random_state=42)
            
            statistic, p_value = stats.shapiro(clean_series)
            
            self.test_results.append({
                'fold_id': fold_id,
                'fold_type': fold_type,
                'feature_name': feature_name,
                'test_name': 'Shapiro-Wilk',
                'test_category': 'normality',
                'statistic': statistic,
                'p_value': p_value,
                'n_obs': len(clean_series)
            })
        except Exception as e:
            logger(f"  Shapiro-Wilk test failed for fold {fold_id}, {feature_name}: {str(e)}", "WARNING")
    
    def test_anderson_darling(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """Anderson-Darling test for normality."""
        try:
            clean_series = series.dropna()
            result = stats.anderson(clean_series, dist='norm')
            
            self.test_results.append({
                'fold_id': fold_id,
                'fold_type': fold_type,
                'feature_name': feature_name,
                'test_name': 'Anderson-Darling',
                'test_category': 'normality',
                'statistic': result.statistic,
                'critical_value_1%': result.critical_values[4],
                'critical_value_5%': result.critical_values[2],
                'critical_value_10%': result.critical_values[1],
                'n_obs': len(clean_series)
            })
        except Exception as e:
            logger(f"  Anderson-Darling test failed for fold {fold_id}, {feature_name}: {str(e)}", "WARNING")
    
    def test_dagostino_pearson(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """D'Agostino-Pearson omnibus test for normality."""
        try:
            clean_series = series.dropna()
            statistic, p_value = stats.normaltest(clean_series)
            
            self.test_results.append({
                'fold_id': fold_id,
                'fold_type': fold_type,
                'feature_name': feature_name,
                'test_name': "D'Agostino-Pearson",
                'test_category': 'normality',
                'statistic': statistic,
                'p_value': p_value,
                'n_obs': len(clean_series)
            })
        except Exception as e:
            logger(f"  D'Agostino-Pearson test failed for fold {fold_id}, {feature_name}: {str(e)}", "WARNING")
    
    # =============================================================================================
    # Autocorrelation Tests
    # =============================================================================================
    
    def test_ljung_box(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """Ljung-Box test for autocorrelation."""
        try:
            clean_series = series.dropna()
            lags = min(10, len(clean_series) // 5)
            result = acorr_ljungbox(clean_series, lags=lags, return_df=False)
            
            # Use the test at lag 10 (or maximum available lag)
            statistic = result[0][-1] if len(result[0]) > 0 else np.nan
            p_value = result[1][-1] if len(result[1]) > 0 else np.nan
            
            self.test_results.append({
                'fold_id': fold_id,
                'fold_type': fold_type,
                'feature_name': feature_name,
                'test_name': 'Ljung-Box',
                'test_category': 'autocorrelation',
                'statistic': statistic,
                'p_value': p_value,
                'n_lags_used': lags,
                'n_obs': len(clean_series)
            })
        except Exception as e:
            logger(f"  Ljung-Box test failed for fold {fold_id}, {feature_name}: {str(e)}", "WARNING")
    
    def test_durbin_watson(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """Durbin-Watson test for autocorrelation."""
        try:
            from statsmodels.stats.stattools import durbin_watson
            clean_series = series.dropna()
            statistic = durbin_watson(clean_series)
            
            self.test_results.append({
                'fold_id': fold_id,
                'fold_type': fold_type,
                'feature_name': feature_name,
                'test_name': 'Durbin-Watson',
                'test_category': 'autocorrelation',
                'statistic': statistic,
                'n_obs': len(clean_series)
            })
        except Exception as e:
            logger(f"  Durbin-Watson test failed for fold {fold_id}, {feature_name}: {str(e)}", "WARNING")
    
    # =============================================================================================
    # Heteroskedasticity Tests
    # =============================================================================================
    
    def test_arch_lm(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """ARCH-LM test for conditional heteroskedasticity."""
        try:
            clean_series = series.dropna()
            lags = min(5, len(clean_series) // 10)
            result = het_arch(clean_series, nlags=lags)
            
            self.test_results.append({
                'fold_id': fold_id,
                'fold_type': fold_type,
                'feature_name': feature_name,
                'test_name': 'ARCH-LM',
                'test_category': 'heteroskedasticity',
                'statistic': result[0],
                'p_value': result[1],
                'n_lags_used': lags,
                'n_obs': len(clean_series)
            })
        except Exception as e:
            logger(f"  ARCH-LM test failed for fold {fold_id}, {feature_name}: {str(e)}", "WARNING")
    
    def test_breusch_pagan(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """Breusch-Pagan test for heteroskedasticity."""
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            from statsmodels.regression.linear_model import OLS
            import statsmodels.api as sm
            
            clean_series = series.dropna()
            
            # Create time trend as regressor
            X = sm.add_constant(np.arange(len(clean_series)))
            y = clean_series.values
            
            # Fit OLS model
            model = OLS(y, X).fit()
            
            # Run Breusch-Pagan test
            result = het_breuschpagan(model.resid, X)
            
            self.test_results.append({
                'fold_id': fold_id,
                'fold_type': fold_type,
                'feature_name': feature_name,
                'test_name': 'Breusch-Pagan',
                'test_category': 'heteroskedasticity',
                'statistic': result[0],
                'p_value': result[1],
                'n_obs': len(clean_series)
            })
        except Exception as e:
            logger(f"  Breusch-Pagan test failed for fold {fold_id}, {feature_name}: {str(e)}", "WARNING")
    
    def test_white(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """White test for heteroskedasticity."""
        try:
            from statsmodels.stats.diagnostic import het_white
            from statsmodels.regression.linear_model import OLS
            import statsmodels.api as sm
            
            clean_series = series.dropna()
            
            # Create time trend as regressor
            X = sm.add_constant(np.arange(len(clean_series)))
            y = clean_series.values
            
            # Fit OLS model
            model = OLS(y, X).fit()
            
            # Run White test
            result = het_white(model.resid, X)
            
            self.test_results.append({
                'fold_id': fold_id,
                'fold_type': fold_type,
                'feature_name': feature_name,
                'test_name': 'White',
                'test_category': 'heteroskedasticity',
                'statistic': result[0],
                'p_value': result[1],
                'n_obs': len(clean_series)
            })
        except Exception as e:
            logger(f"  White test failed for fold {fold_id}, {feature_name}: {str(e)}", "WARNING")
    
    # =============================================================================================
    # Independence Tests
    # =============================================================================================
    
    def test_bds(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """BDS test for independence."""
        try:
            from statsmodels.tsa.stattools import bds
            clean_series = series.dropna()
            
            # Need at least 100 samples for BDS test
            if len(clean_series) < 100:
                return
            
            # BDS test with embedding dimension 2 and distance epsilon=0.5*std
            result = bds(clean_series, max_dim=2, epsilon=0.5)
            
            # result is a tuple: (statistic_array, pvalue_array)
            # Extract values safely
            if hasattr(result[0], '__len__') and len(result[0]) > 0:
                statistic = float(result[0][0])
                p_value = float(result[1][0])
            else:
                statistic = float(result[0])
                p_value = float(result[1])
            
            self.test_results.append({
                'fold_id': fold_id,
                'fold_type': fold_type,
                'feature_name': feature_name,
                'test_name': 'BDS',
                'test_category': 'independence',
                'statistic': statistic,
                'p_value': p_value,
                'n_obs': len(clean_series)
            })
        except Exception as e:
            logger(f"  BDS test failed for fold {fold_id}, {feature_name}: {str(e)}", "WARNING")
    
    def test_runs(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """Runs test for randomness."""
        try:
            clean_series = series.dropna().reset_index(drop=True)  # Reset index
            
            if len(clean_series) < 20:
                return
            
            median = clean_series.median()
            
            # Create binary sequence (above/below median)
            runs = (clean_series > median).astype(int)
            
            # Count runs
            n_runs = ((runs.iloc[1:].values != runs.iloc[:-1].values).sum()) + 1
            n_pos = (runs == 1).sum()
            n_neg = (runs == 0).sum()
            n = len(runs)
            
            if n == 0 or n_pos == 0 or n_neg == 0:
                return
            
            # Expected runs and variance
            expected_runs = (2 * n_pos * n_neg / n) + 1
            variance_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1))
            
            if variance_runs <= 0:
                return
            
            # Z-statistic
            z_stat = (n_runs - expected_runs) / np.sqrt(variance_runs)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            self.test_results.append({
                'fold_id': fold_id,
                'fold_type': fold_type,
                'feature_name': feature_name,
                'test_name': 'Runs',
                'test_category': 'independence',
                'statistic': z_stat,
                'p_value': p_value,
                'n_obs': n
            })
        except Exception as e:
            logger(f"  Runs test failed for fold {fold_id}, {feature_name}: {str(e)}", "WARNING")
    
    def test_mcleod_li(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """McLeod-Li test for ARCH effects."""
        try:
            clean_series = series.dropna()
            
            # Test on squared series
            squared_series = clean_series ** 2
            lags = min(10, len(squared_series) // 5)
            
            result = acorr_ljungbox(squared_series, lags=lags, return_df=False)
            
            # Use the test at lag 10 (or maximum available lag)
            statistic = result[0][-1] if len(result[0]) > 0 else np.nan
            p_value = result[1][-1] if len(result[1]) > 0 else np.nan
            
            self.test_results.append({
                'fold_id': fold_id,
                'fold_type': fold_type,
                'feature_name': feature_name,
                'test_name': 'McLeod-Li',
                'test_category': 'independence',
                'statistic': statistic,
                'p_value': p_value,
                'n_lags_used': lags,
                'n_obs': len(clean_series)
            })
        except Exception as e:
            logger(f"  McLeod-Li test failed for fold {fold_id}, {feature_name}: {str(e)}", "WARNING")
    
    # =============================================================================================
    # Distributional Shape
    # =============================================================================================
    
    def compute_moments(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """Compute statistical moments."""
        try:
            clean_series = series.dropna()
            
            self.test_results.append({
                'fold_id': fold_id,
                'fold_type': fold_type,
                'feature_name': feature_name,
                'test_name': 'Moments',
                'test_category': 'distribution',
                'mean': clean_series.mean(),
                'std': clean_series.std(),
                'skewness': stats.skew(clean_series),
                'kurtosis': stats.kurtosis(clean_series, fisher=True),  # Excess kurtosis
                'n_obs': len(clean_series)
            })
        except Exception as e:
            logger(f"  Moments computation failed for fold {fold_id}, {feature_name}: {str(e)}", "WARNING")
    
    def compute_tail_index(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """Estimate tail index using Hill estimator."""
        try:
            clean_series = series.dropna()
            
            # Sort absolute values in descending order
            sorted_abs = np.sort(np.abs(clean_series))[::-1]
            
            # Use top 10% for tail estimation
            k = max(10, int(len(sorted_abs) * 0.1))
            tail_values = sorted_abs[:k]
            
            # Hill estimator
            if len(tail_values) > 1 and tail_values[0] > 0:
                log_ratios = np.log(tail_values[:-1] / tail_values[-1])
                tail_index = 1.0 / np.mean(log_ratios) if np.mean(log_ratios) > 0 else np.nan
            else:
                tail_index = np.nan
            
            self.test_results.append({
                'fold_id': fold_id,
                'fold_type': fold_type,
                'feature_name': feature_name,
                'test_name': 'Tail Index',
                'test_category': 'distribution',
                'tail_index': tail_index,
                'n_obs': len(clean_series)
            })
        except Exception as e:
            logger(f"  Tail index computation failed for fold {fold_id}, {feature_name}: {str(e)}", "WARNING")
    
    # =============================================================================================
    # Long Memory
    # =============================================================================================
    
    def compute_hurst_exponent(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """Compute Hurst exponent using R/S analysis."""
        try:
            clean_series = series.dropna().values
            
            if len(clean_series) < 100:
                return
            
            # R/S analysis
            lags = np.logspace(1, np.log10(len(clean_series) // 2), 20).astype(int)
            lags = np.unique(lags)
            
            rs = []
            for lag in lags:
                # Split into chunks
                n_chunks = len(clean_series) // lag
                if n_chunks == 0:
                    continue
                
                chunks = clean_series[:n_chunks * lag].reshape(n_chunks, lag)
                
                # Calculate R/S for each chunk
                rs_chunk = []
                for chunk in chunks:
                    mean_chunk = chunk.mean()
                    deviations = chunk - mean_chunk
                    cumsum = np.cumsum(deviations)
                    R = cumsum.max() - cumsum.min()
                    S = chunk.std()
                    if S > 0:
                        rs_chunk.append(R / S)
                
                if rs_chunk:
                    rs.append(np.mean(rs_chunk))
            
            # Estimate Hurst from log-log regression
            if len(rs) > 2:
                log_lags = np.log(lags[:len(rs)])
                log_rs = np.log(rs)
                
                # Linear regression
                coeffs = np.polyfit(log_lags, log_rs, 1)
                hurst = coeffs[0]
            else:
                hurst = np.nan
            
            self.test_results.append({
                'fold_id': fold_id,
                'fold_type': fold_type,
                'feature_name': feature_name,
                'test_name': 'Hurst Exponent',
                'test_category': 'long_memory',
                'hurst_exponent': hurst,
                'n_obs': len(clean_series)
            })
        except Exception as e:
            logger(f"  Hurst exponent computation failed for fold {fold_id}, {feature_name}: {str(e)}", "WARNING")
    
    # =============================================================================================
    # Test Execution
    # =============================================================================================
    
    def run_all_tests(self, series: pd.Series, fold_id: int, fold_type: str, feature_name: str):
        """
        Runs all statistical tests on a given series.
        
        Skips testing if series has insufficient non-null samples.
        """
        # Check if series has sufficient samples
        clean_series = series.dropna()
        n_samples = len(clean_series)
        
        # Minimum samples needed for reliable tests (most tests need at least 20-30)
        min_required = 30
        
        if n_samples < min_required:
            logger(f"  Skipping {feature_name}: only {n_samples} samples (need {min_required})", "WARNING")
            return
        
        logger(f"  Testing {feature_name} ({n_samples} samples)...", "DEBUG")
        
        # Stationarity tests
        self.test_adf(series, fold_id, fold_type, feature_name)
        self.test_kpss(series, fold_id, fold_type, feature_name)
        self.test_phillips_perron(series, fold_id, fold_type, feature_name)
        
        # Normality tests
        self.test_jarque_bera(series, fold_id, fold_type, feature_name)
        self.test_shapiro_wilk(series, fold_id, fold_type, feature_name)
        self.test_anderson_darling(series, fold_id, fold_type, feature_name)
        self.test_dagostino_pearson(series, fold_id, fold_type, feature_name)
        
        # Autocorrelation tests
        self.test_ljung_box(series, fold_id, fold_type, feature_name)
        self.test_durbin_watson(series, fold_id, fold_type, feature_name)
        
        # Heteroskedasticity tests
        self.test_arch_lm(series, fold_id, fold_type, feature_name)
        self.test_breusch_pagan(series, fold_id, fold_type, feature_name)
        self.test_white(series, fold_id, fold_type, feature_name)
        
        # Independence tests
        self.test_bds(series, fold_id, fold_type, feature_name)
        self.test_runs(series, fold_id, fold_type, feature_name)
        self.test_mcleod_li(series, fold_id, fold_type, feature_name)
        
        # Distributional shape
        self.compute_moments(series, fold_id, fold_type, feature_name)
        self.compute_tail_index(series, fold_id, fold_type, feature_name)
        
        # Long memory
        self.compute_hurst_exponent(series, fold_id, fold_type, feature_name)
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Returns all test results as a DataFrame.
        """
        return pd.DataFrame(self.test_results)