"""
EWMA Half-Life Aggregator

Aggregates half-life selection results across all splits.
"""

import numpy as np
from typing import Dict
from collections import Counter

from src.utils.logging import logger


def aggregate_across_splits(all_split_results: Dict[int, Dict]) -> Dict:
    """
    Aggregate half-life selection results across all splits.
    
    Computes:
    - Frequency of each half-life winning per feature
    - Average validation scores per half-life
    - Half-life stability (consistency across splits)
    - Final recommended half-life
    
    Args:
        all_split_results: Dictionary mapping split_id -> split results
        
    Returns:
        Dictionary with aggregated results per feature
    """
    logger('=' * 80, "INFO")
    logger('AGGREGATING ACROSS SPLITS', "INFO")
    logger('=' * 80, "INFO")
    
    # Get feature names from first split
    first_split = next(iter(all_split_results.values()))
    feature_names = list(first_split.keys())
    
    aggregated = {}
    
    for feat_name in feature_names:
        # Collect best half-lives across splits
        best_half_lives = []
        avg_scores = {}
        
        for split_id, split_results in all_split_results.items():
            if feat_name not in split_results:
                continue
            
            feat_res = split_results[feat_name]
            
            # Collect best half-life for this split
            if 'best_half_life' in feat_res:
                best_half_lives.append(feat_res['best_half_life'])
            
            # Accumulate scores per half-life
            for half_life, score in feat_res.get('validation_scores', {}).items():
                if half_life not in avg_scores:
                    avg_scores[half_life] = []
                avg_scores[half_life].append(score)
        
        # Frequency count
        frequency = Counter(best_half_lives)
        most_frequent = frequency.most_common(1)[0][0] if frequency else 20  # Default to 20
        
        # Average scores per half-life
        avg_scores_final = {
            half_life: np.mean(scores)
            for half_life, scores in avg_scores.items()
        }
        best_avg = min(avg_scores_final, key=avg_scores_final.get) if avg_scores_final else 20
        
        # Stability (how often most frequent won / total splits)
        stability = frequency[most_frequent] / len(best_half_lives) if best_half_lives else 0.0
        
        # Compile aggregated results
        aggregated[feat_name] = {
            'most_frequent_half_life': int(most_frequent),
            'frequency_count': {int(k): v for k, v in frequency.items()},
            'stability': float(stability),
            'best_avg_half_life': int(best_avg),
            'avg_scores': {int(k): float(v) for k, v in avg_scores_final.items()},
            'n_splits': len(best_half_lives)
        }
        
        logger(f'{feat_name}: most_frequent={most_frequent} '
               f'(stability={stability:.2f}), best_avg={best_avg}', "INFO")
    
    return aggregated


def select_final_half_lives(aggregated: Dict, strategy: str = 'most_frequent') -> Dict:
    """
    Select final half-life per feature based on aggregation strategy.
    
    Args:
        aggregated: Aggregated results from aggregate_across_splits
        strategy: Selection strategy ('most_frequent' or 'best_avg')
        
    Returns:
        Dictionary mapping feature_name -> half_life
    """
    final_half_lives = {}
    
    for feat_name, agg_res in aggregated.items():
        if strategy == 'most_frequent':
            final_half_lives[feat_name] = agg_res['most_frequent_half_life']
        elif strategy == 'best_avg':
            final_half_lives[feat_name] = agg_res['best_avg_half_life']
        else:
            # Default to most frequent
            final_half_lives[feat_name] = agg_res['most_frequent_half_life']
    
    return final_half_lives