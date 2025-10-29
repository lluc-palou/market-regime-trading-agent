import numpy as np
from typing import Dict
from collections import Counter

from src.utils.logging import logger


def aggregate_across_splits(all_split_results: Dict[int, Dict]) -> Dict:
    """
    Aggregate transformation selection results across all splits.
    
    Computes:
    - Frequency of each transformation winning per feature
    - Average validation scores per transformation
    - Transform stability (consistency across splits)
    - Final recommended transformation
    
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
        # Collect best transforms across splits
        best_transforms = []
        avg_scores = {}
        
        for split_id, split_results in all_split_results.items():
            if feat_name not in split_results:
                continue
            
            feat_res = split_results[feat_name]
            
            # Collect best transform for this split
            if 'best_transform' in feat_res:
                best_transforms.append(feat_res['best_transform'])
            
            # Accumulate scores per transform
            for transform, score in feat_res.get('validation_scores', {}).items():
                if transform not in avg_scores:
                    avg_scores[transform] = []
                avg_scores[transform].append(score)
        
        # Frequency count
        frequency = Counter(best_transforms)
        most_frequent = frequency.most_common(1)[0][0] if frequency else 'identity'
        
        # Average scores per transform
        avg_scores_final = {
            transform: np.mean(scores) 
            for transform, scores in avg_scores.items()
        }
        best_avg = min(avg_scores_final, key=avg_scores_final.get) if avg_scores_final else 'identity'
        
        # Stability (how often most frequent won / total splits)
        stability = frequency[most_frequent] / len(best_transforms) if best_transforms else 0.0
        
        # Compile aggregated results
        aggregated[feat_name] = {
            'most_frequent_transform': most_frequent,
            'frequency_count': dict(frequency),
            'stability': stability,
            'best_avg_transform': best_avg,
            'avg_scores': avg_scores_final,
            'n_splits': len(best_transforms)
        }
        
        logger(f'{feat_name}: most_frequent={most_frequent} '
               f'(stability={stability:.2f}), best_avg={best_avg}', "INFO")
    
    return aggregated


def select_final_transforms(aggregated: Dict, strategy: str = 'most_frequent') -> Dict:
    """
    Select final transformation per feature based on aggregation strategy.
    
    Args:
        aggregated: Aggregated results from aggregate_across_splits
        strategy: Selection strategy ('most_frequent' or 'best_avg')
        
    Returns:
        Dictionary mapping feature_name -> transform_type
    """
    final_transforms = {}
    
    for feat_name, agg_res in aggregated.items():
        if strategy == 'most_frequent':
            final_transforms[feat_name] = agg_res['most_frequent_transform']
        elif strategy == 'best_avg':
            final_transforms[feat_name] = agg_res['best_avg_transform']
        else:
            # Default to most frequent
            final_transforms[feat_name] = agg_res['most_frequent_transform']
    
    return final_transforms