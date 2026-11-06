import numpy as np
import pandas as pd
from metrics import compute_error_detection_auprc, compute_spearman_iou

def run_coverage_sweep(y_true, y_pred_probs, uncertainty, iou_df=None):
    """
    Performs coverage sweep from 100% down to 50% based on uncertainty.
    Sorts by uncertainty ascending (lowest uncertainty first).
    """
    N = len(y_true)
    # Mean uncertainty across diseases per image to rank images
    mean_img_uncertainty = np.mean(uncertainty, axis=1)
    sorted_indices = np.argsort(mean_img_uncertainty)
    
    coverages = np.arange(1.0, 0.45, -0.05)
    results = []
    
    for cov in coverages:
        k = int(cov * N)
        if k == 0:
            continue
        subset_idx = sorted_indices[:k]
        
        subset_true = y_true[subset_idx]
        subset_preds = y_pred_probs[subset_idx]
        
        # Calculate mean metrics for this coverage level
        results.append({
            'coverage': cov,
            'num_samples': k,
        })
        
    return pd.DataFrame(results)
