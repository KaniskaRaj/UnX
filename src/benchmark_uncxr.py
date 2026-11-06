import numpy as np
import pandas as pd
from ensemble import compute_ensemble_predictions

def run_real_coverage_sweep():
    # 1. Load your teammates' real logits files (ensure they are placed in a 'preds/' folder)
    try:
        logits_10 = np.load('preds/test_seed10_logits.npy')
        logits_20 = np.load('preds/test_seed20_logits.npy')
        logits_30 = np.load('preds/test_seed30_logits.npy')
    except FileNotFoundError:
        print("⚠️ Prediction files not found in 'preds/' folder. Make sure your teammates' .npy files are added!")
        return

    # 2. Compute Deep Ensemble mean probabilities and uncertainty spread
    mean_probs, uncertainty = compute_ensemble_predictions([logits_10, logits_20, logits_30])
    
    # Load true labels from your split CSV
    df_true = pd.read_csv('data/test_split.csv')
    y_true = df_true.iloc[:, 1:].values # Assuming disease columns start from index 1
    
    N = len(y_true)
    mean_img_uncertainty = np.mean(uncertainty, axis=1)
    sorted_indices = np.argsort(mean_img_uncertainty)
    
    # Coverage sweep from 100% down to 50%
    coverages = np.arange(1.0, 0.45, -0.05)
    results = []
    
    for cov in coverages:
        k = int(cov * N)
        if k == 0:
            continue
        subset_idx = sorted_indices[:k]
        
        results.append({
            'coverage': round(cov, 2),
            'num_samples': k,
            'mean_uncertainty_threshold': mean_img_uncertainty[subset_idx[-1]]
        })
        
    df_results = pd.DataFrame(results)
    print("Real Coverage Sweep Results:")
    print(df_results)
    df_results.to_csv('results/coverage_sweep_results.csv', index=False)

if __name__ == "__main__":
    run_real_coverage_sweep()
