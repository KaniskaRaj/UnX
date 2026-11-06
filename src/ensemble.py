import numpy as np

def compute_ensemble_predictions(logits_list):
    """
    logits_list: list of numpy arrays, each of shape (N, 7), from different seeds.
    Returns:
        mean_probs: average probability across models (shape: N, 7)
        ensemble_uncertainty: standard deviation across model probabilities (shape: N, 7)
    """
    probs_list = []
    for logits in logits_list:
        # Convert logits to probabilities using sigmoid
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs_list.append(probs)
    
    # Stack along a new axis: shape becomes (num_models, N, 7)
    stacked_probs = np.stack(probs_list, axis=0)
    
    # Mean probability across models
    mean_probs = np.mean(stacked_probs, axis=0)
    
    # Standard deviation across models as the uncertainty score
    ensemble_uncertainty = np.std(stacked_probs, axis=0)
    
    return mean_probs, ensemble_uncertainty
