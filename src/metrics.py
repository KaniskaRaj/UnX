import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, ks_2samp

def compute_per_disease_auroc(y_true, y_pred_probs):
    """
    Computes AUROC per disease.
    y_true: numpy array of shape (N, 7) with 0 or 1
    y_pred_probs: numpy array of shape (N, 7) with probabilities
    """
    aurocs = []
    for i in range(y_true.shape[1]):
        # Handle cases where a disease might not have positives in a tiny subset
        try:
            score = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
        except ValueError:
            score = float('nan')
        aurocs.append(score)
    return np.array(aurocs)

def compute_error_detection_auprc(y_true, y_pred_probs, uncertainty):
    """
    Computes AUPRC for error detection.
    Error is defined as 1 if the prediction is wrong (e.g. argmax or threshold mismatch), else 0.
    """
    preds = (y_pred_probs >= 0.5).astype(int)
    errors = (preds != y_true).astype(int)
    
    auprcs = []
    for i in range(y_true.shape[1]):
        try:
            score = average_precision_score(errors[:, i], uncertainty[:, i])
        except ValueError:
            score = float('nan')
        auprcs.append(score)
    return np.array(auprcs)

def compute_spearman_iou(uncertainty_flat, iou_flat):
    """
    Computes Spearman correlation between uncertainty and (1 - IoU).
    """
    # 1 - IoU means higher error in box overlap should correlate with higher uncertainty
    corr, p_value = spearmanr(uncertainty_flat, 1.0 - iou_flat, nan_policy='omit')
    return corr, p_value

def compute_distribution_shift(nih_uncertainty, chexpert_uncertainty):
    """
    Computes Kolmogorov-Smirnov test (ks_2samp) to check OOD shift between NIH and CheXpert.
    """
    p_values = []
    statistic = []
    for i in range(nih_uncertainty.shape[1]):
        res = ks_2samp(nih_uncertainty[:, i], chexpert_uncertainty[:, i])
        statistic.append(res.statistic)
        p_values.append(res.pvalue)
    return np.array(statistic), np.array(p_values)
