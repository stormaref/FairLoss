import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.calibration import calibration_curve

class Metric:
    def __init__(self):
        pass
    
    def ece(self, y_true, y_prob, n_bins=10):
        """
        Calculate Expected Calibration Error (ECE)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Determine if sample is in bin m (between bin lower & upper)
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def nll(self, y_true, y_prob, epsilon=1e-7):
        """
        Calculate Negative Log Likelihood (NLL)
        """
        # Clip probabilities to avoid log(0)
        y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
        
        # Calculate NLL
        nll = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
        return nll
    
    def f1(self, y_true, y_pred, average='binary'):
        """
        Calculate F1 Score
        """
        return f1_score(y_true, y_pred, average=average)
    
    def precision(self, y_true, y_pred, average='binary'):
        """
        Calculate Precision
        """
        return precision_score(y_true, y_pred, average=average)
    
    def recall(self, y_true, y_pred, average='binary'):
        """
        Calculate Recall
        """
        return recall_score(y_true, y_pred, average=average)
    
    def calculate_all_metrics(self, y_true, y_pred, y_prob=None, threshold=0.5):
        """
        Wrapper to calculate all metrics
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels (or probabilities if y_prob is None)
            y_prob: Predicted probabilities (optional)
            threshold: Threshold for converting probabilities to binary predictions
        
        Returns:
            Dictionary containing all metrics
        """
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # If y_prob is not provided, assume y_pred contains probabilities
        if y_prob is None:
            y_prob = y_pred
            y_pred = (y_prob >= threshold).astype(int)
        else:
            y_prob = np.array(y_prob)
        
        metrics = {
            'f1': self.f1(y_true, y_pred),
            'precision': self.precision(y_true, y_pred),
            'recall': self.recall(y_true, y_pred),
            'ece': self.ece(y_true, y_prob),
            'nll': self.nll(y_true, y_prob)
        }
        
        return metrics