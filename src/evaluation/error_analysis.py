"""
Error analysis module for RADAR# project.

This module implements tools for analyzing classification errors
and understanding model performance across different categories.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import Counter, defaultdict

class ErrorAnalyzer:
    """
    Error analysis for RADAR# models.
    
    This class implements tools for analyzing classification errors
    and understanding model performance across different categories.
    
    Attributes:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        y_prob (np.ndarray): Predicted probabilities.
        texts (List[str]): Original text samples.
        class_names (List[str]): List of class names.
        error_indices (np.ndarray): Indices of misclassified samples.
    """
    
    def __init__(self, 
                y_true: np.ndarray,
                y_pred: np.ndarray,
                y_prob: Optional[np.ndarray] = None,
                texts: Optional[List[str]] = None,
                class_names: Optional[List[str]] = None):
        """
        Initialize the error analyzer.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            y_prob: Predicted probabilities.
            texts: Original text samples.
            class_names: List of class names.
        """
        # Convert one-hot encoded labels to class indices if needed
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            self.y_true_indices = np.argmax(y_true, axis=1)
            self.y_true = y_true
        else:
            self.y_true_indices = y_true
            self.y_true = y_true
        
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            self.y_pred_indices = np.argmax(y_pred, axis=1)
            self.y_pred = y_pred
        else:
            self.y_pred_indices = y_pred
            self.y_pred = y_pred
        
        self.y_prob = y_prob
        self.texts = texts
        
        # Set class names
        if class_names is None:
            n_classes = len(np.unique(self.y_true_indices))
            self.class_names = [f'Class {i}' for i in range(n_classes)]
        else:
            self.class_names = class_names
        
        # Find indices of misclassified samples
        self.error_indices = np.where(self.y_true_indices != self.y_pred_indices)[0]
    
    def get_error_samples(self) -> Dict[str, List]:
        """
        Get misclassified samples.
        
        Returns:
            Dictionary containing misclassified samples.
        """
        error_samples = {
            'indices': self.error_indices,
            'true_labels': self.y_true_indices[self.error_indices],
            'pred_labels': self.y_pred_indices[self.error_indices]
        }
        
        if self.y_prob is not None:
            error_samples['probabilities'] = self.y_prob[self.error_indices]
        
        if self.texts is not None:
            error_samples['texts'] = [self.texts[i] for i in self.error_indices]
        
        return error_samples
    
    def get_error_distribution(self) -> Dict[str, float]:
        """
        Get distribution of errors by category.
        
        Returns:
            Dictionary mapping error categories to percentages.
        """
        # Count errors by true class
        error_counts = Counter()
        for i in self.error_indices:
            true_label = self.y_true_indices[i]
            pred_label = self.y_pred_indices[i]
            error_counts[f"{self.class_names[true_label]} → {self.class_names[pred_label]}"] += 1
        
        # Convert counts to percentages
        total_errors = len(self.error_indices)
        error_distribution = {k: (v / total_errors) * 100 for k, v in error_counts.items()}
        
        return error_distribution
    
    def get_error_sources(self) -> Dict[str, float]:
        """
        Analyze sources of classification errors.
        
        Returns:
            Dictionary mapping error sources to percentages.
        """
        # This is a simplified implementation that categorizes errors
        # In a real implementation, this would use more sophisticated analysis
        error_sources = defaultdict(int)
        
        for i in self.error_indices:
            true_label = self.y_true_indices[i]
            pred_label = self.y_pred_indices[i]
            
            # Check confidence of prediction
            if self.y_prob is not None:
                confidence = self.y_prob[i][pred_label]
                
                if confidence > 0.9:
                    # High confidence errors often indicate ambiguous content
                    error_sources["Ambiguous Content"] += 1
                elif confidence < 0.6:
                    # Low confidence errors often indicate insufficient features
                    error_sources["Insufficient Features"] += 1
                else:
                    # Medium confidence errors often indicate feature confusion
                    error_sources["Feature Confusion"] += 1
            else:
                # If probabilities are not available, categorize by class
                error_sources[f"{self.class_names[true_label]}-{self.class_names[pred_label]} Confusion"] += 1
        
        # Convert counts to percentages
        total_errors = len(self.error_indices)
        error_sources_pct = {k: (v / total_errors) * 100 for k, v in error_sources.items()}
        
        return error_sources_pct
    
    def get_per_class_metrics(self) -> pd.DataFrame:
        """
        Calculate performance metrics for each class.
        
        Returns:
            DataFrame with per-class metrics.
        """
        from sklearn.metrics import precision_recall_fscore_support
        
        # Calculate precision, recall, and F1 score for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true_indices, self.y_pred_indices, average=None
        )
        
        # Create DataFrame
        metrics_df = pd.DataFrame({
            'Class': self.class_names,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Support': support
        })
        
        return metrics_df
    
    def plot_error_distribution(self, 
                               figsize: Tuple[int, int] = (12, 8),
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of classification errors.
        
        Args:
            figsize: Figure size.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        # Get error distribution
        error_dist = self.get_error_distribution()
        
        # Sort by percentage
        error_dist = {k: v for k, v in sorted(error_dist.items(), key=lambda item: item[1], reverse=True)}
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot bar chart
        plt.bar(error_dist.keys(), error_dist.values(), color='#ff6b6b')
        
        plt.xlabel('Error Category', fontsize=14)
        plt.ylabel('Percentage of Errors (%)', fontsize=14)
        plt.title('Distribution of Classification Errors', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_confidence_distribution(self,
                                    figsize: Tuple[int, int] = (10, 6),
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of prediction confidence for correct and incorrect predictions.
        
        Args:
            figsize: Figure size.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        if self.y_prob is None:
            raise ValueError("Prediction probabilities are required for confidence distribution")
        
        # Get confidence values
        confidences = np.max(self.y_prob, axis=1)
        
        # Separate confidences for correct and incorrect predictions
        correct_mask = self.y_true_indices == self.y_pred_indices
        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[~correct_mask]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot histograms
        plt.hist(
            correct_confidences, bins=20, alpha=0.7, label='Correct Predictions',
            color='#4CAF50'
        )
        plt.hist(
            incorrect_confidences, bins=20, alpha=0.7, label='Incorrect Predictions',
            color='#F44336'
        )
        
        plt.xlabel('Prediction Confidence', fontsize=14)
        plt.ylabel('Number of Samples', fontsize=14)
        plt.title('Distribution of Prediction Confidence', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_error_heatmap(self,
                          figsize: Tuple[int, int] = (10, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot heatmap of classification errors.
        
        Args:
            figsize: Figure size.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        from sklearn.metrics import confusion_matrix
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_true_indices, self.y_pred_indices)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=self.class_names, yticklabels=self.class_names
        )
        
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.title('Classification Error Heatmap', fontsize=16)
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def get_most_confused_samples(self, n: int = 10) -> pd.DataFrame:
        """
        Get the most confused samples (highest confidence in wrong prediction).
        
        Args:
            n: Number of samples to return.
            
        Returns:
            DataFrame with confused samples.
        """
        if self.y_prob is None:
            raise ValueError("Prediction probabilities are required for confused samples")
        
        if self.texts is None:
            raise ValueError("Text samples are required for confused samples")
        
        # Get confidence values for incorrect predictions
        error_confidences = []
        for i in self.error_indices:
            pred_label = self.y_pred_indices[i]
            confidence = self.y_prob[i][pred_label]
            error_confidences.append((i, confidence))
        
        # Sort by confidence (descending)
        error_confidences.sort(key=lambda x: x[1], reverse=True)
        
        # Get top n confused samples
        confused_samples = []
        for i, confidence in error_confidences[:n]:
            true_label = self.y_true_indices[i]
            pred_label = self.y_pred_indices[i]
            
            confused_samples.append({
                'Index': i,
                'Text': self.texts[i],
                'True Label': self.class_names[true_label],
                'Predicted Label': self.class_names[pred_label],
                'Confidence': confidence
            })
        
        return pd.DataFrame(confused_samples)
    
    def analyze_error_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in classification errors.
        
        Returns:
            Dictionary with error pattern analysis.
        """
        # This is a simplified implementation
        # In a real implementation, this would use more sophisticated analysis
        
        # Calculate per-class error rates
        class_counts = Counter(self.y_true_indices)
        class_error_counts = Counter()
        
        for i in self.error_indices:
            true_label = self.y_true_indices[i]
            class_error_counts[true_label] += 1
        
        class_error_rates = {
            self.class_names[label]: (count / class_counts[label]) * 100
            for label, count in class_error_counts.items()
        }
        
        # Identify most common misclassifications
        misclassifications = Counter()
        for i in self.error_indices:
            true_label = self.y_true_indices[i]
            pred_label = self.y_pred_indices[i]
            misclassifications[(true_label, pred_label)] += 1
        
        top_misclassifications = [
            {
                'True': self.class_names[true],
                'Predicted': self.class_names[pred],
                'Count': count,
                'Percentage': (count / len(self.error_indices)) * 100
            }
            for (true, pred), count in misclassifications.most_common(5)
        ]
        
        # Return analysis results
        return {
            'class_error_rates': class_error_rates,
            'top_misclassifications': top_misclassifications,
            'total_errors': len(self.error_indices),
            'error_rate': (len(self.error_indices) / len(self.y_true_indices)) * 100
        }
