"""
Evaluation metrics for RADAR# project.

This module implements various evaluation metrics for assessing
the performance of RADAR# models on Arabic radicalization detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report as sk_classification_report,
    precision_recall_curve, roc_curve, auc
)
from typing import Dict, List, Tuple, Union, Optional, Any

def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray, 
                     y_prob: Optional[np.ndarray] = None,
                     average: str = 'weighted') -> Dict[str, float]:
    """
    Calculate various classification metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (for ROC AUC).
        average: Averaging method for multi-class metrics.
        
    Returns:
        Dictionary of metrics.
    """
    # Convert one-hot encoded labels to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_indices = np.argmax(y_pred, axis=1)
        y_prob = y_pred  # Store probabilities before converting to indices
        y_pred = y_pred_indices
    
    # Calculate basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # Calculate ROC AUC if probabilities are provided
    if y_prob is not None:
        # Binary classification
        if len(np.unique(y_true)) == 2:
            if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                # Use the probability of the positive class
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        # Multi-class classification
        else:
            try:
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_prob, average=average, multi_class='ovr'
                )
            except ValueError:
                # If ROC AUC calculation fails, skip it
                pass
    
    return metrics

def classification_report(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         target_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Generate a classification report with per-class metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        target_names: List of class names.
        
    Returns:
        Dictionary containing the classification report.
    """
    # Convert one-hot encoded labels to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Generate classification report
    report = sk_classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )
    
    return report

def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         normalize: bool = False,
                         figsize: Tuple[int, int] = (10, 8),
                         cmap: str = 'Blues',
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot a confusion matrix.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        normalize: Whether to normalize the confusion matrix.
        figsize: Figure size.
        cmap: Colormap for the plot.
        save_path: Path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    # Convert one-hot encoded labels to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot confusion matrix
    sns.heatmap(
        cm, annot=True, fmt='.2f' if normalize else 'd',
        cmap=cmap, xticklabels=class_names, yticklabels=class_names
    )
    
    plt.title('Confusion Matrix for RADAR# Classification', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_roc_curve(y_true: np.ndarray, 
                  y_prob: np.ndarray,
                  class_names: Optional[List[str]] = None,
                  figsize: Tuple[int, int] = (10, 8),
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curves.
    
    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        class_names: List of class names.
        figsize: Figure size.
        save_path: Path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    # Convert one-hot encoded labels to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Binary classification
    if len(np.unique(y_true_indices)) == 2:
        # Get probabilities for the positive class
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = y_prob
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true_indices, y_prob_pos)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(
            fpr, tpr, lw=2,
            label=f'ROC curve (area = {roc_auc:.2f})'
        )
    
    # Multi-class classification
    else:
        # One-vs-Rest ROC curves for each class
        n_classes = len(np.unique(y_true_indices))
        
        # Ensure class_names has the right length
        if class_names is None or len(class_names) != n_classes:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Binarize the labels for one-vs-rest ROC
        y_true_bin = np.zeros((len(y_true_indices), n_classes))
        for i in range(n_classes):
            y_true_bin[:, i] = (y_true_indices == i).astype(int)
        
        # Plot ROC curve for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr, lw=2,
                label=f'{class_names[i]} (area = {roc_auc:.2f})'
            )
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_precision_recall_curve(y_true: np.ndarray, 
                               y_prob: np.ndarray,
                               class_names: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot precision-recall curves.
    
    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        class_names: List of class names.
        figsize: Figure size.
        save_path: Path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    # Convert one-hot encoded labels to class indices if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Binary classification
    if len(np.unique(y_true_indices)) == 2:
        # Get probabilities for the positive class
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = y_prob
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true_indices, y_prob_pos)
        pr_auc = auc(recall, precision)
        
        # Plot precision-recall curve
        plt.plot(
            recall, precision, lw=2,
            label=f'PR curve (area = {pr_auc:.2f})'
        )
    
    # Multi-class classification
    else:
        # One-vs-Rest precision-recall curves for each class
        n_classes = len(np.unique(y_true_indices))
        
        # Ensure class_names has the right length
        if class_names is None or len(class_names) != n_classes:
            class_names = [f'Class {i}' for i in range(n_classes)]
        
        # Binarize the labels for one-vs-rest precision-recall
        y_true_bin = np.zeros((len(y_true_indices), n_classes))
        for i in range(n_classes):
            y_true_bin[:, i] = (y_true_indices == i).astype(int)
        
        # Plot precision-recall curve for each class
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            pr_auc = auc(recall, precision)
            
            plt.plot(
                recall, precision, lw=2,
                label=f'{class_names[i]} (area = {pr_auc:.2f})'
            )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_error_distribution(error_sources: Dict[str, float],
                           figsize: Tuple[int, int] = (10, 6),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of classification errors by source.
    
    Args:
        error_sources: Dictionary mapping error sources to percentages.
        figsize: Figure size.
        save_path: Path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Extract sources and percentages
    sources = list(error_sources.keys())
    percentages = list(error_sources.values())
    
    # Create pie chart
    plt.pie(
        percentages, labels=sources, autopct='%1.1f%%',
        startangle=90, shadow=False,
        colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    )
    
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Distribution of Classification Errors by Source', fontsize=16)
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_model_comparison(models: List[str],
                         metrics: Dict[str, List[float]],
                         figsize: Tuple[int, int] = (12, 6),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparative performance of multiple models.
    
    Args:
        models: List of model names.
        metrics: Dictionary mapping metric names to lists of values for each model.
        figsize: Figure size.
        save_path: Path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Number of models and metrics
    n_models = len(models)
    n_metrics = len(metrics)
    
    # Set up bar positions
    x = np.arange(n_models)
    width = 0.8 / n_metrics
    
    # Plot bars for each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        plt.bar(
            x + (i - n_metrics/2 + 0.5) * width,
            values,
            width,
            label=metric_name
        )
    
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Performance', fontsize=14)
    plt.title('Comparative Performance of Models', fontsize=16)
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_learning_curves(history: Dict[str, List[float]],
                        figsize: Tuple[int, int] = (12, 5),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot learning curves from training history.
    
    Args:
        history: Dictionary of training history.
        figsize: Figure size.
        save_path: Path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot training and validation loss
    ax1.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Validation Loss')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot training and validation accuracy
    ax2.plot(history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history:
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
