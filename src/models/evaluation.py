"""
Evaluation Module for RADAR#
This module implements evaluation metrics and visualization tools for the RADAR# model.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import pandas as pd
import os

class RADAREvaluator:
    """
    Evaluation class for RADAR# model performance assessment
    """
    
    def __init__(self, output_dir='./results/'):
        """
        Initialize the evaluator
        
        Args:
            output_dir: Directory to save evaluation results and visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_model(self, y_true, y_pred, y_pred_proba=None):
        """
        Evaluate model performance with multiple metrics
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
        }
        
        # Add ROC-AUC if probabilities are provided
        if y_pred_proba is not None:
            from sklearn.metrics import roc_auc_score
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def print_classification_report(self, y_true, y_pred, target_names=None):
        """
        Print and save classification report
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            target_names: Names of the classes
            
        Returns:
            Classification report as string
        """
        if target_names is None:
            target_names = ['Non-Extremist', 'Extremist']
        
        report = classification_report(y_true, y_pred, target_names=target_names)
        
        # Save report to file
        with open(os.path.join(self.output_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        return report
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=False, title=None, cmap=plt.cm.Blues):
        """
        Plot and save confusion matrix
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            normalize: Whether to normalize the confusion matrix
            title: Plot title
            cmap: Color map
            
        Returns:
            Confusion matrix as numpy array
        """
        if title is None:
            title = 'Confusion Matrix'
            if normalize:
                title = 'Normalized ' + title
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                    xticklabels=['Non-Extremist', 'Extremist'],
                    yticklabels=['Non-Extremist', 'Extremist'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        """
        Plot and save ROC curve
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            ROC AUC score
        """
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'), dpi=300)
        plt.close()
        
        return roc_auc
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba):
        """
        Plot and save precision-recall curve
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Average precision score
        """
        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        average_precision = average_precision_score(y_true, y_pred_proba)
        
        # Plot precision-recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                 label=f'Precision-Recall curve (AP = {average_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'precision_recall_curve.png'), dpi=300)
        plt.close()
        
        return average_precision
    
    def plot_model_comparison(self, model_metrics, metric_name='f1_score'):
        """
        Plot and save model comparison bar chart
        
        Args:
            model_metrics: Dictionary of model names and their metrics
            metric_name: Name of the metric to compare
            
        Returns:
            None
        """
        models = list(model_metrics.keys())
        values = [metrics[metric_name] for metrics in model_metrics.values()]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, values, color='skyblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.4f}', ha='center', va='bottom')
        
        plt.xlabel('Models')
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.title(f'Model Comparison by {metric_name.replace("_", " ").title()}')
        plt.ylim(0, 1.1)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f'model_comparison_{metric_name}.png'), dpi=300)
        plt.close()
    
    def plot_attention_heatmap(self, text, attention_weights):
        """
        Plot and save attention weights heatmap
        
        Args:
            text: List of tokens
            attention_weights: Attention weights matrix
            
        Returns:
            None
        """
        # Ensure attention_weights is 2D
        if len(attention_weights.shape) > 2:
            attention_weights = attention_weights.squeeze()
        
        # Truncate if necessary
        max_len = min(len(text), attention_weights.shape[0])
        text = text[:max_len]
        attention_weights = attention_weights[:max_len]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention_weights.reshape(1, -1), annot=False, cmap='viridis',
                    xticklabels=text, yticklabels=['Attention'])
        plt.title('Attention Weights Visualization')
        plt.xlabel('Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, 'attention_heatmap.png'), dpi=300)
        plt.close()
    
    def plot_learning_curves(self, history, metrics=None):
        """
        Plot and save learning curves from training history
        
        Args:
            history: Training history object
            metrics: List of metrics to plot (default: accuracy and loss)
            
        Returns:
            None
        """
        if metrics is None:
            metrics = ['accuracy', 'loss']
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            # Plot training metric
            plt.plot(history.history[metric], label=f'Training {metric}')
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in history.history:
                plt.plot(history.history[val_metric], label=f'Validation {metric}')
            
            plt.title(f'Model {metric.capitalize()} over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, f'learning_curve_{metric}.png'), dpi=300)
            plt.close()
    
    def generate_evaluation_report(self, model_name, metrics, dataset_info=None):
        """
        Generate and save comprehensive evaluation report
        
        Args:
            model_name: Name of the evaluated model
            metrics: Dictionary of evaluation metrics
            dataset_info: Information about the dataset used
            
        Returns:
            Report as string
        """
        report = f"# Evaluation Report for {model_name}\n\n"
        
        # Add date and time
        from datetime import datetime
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add dataset information if provided
        if dataset_info:
            report += "## Dataset Information\n\n"
            for key, value in dataset_info.items():
                report += f"- **{key}**: {value}\n"
            report += "\n"
        
        # Add performance metrics
        report += "## Performance Metrics\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        for metric, value in metrics.items():
            report += f"| {metric.replace('_', ' ').title()} | {value:.4f} |\n"
        report += "\n"
        
        # Add references to generated visualizations
        report += "## Visualizations\n\n"
        report += "The following visualizations have been generated:\n\n"
        report += "1. Confusion Matrix (`confusion_matrix.png`)\n"
        report += "2. ROC Curve (`roc_curve.png`)\n"
        report += "3. Precision-Recall Curve (`precision_recall_curve.png`)\n"
        report += "4. Learning Curves (`learning_curve_accuracy.png`, `learning_curve_loss.png`)\n"
        
        # Save report to file
        with open(os.path.join(self.output_dir, 'evaluation_report.md'), 'w') as f:
            f.write(report)
        
        return report


# Example usage
if __name__ == "__main__":
    # Sample data (would be actual model predictions in real implementation)
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1])
    y_pred_proba = np.array([0.2, 0.8, 0.3, 0.4, 0.7, 0.2, 0.9, 0.6, 0.3, 0.8])
    
    # Initialize evaluator
    evaluator = RADAREvaluator(output_dir='./sample_results/')
    
    # Evaluate model
    metrics = evaluator.evaluate_model(y_true, y_pred, y_pred_proba)
    print("Evaluation metrics:", metrics)
    
    # Generate classification report
    report = evaluator.print_classification_report(y_true, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(y_true, y_pred, normalize=True)
    
    # Plot ROC curve
    evaluator.plot_roc_curve(y_true, y_pred_proba)
    
    # Plot precision-recall curve
    evaluator.plot_precision_recall_curve(y_true, y_pred_proba)
    
    # Plot model comparison
    model_metrics = {
        'CNN-Bi-LSTM': {'f1_score': 0.85, 'accuracy': 0.87},
        'AraBERT': {'f1_score': 0.88, 'accuracy': 0.89},
        'MARBERT': {'f1_score': 0.90, 'accuracy': 0.91},
        'Ensemble': {'f1_score': 0.93, 'accuracy': 0.94}
    }
    evaluator.plot_model_comparison(model_metrics, 'f1_score')
    
    # Generate evaluation report
    dataset_info = {
        'Name': 'Arabic Tweets Dataset',
        'Size': '89,816 tweets',
        'Classes': 'Extremist (56%), Non-extremist (44%)',
        'Period': '2011-2021'
    }
    evaluator.generate_evaluation_report('RADAR# Ensemble', metrics, dataset_info)
