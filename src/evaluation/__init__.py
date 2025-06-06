"""
Evaluation module initialization for RADAR# project.

This package contains modules for model evaluation, error analysis,
and interpretability visualization.
"""

from .metrics import calculate_metrics, classification_report
from .error_analysis import ErrorAnalyzer
from .interpretability import AttentionVisualizer

__all__ = ['calculate_metrics', 'classification_report', 'ErrorAnalyzer', 'AttentionVisualizer']
