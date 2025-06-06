"""
Interpretability visualization module for RADAR# project.

This module implements tools for visualizing model interpretability,
particularly attention weights and feature importance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

class AttentionVisualizer:
    """
    Visualizer for attention weights in RADAR# models.
    
    This class implements tools for visualizing attention weights
    and interpreting model decisions.
    
    Attributes:
        texts (List[str]): Original text samples.
        attention_weights (np.ndarray): Attention weights for each sample.
        predictions (np.ndarray): Model predictions.
        true_labels (np.ndarray): Ground truth labels.
        class_names (List[str]): List of class names.
    """
    
    def __init__(self, 
                texts: List[str],
                attention_weights: np.ndarray,
                predictions: Optional[np.ndarray] = None,
                true_labels: Optional[np.ndarray] = None,
                class_names: Optional[List[str]] = None):
        """
        Initialize the attention visualizer.
        
        Args:
            texts: Original text samples.
            attention_weights: Attention weights for each sample.
            predictions: Model predictions.
            true_labels: Ground truth labels.
            class_names: List of class names.
        """
        self.texts = texts
        self.attention_weights = attention_weights
        self.predictions = predictions
        self.true_labels = true_labels
        
        # Set class names
        if class_names is None and predictions is not None:
            if len(predictions.shape) > 1:
                n_classes = predictions.shape[1]
            else:
                n_classes = len(np.unique(predictions))
            self.class_names = [f'Class {i}' for i in range(n_classes)]
        else:
            self.class_names = class_names
    
    def visualize_attention(self, 
                           sample_idx: int,
                           figsize: Tuple[int, int] = (12, 4),
                           cmap: str = 'YlOrRd',
                           save_path: Optional[str] = None) -> Figure:
        """
        Visualize attention weights for a single sample.
        
        Args:
            sample_idx: Index of the sample to visualize.
            figsize: Figure size.
            cmap: Colormap for attention weights.
            save_path: Path to save the visualization.
            
        Returns:
            Matplotlib figure.
        """
        # Get text and attention weights for the sample
        text = self.texts[sample_idx]
        attention = self.attention_weights[sample_idx]
        
        # Split text into tokens (simplified)
        tokens = text.split()
        
        # Ensure attention weights match the number of tokens
        if len(tokens) != len(attention):
            # Truncate or pad attention weights to match tokens
            if len(tokens) < len(attention):
                attention = attention[:len(tokens)]
            else:
                attention = np.pad(attention, (0, len(tokens) - len(attention)), 'constant')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a color map
        norm = mcolors.Normalize(vmin=0, vmax=np.max(attention))
        cmap_obj = plt.cm.get_cmap(cmap)
        
        # Plot tokens with attention weights as background color
        for i, (token, weight) in enumerate(zip(tokens, attention)):
            # Calculate position
            x_pos = i
            y_pos = 0
            
            # Get color based on attention weight
            color = cmap_obj(norm(weight))
            
            # Add colored rectangle
            rect = mpatches.Rectangle((x_pos, y_pos), 1, 1, color=color, alpha=0.7)
            ax.add_patch(rect)
            
            # Add token text
            ax.text(x_pos + 0.5, y_pos + 0.5, token, ha='center', va='center', fontsize=12)
        
        # Set axis limits
        ax.set_xlim(0, len(tokens))
        ax.set_ylim(0, 1)
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label('Attention Weight')
        
        # Add title with prediction and true label if available
        title = f"Attention Visualization for Sample {sample_idx}"
        if self.predictions is not None and self.true_labels is not None:
            pred_idx = self.predictions[sample_idx] if len(self.predictions.shape) == 1 else np.argmax(self.predictions[sample_idx])
            true_idx = self.true_labels[sample_idx] if len(self.true_labels.shape) == 1 else np.argmax(self.true_labels[sample_idx])
            
            pred_label = self.class_names[pred_idx] if self.class_names else f"Class {pred_idx}"
            true_label = self.class_names[true_idx] if self.class_names else f"Class {true_idx}"
            
            title += f"\nPrediction: {pred_label}, True: {true_label}"
        
        plt.title(title, fontsize=14)
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_attention_heatmap(self, 
                                  sample_idx: int,
                                  figsize: Tuple[int, int] = (12, 6),
                                  cmap: str = 'YlOrRd',
                                  save_path: Optional[str] = None) -> Figure:
        """
        Visualize attention weights as a heatmap for a single sample.
        
        Args:
            sample_idx: Index of the sample to visualize.
            figsize: Figure size.
            cmap: Colormap for attention weights.
            save_path: Path to save the visualization.
            
        Returns:
            Matplotlib figure.
        """
        # Get text and attention weights for the sample
        text = self.texts[sample_idx]
        attention = self.attention_weights[sample_idx]
        
        # Split text into tokens (simplified)
        tokens = text.split()
        
        # Ensure attention weights match the number of tokens
        if len(tokens) != len(attention):
            # Truncate or pad attention weights to match tokens
            if len(tokens) < len(attention):
                attention = attention[:len(tokens)]
            else:
                attention = np.pad(attention, (0, len(tokens) - len(attention)), 'constant')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        attention_matrix = attention.reshape(1, -1)
        sns.heatmap(
            attention_matrix, cmap=cmap, annot=False,
            xticklabels=tokens, yticklabels=False,
            ax=ax, cbar_kws={'label': 'Attention Weight'}
        )
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Add title with prediction and true label if available
        title = f"Attention Heatmap for Sample {sample_idx}"
        if self.predictions is not None and self.true_labels is not None:
            pred_idx = self.predictions[sample_idx] if len(self.predictions.shape) == 1 else np.argmax(self.predictions[sample_idx])
            true_idx = self.true_labels[sample_idx] if len(self.true_labels.shape) == 1 else np.argmax(self.true_labels[sample_idx])
            
            pred_label = self.class_names[pred_idx] if self.class_names else f"Class {pred_idx}"
            true_label = self.class_names[true_idx] if self.class_names else f"Class {true_idx}"
            
            title += f"\nPrediction: {pred_label}, True: {true_label}"
        
        plt.title(title, fontsize=14)
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_semantic_attention_distribution(self,
                                               semantic_categories: Dict[str, List[str]],
                                               figsize: Tuple[int, int] = (10, 6),
                                               save_path: Optional[str] = None) -> Figure:
        """
        Visualize distribution of attention weights across semantic categories.
        
        Args:
            semantic_categories: Dictionary mapping category names to lists of keywords.
            figsize: Figure size.
            save_path: Path to save the visualization.
            
        Returns:
            Matplotlib figure.
        """
        # Calculate attention weight per semantic category
        category_weights = {}
        
        for category, keywords in semantic_categories.items():
            category_weights[category] = 0
            count = 0
            
            for i, text in enumerate(self.texts):
                tokens = text.split()
                for j, token in enumerate(tokens):
                    if j < len(self.attention_weights[i]) and token.lower() in keywords:
                        category_weights[category] += self.attention_weights[i][j]
                        count += 1
            
            # Calculate average attention weight for the category
            if count > 0:
                category_weights[category] /= count
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot bar chart
        categories = list(category_weights.keys())
        weights = list(category_weights.values())
        
        bars = ax.bar(categories, weights, color='#5DA5DA')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10
            )
        
        plt.xlabel('Semantic Category', fontsize=14)
        plt.ylabel('Average Attention Weight', fontsize=14)
        plt.title('Distribution of Attention Weights Across Semantic Categories', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_attention_flow(self,
                               sample_idx: int,
                               figsize: Tuple[int, int] = (12, 8),
                               save_path: Optional[str] = None) -> Figure:
        """
        Visualize attention flow from input to prediction for a single sample.
        
        Args:
            sample_idx: Index of the sample to visualize.
            figsize: Figure size.
            save_path: Path to save the visualization.
            
        Returns:
            Matplotlib figure.
        """
        # This is a simplified implementation
        # In a real implementation, this would visualize the flow of attention
        # through the model layers
        
        # Get text and attention weights for the sample
        text = self.texts[sample_idx]
        attention = self.attention_weights[sample_idx]
        
        # Split text into tokens (simplified)
        tokens = text.split()
        
        # Ensure attention weights match the number of tokens
        if len(tokens) != len(attention):
            # Truncate or pad attention weights to match tokens
            if len(tokens) < len(attention):
                attention = attention[:len(tokens)]
            else:
                attention = np.pad(attention, (0, len(tokens) - len(attention)), 'constant')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Number of tokens to display
        n_tokens = min(len(tokens), 20)  # Limit to 20 tokens for readability
        
        # Get top tokens by attention weight
        top_indices = np.argsort(attention)[-n_tokens:]
        top_tokens = [tokens[i] for i in top_indices]
        top_weights = [attention[i] for i in top_indices]
        
        # Normalize weights for visualization
        norm_weights = np.array(top_weights) / np.max(top_weights)
        
        # Plot tokens as nodes
        for i, (token, weight) in enumerate(zip(top_tokens, norm_weights)):
            # Calculate position
            x_pos = 0.2
            y_pos = i / n_tokens
            
            # Add token node
            circle = plt.Circle((x_pos, y_pos), 0.03 + weight * 0.05, color='#5DA5DA', alpha=0.7)
            ax.add_patch(circle)
            
            # Add token text
            ax.text(x_pos + 0.1, y_pos, token, ha='left', va='center', fontsize=12)
            
            # Add weight text
            ax.text(x_pos - 0.1, y_pos, f'{top_weights[i]:.3f}', ha='right', va='center', fontsize=10)
        
        # Add prediction node
        if self.predictions is not None:
            pred_idx = self.predictions[sample_idx] if len(self.predictions.shape) == 1 else np.argmax(self.predictions[sample_idx])
            pred_label = self.class_names[pred_idx] if self.class_names else f"Class {pred_idx}"
            
            # Draw prediction node
            pred_circle = plt.Circle((0.8, 0.5), 0.1, color='#F15854', alpha=0.7)
            ax.add_patch(pred_circle)
            
            # Add prediction text
            ax.text(0.8, 0.5, pred_label, ha='center', va='center', fontsize=14, color='white')
            
            # Draw arrows from tokens to prediction
            for i in range(n_tokens):
                y_pos = i / n_tokens
                ax.arrow(
                    0.25, y_pos, 0.5, 0.5 - y_pos,
                    head_width=0.02, head_length=0.02,
                    fc='black', ec='black', alpha=0.3 + norm_weights[i] * 0.7
                )
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title
        title = f"Attention Flow for Sample {sample_idx}"
        if self.true_labels is not None:
            true_idx = self.true_labels[sample_idx] if len(self.true_labels.shape) == 1 else np.argmax(self.true_labels[sample_idx])
            true_label = self.class_names[true_idx] if self.class_names else f"Class {true_idx}"
            title += f"\nTrue Label: {true_label}"
        
        plt.title(title, fontsize=16)
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_top_attended_tokens(self, 
                              sample_idx: int,
                              top_n: int = 10) -> pd.DataFrame:
        """
        Get the top attended tokens for a single sample.
        
        Args:
            sample_idx: Index of the sample to analyze.
            top_n: Number of top tokens to return.
            
        Returns:
            DataFrame with top attended tokens.
        """
        # Get text and attention weights for the sample
        text = self.texts[sample_idx]
        attention = self.attention_weights[sample_idx]
        
        # Split text into tokens (simplified)
        tokens = text.split()
        
        # Ensure attention weights match the number of tokens
        if len(tokens) != len(attention):
            # Truncate or pad attention weights to match tokens
            if len(tokens) < len(attention):
                attention = attention[:len(tokens)]
            else:
                attention = np.pad(attention, (0, len(tokens) - len(attention)), 'constant')
        
        # Create token-weight pairs
        token_weights = [(token, weight) for token, weight in zip(tokens, attention)]
        
        # Sort by weight (descending)
        token_weights.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N tokens
        top_tokens = token_weights[:top_n]
        
        # Create DataFrame
        df = pd.DataFrame(top_tokens, columns=['Token', 'Attention Weight'])
        
        return df
    
    def analyze_attention_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in attention weights across the dataset.
        
        Returns:
            Dictionary with attention pattern analysis.
        """
        # This is a simplified implementation
        # In a real implementation, this would use more sophisticated analysis
        
        # Calculate average attention weight per token position
        avg_attention_by_position = np.mean(self.attention_weights, axis=0)
        
        # Find positions with highest average attention
        top_positions = np.argsort(avg_attention_by_position)[-5:]
        top_position_weights = avg_attention_by_position[top_positions]
        
        # Count tokens that receive highest attention
        token_attention = {}
        for i, text in enumerate(self.texts):
            tokens = text.split()
            for j, token in enumerate(tokens):
                if j < len(self.attention_weights[i]):
                    if token.lower() not in token_attention:
                        token_attention[token.lower()] = []
                    token_attention[token.lower()].append(self.attention_weights[i][j])
        
        # Calculate average attention per token
        avg_token_attention = {
            token: np.mean(weights)
            for token, weights in token_attention.items()
            if len(weights) >= 5  # Only consider tokens that appear at least 5 times
        }
        
        # Get top tokens by average attention
        top_tokens = sorted(avg_token_attention.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Return analysis results
        return {
            'avg_attention_by_position': avg_attention_by_position.tolist(),
            'top_positions': top_positions.tolist(),
            'top_position_weights': top_position_weights.tolist(),
            'top_tokens': top_tokens
        }
