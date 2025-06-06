"""
Generate visualization of the RADAR# model architecture
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
import os

def create_model_architecture_diagram(output_path):
    """
    Create a detailed visualization of the RADAR# model architecture
    
    Args:
        output_path: Path to save the diagram
        
    Returns:
        Path to the saved diagram
    """
    # Create figure
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    
    # Define colors
    colors = {
        'input': '#E6F2FF',
        'embedding': '#CCEBFF',
        'cnn': '#99D6FF',
        'lstm': '#66C2FF',
        'attention': '#33ADFF',
        'transformer': '#FFD699',
        'ensemble': '#FFCC80',
        'output': '#B3E6CC',
        'arrow': '#404040'
    }
    
    # Define component positions and sizes
    components = {
        'input': {'x': 0.2, 'y': 0.9, 'width': 0.2, 'height': 0.06, 'color': colors['input']},
        'embedding': {'x': 0.2, 'y': 0.8, 'width': 0.2, 'height': 0.06, 'color': colors['embedding']},
        'cnn1': {'x': 0.15, 'y': 0.7, 'width': 0.08, 'height': 0.06, 'color': colors['cnn']},
        'cnn2': {'x': 0.26, 'y': 0.7, 'width': 0.08, 'height': 0.06, 'color': colors['cnn']},
        'cnn3': {'x': 0.37, 'y': 0.7, 'width': 0.08, 'height': 0.06, 'color': colors['cnn']},
        'concat': {'x': 0.2, 'y': 0.6, 'width': 0.2, 'height': 0.06, 'color': colors['cnn']},
        'bilstm': {'x': 0.2, 'y': 0.5, 'width': 0.2, 'height': 0.06, 'color': colors['lstm']},
        'attention': {'x': 0.2, 'y': 0.4, 'width': 0.2, 'height': 0.06, 'color': colors['attention']},
        'dense': {'x': 0.2, 'y': 0.3, 'width': 0.2, 'height': 0.06, 'color': colors['output']},
        'output_cnn_lstm': {'x': 0.2, 'y': 0.2, 'width': 0.2, 'height': 0.06, 'color': colors['output']},
        
        'arabert_input': {'x': 0.7, 'y': 0.9, 'width': 0.2, 'height': 0.06, 'color': colors['input']},
        'arabert': {'x': 0.7, 'y': 0.8, 'width': 0.2, 'height': 0.12, 'color': colors['transformer']},
        'arabert_output': {'x': 0.7, 'y': 0.65, 'width': 0.2, 'height': 0.06, 'color': colors['output']},
        
        'marbert_input': {'x': 0.7, 'y': 0.55, 'width': 0.2, 'height': 0.06, 'color': colors['input']},
        'marbert': {'x': 0.7, 'y': 0.45, 'width': 0.2, 'height': 0.12, 'color': colors['transformer']},
        'marbert_output': {'x': 0.7, 'y': 0.3, 'width': 0.2, 'height': 0.06, 'color': colors['output']},
        
        'ensemble': {'x': 0.45, 'y': 0.15, 'width': 0.2, 'height': 0.08, 'color': colors['ensemble']},
        'final_output': {'x': 0.45, 'y': 0.05, 'width': 0.2, 'height': 0.06, 'color': colors['output']}
    }
    
    # Draw components
    for name, comp in components.items():
        ax.add_patch(Rectangle(
            (comp['x'], comp['y']), comp['width'], comp['height'],
            facecolor=comp['color'], edgecolor='black', alpha=0.8, zorder=1
        ))
        
        # Add text
        if 'cnn' in name and name != 'concat':
            plt.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, 
                    f"CNN\nFilter {name[-1]}", ha='center', va='center', fontsize=9)
        elif name == 'concat':
            plt.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, 
                    "Concatenate", ha='center', va='center', fontsize=10)
        elif name == 'bilstm':
            plt.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, 
                    "Bidirectional LSTM", ha='center', va='center', fontsize=10)
        elif name == 'attention':
            plt.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, 
                    "Attention Layer", ha='center', va='center', fontsize=10)
        elif name == 'dense':
            plt.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, 
                    "Dense Layer", ha='center', va='center', fontsize=10)
        elif name == 'output_cnn_lstm':
            plt.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, 
                    "CNN-Bi-LSTM Output", ha='center', va='center', fontsize=10)
        elif name == 'arabert':
            plt.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, 
                    "AraBERT\nTransformer Model", ha='center', va='center', fontsize=10)
        elif name == 'arabert_input':
            plt.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, 
                    "AraBERT Input", ha='center', va='center', fontsize=10)
        elif name == 'arabert_output':
            plt.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, 
                    "AraBERT Output", ha='center', va='center', fontsize=10)
        elif name == 'marbert':
            plt.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, 
                    "MARBERT\nTransformer Model", ha='center', va='center', fontsize=10)
        elif name == 'marbert_input':
            plt.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, 
                    "MARBERT Input", ha='center', va='center', fontsize=10)
        elif name == 'marbert_output':
            plt.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, 
                    "MARBERT Output", ha='center', va='center', fontsize=10)
        elif name == 'ensemble':
            plt.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, 
                    "Weighted Ensemble", ha='center', va='center', fontsize=11, fontweight='bold')
        elif name == 'final_output':
            plt.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, 
                    "Final Classification", ha='center', va='center', fontsize=10)
        elif name == 'input':
            plt.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, 
                    "Input Text", ha='center', va='center', fontsize=10)
        elif name == 'embedding':
            plt.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2, 
                    "Embedding Layer", ha='center', va='center', fontsize=10)
    
    # Draw arrows
    arrows = [
        # CNN-Bi-LSTM path
        ('input', 'embedding'),
        ('embedding', 'cnn1'),
        ('embedding', 'cnn2'),
        ('embedding', 'cnn3'),
        ('cnn1', 'concat'),
        ('cnn2', 'concat'),
        ('cnn3', 'concat'),
        ('concat', 'bilstm'),
        ('bilstm', 'attention'),
        ('attention', 'dense'),
        ('dense', 'output_cnn_lstm'),
        ('output_cnn_lstm', 'ensemble'),
        
        # AraBERT path
        ('arabert_input', 'arabert'),
        ('arabert', 'arabert_output'),
        ('arabert_output', 'ensemble'),
        
        # MARBERT path
        ('marbert_input', 'marbert'),
        ('marbert', 'marbert_output'),
        ('marbert_output', 'ensemble'),
        
        # Final output
        ('ensemble', 'final_output')
    ]
    
    for start, end in arrows:
        start_comp = components[start]
        end_comp = components[end]
        
        # Calculate start and end points
        if start in ['cnn1', 'cnn2', 'cnn3'] and end == 'concat':
            # Special case for CNN to concat
            start_x = start_comp['x'] + start_comp['width']/2
            start_y = start_comp['y']
            end_x = end_comp['x'] + end_comp['width']/2
            end_y = end_comp['y'] + end_comp['height']
        elif start == 'output_cnn_lstm' and end == 'ensemble':
            # Special case for CNN-Bi-LSTM to ensemble
            start_x = start_comp['x'] + start_comp['width']
            start_y = start_comp['y'] + start_comp['height']/2
            end_x = end_comp['x']
            end_y = end_comp['y'] + end_comp['height']/2
        elif start == 'arabert_output' and end == 'ensemble':
            # Special case for AraBERT to ensemble
            start_x = start_comp['x'] + start_comp['width']/2
            start_y = start_comp['y']
            end_x = end_comp['x'] + end_comp['width']*0.75
            end_y = end_comp['y'] + end_comp['height']
        elif start == 'marbert_output' and end == 'ensemble':
            # Special case for MARBERT to ensemble
            start_x = start_comp['x'] + start_comp['width']/2
            start_y = start_comp['y']
            end_x = end_comp['x'] + end_comp['width']*0.25
            end_y = end_comp['y'] + end_comp['height']
        elif start_comp['y'] > end_comp['y']:
            # Vertical arrow down
            start_x = start_comp['x'] + start_comp['width']/2
            start_y = start_comp['y']
            end_x = end_comp['x'] + end_comp['width']/2
            end_y = end_comp['y'] + end_comp['height']
        else:
            # Vertical arrow up
            start_x = start_comp['x'] + start_comp['width']/2
            start_y = start_comp['y'] + start_comp['height']
            end_x = end_comp['x'] + end_comp['width']/2
            end_y = end_comp['y']
        
        # Draw arrow
        arrow = FancyArrowPatch(
            (start_x, start_y), (end_x, end_y),
            arrowstyle='-|>', color=colors['arrow'], linewidth=1.5,
            connectionstyle='arc3,rad=0.0', zorder=0
        )
        ax.add_patch(arrow)
    
    # Add model title
    plt.text(0.5, 0.97, 'RADAR# Model Architecture', ha='center', va='center', 
             fontsize=16, fontweight='bold')
    
    # Add section labels
    plt.text(0.2, 0.96, 'CNN-Bi-LSTM Branch', ha='center', va='center', 
             fontsize=12, fontweight='bold', color='#0066CC')
    plt.text(0.7, 0.96, 'Transformer Branches', ha='center', va='center', 
             fontsize=12, fontweight='bold', color='#CC6600')
    
    # Add legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor=colors['input'], edgecolor='black', alpha=0.8, label='Input'),
        Rectangle((0, 0), 1, 1, facecolor=colors['embedding'], edgecolor='black', alpha=0.8, label='Embedding'),
        Rectangle((0, 0), 1, 1, facecolor=colors['cnn'], edgecolor='black', alpha=0.8, label='CNN Layers'),
        Rectangle((0, 0), 1, 1, facecolor=colors['lstm'], edgecolor='black', alpha=0.8, label='LSTM Layer'),
        Rectangle((0, 0), 1, 1, facecolor=colors['attention'], edgecolor='black', alpha=0.8, label='Attention Layer'),
        Rectangle((0, 0), 1, 1, facecolor=colors['transformer'], edgecolor='black', alpha=0.8, label='Transformer Models'),
        Rectangle((0, 0), 1, 1, facecolor=colors['ensemble'], edgecolor='black', alpha=0.8, label='Ensemble Layer'),
        Rectangle((0, 0), 1, 1, facecolor=colors['output'], edgecolor='black', alpha=0.8, label='Output Layers')
    ]
    
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=4, fontsize=10)
    
    # Set axis limits and remove ticks
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_preprocessing_pipeline_diagram(output_path):
    """
    Create a visualization of the Arabic text preprocessing pipeline
    
    Args:
        output_path: Path to save the diagram
        
    Returns:
        Path to the saved diagram
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Define colors
    colors = {
        'raw': '#FFE6CC',
        'clean': '#CCFFCC',
        'normalize': '#CCFFFF',
        'segment': '#CCCCFF',
        'stopwords': '#FFCCFF',
        'lemmatize': '#FFFFCC',
        'tokenize': '#E6CCFF',
        'embed': '#FFCCE6',
        'arrow': '#404040'
    }
    
    # Define steps and example text
    steps = [
        {'name': 'Raw Text', 'color': colors['raw'], 
         'text': 'داعش يعلن مسؤوليته عن الهجوم #الإرهابي @user https://example.com'},
        {'name': 'Text Cleaning', 'color': colors['clean'], 
         'text': 'داعش يعلن مسؤوليته عن الهجوم الإرهابي'},
        {'name': 'Normalization', 'color': colors['normalize'], 
         'text': 'داعش يعلن مسووليته عن الهجوم الارهابي'},
        {'name': 'Segmentation', 'color': colors['segment'], 
         'text': 'داعش يعلن مسوول يه عن ال هجوم ال ارهاب ي'},
        {'name': 'Stopword Removal', 'color': colors['stopwords'], 
         'text': 'داعش يعلن مسوول هجوم ارهاب'},
        {'name': 'Lemmatization', 'color': colors['lemmatize'], 
         'text': 'داعش علن سؤول هجم رهب'},
        {'name': 'Tokenization', 'color': colors['tokenize'], 
         'text': '[داعش] [علن] [سؤول] [هجم] [رهب]'},
        {'name': 'Word Embedding', 'color': colors['embed'], 
         'text': '[0.2, -0.5, 0.1...], [0.7, 0.3, -0.2...], ...'}
    ]
    
    # Draw steps
    y_positions = np.linspace(0.9, 0.2, len(steps))
    box_height = 0.08
    
    for i, (step, y) in enumerate(zip(steps, y_positions)):
        # Draw box
        ax.add_patch(Rectangle(
            (0.1, y - box_height/2), 0.8, box_height,
            facecolor=step['color'], edgecolor='black', alpha=0.8, zorder=1
        ))
        
        # Add step name
        plt.text(0.2, y, step['name'], ha='center', va='center', 
                fontsize=12, fontweight='bold')
        
        # Add example text
        plt.text(0.6, y, step['text'], ha='center', va='center', 
                fontsize=10, fontfamily='sans-serif')
        
        # Add arrow to next step
        if i < len(steps) - 1:
            arrow = FancyArrowPatch(
                (0.5, y - box_height/2), (0.5, y_positions[i+1] + box_height/2),
                arrowstyle='-|>', color=colors['arrow'], linewidth=1.5,
                connectionstyle='arc3,rad=0.0', zorder=0
            )
            ax.add_patch(arrow)
    
    # Add title
    plt.text(0.5, 0.97, 'Arabic Text Preprocessing Pipeline for RADAR#', 
             ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Set axis limits and remove ticks
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_ensemble_approach_diagram(output_path):
    """
    Create a visualization of the ensemble approach
    
    Args:
        output_path: Path to save the diagram
        
    Returns:
        Path to the saved diagram
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Define colors
    colors = {
        'input': '#E6F2FF',
        'cnn_lstm': '#99D6FF',
        'arabert': '#FFD699',
        'marbert': '#FFCC80',
        'ensemble': '#B3E6CC',
        'output': '#CCFFCC',
        'arrow': '#404040'
    }
    
    # Define components
    components = {
        'input': {'x': 0.3, 'y': 0.85, 'width': 0.4, 'height': 0.08, 'color': colors['input']},
        
        'cnn_lstm': {'x': 0.1, 'y': 0.65, 'width': 0.2, 'height': 0.1, 'color': colors['cnn_lstm']},
        'arabert': {'x': 0.4, 'y': 0.65, 'width': 0.2, 'height': 0.1, 'color': colors['arabert']},
        'marbert': {'x': 0.7, 'y': 0.65, 'width': 0.2, 'height': 0.1, 'color': colors['marbert']},
        
        'cnn_lstm_out': {'x': 0.1, 'y': 0.45, 'width': 0.2, 'height': 0.08, 'color': colors['cnn_lstm']},
        'arabert_out': {'x': 0.4, 'y': 0.45, 'width': 0.2, 'height': 0.08, 'color': colors['arabert']},
        'marbert_out': {'x': 0.7, 'y': 0.45, 'width': 0.2, 'height': 0.08, 'color': colors['marbert']},
        
        'weight1': {'x': 0.15, 'y': 0.35, 'width': 0.1, 'height': 0.06, 'color': 'white'},
        'weight2': {'x': 0.45, 'y': 0.35, 'width': 0.1, 'height': 0.06, 'color': 'white'},
        'weight3': {'x': 0.75, 'y': 0.35, 'width': 0.1, 'height': 0.06, 'color': 'white'},
        
        'ensemble': {'x': 0.3, 'y': 0.2, 'width': 0.4, 'height': 0.1, 'color': colors['ensemble']},
        'output': {'x': 0.3, 'y': 0.05, 'width': 0.4, 'height': 0.08, 'color': colors['ou<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>