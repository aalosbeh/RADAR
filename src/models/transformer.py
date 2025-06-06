"""
Transformer model implementation for RADAR# project.

This module implements the transformer component of the RADAR# model,
which integrates AraBERT, a transformer-based model pre-trained on Arabic text.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D, LayerNormalization
)
import transformers
from transformers import TFAutoModel, AutoTokenizer
from typing import List, Dict, Optional, Union, Tuple

class TransformerModel:
    """
    Transformer model for Arabic text classification.
    
    This class implements a wrapper around pre-trained transformer models
    like AraBERT for Arabic text classification tasks.
    
    Attributes:
        model_name (str): Name of the pre-trained transformer model.
        max_sequence_length (int): Maximum sequence length.
        dropout_rate (float): Dropout rate for regularization.
        num_classes (int): Number of output classes.
        tokenizer: Transformer tokenizer.
        transformer: Pre-trained transformer model.
        model (Model): Keras model instance.
    """
    
    def __init__(self, 
                 model_name: str = 'aubmindlab/bert-base-arabertv2',
                 max_sequence_length: int = 128,
                 dropout_rate: float = 0.1,
                 num_classes: int = 2):
        """
        Initialize the transformer model.
        
        Args:
            model_name: Name of the pre-trained transformer model.
            max_sequence_length: Maximum sequence length.
            dropout_rate: Dropout rate for regularization.
            num_classes: Number of output classes.
        """
        self.model_name = model_name
        self.max_sequence_length = max_sequence_length
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.tokenizer = None
        self.transformer = None
        self.model = None
        
        # Initialize tokenizer and transformer
        self.initialize_transformer()
    
    def initialize_transformer(self):
        """
        Initialize the transformer model and tokenizer.
        """
        try:
            print(f"Loading pre-trained transformer model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.transformer = TFAutoModel.from_pretrained(self.model_name)
            print("Transformer model loaded successfully")
        except Exception as e:
            print(f"Error loading transformer model: {e}")
            print("Using a placeholder transformer model")
            # Create a placeholder transformer model for testing
            self.create_placeholder_transformer()
    
    def create_placeholder_transformer(self):
        """
        Create a placeholder transformer model for testing.
        """
        # This is a simplified placeholder and not a real transformer
        # It's only used when the actual pre-trained model cannot be loaded
        vocab_size = 30000
        hidden_size = 768
        
        inputs = Input(shape=(self.max_sequence_length,), dtype=tf.int32)
        embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_size,
            input_length=self.max_sequence_length
        )(inputs)
        
        # Simplified transformer-like layers
        x = LayerNormalization(epsilon=1e-6)(embedding)
        x = tf.keras.layers.Dense(hidden_size, activation='gelu')(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Create a model that returns the sequence output and pooled output
        class PlaceholderTransformerOutput:
            def __init__(self, last_hidden_state, pooler_output):
                self.last_hidden_state = last_hidden_state
                self.pooler_output = pooler_output
        
        def call_fn(inputs):
            last_hidden_state = x
            pooler_output = GlobalAveragePooling1D()(x)
            return PlaceholderTransformerOutput(last_hidden_state, pooler_output)
        
        self.transformer = tf.keras.Model(inputs=inputs, outputs=call_fn(inputs))
        
        # Create a simple tokenizer function
        class PlaceholderTokenizer:
            def __call__(self, texts, padding=True, truncation=True, max_length=None, return_tensors=None):
                # Convert texts to integer sequences (dummy implementation)
                if isinstance(texts, str):
                    texts = [texts]
                
                sequences = []
                for text in texts:
                    # Convert characters to integers (simplified)
                    seq = [ord(c) % 30000 for c in text[:max_length]]
                    sequences.append(seq)
                
                # Pad sequences
                if padding:
                    max_len = max(len(seq) for seq in sequences)
                    sequences = [seq + [0] * (max_len - len(seq)) for seq in sequences]
                
                # Convert to tensors
                if return_tensors == 'tf':
                    return {'input_ids': tf.constant(sequences, dtype=tf.int32)}
                
                return {'input_ids': sequences}
        
        self.tokenizer = PlaceholderTokenizer()
    
    def build_model(self) -> Model:
        """
        Build the transformer-based classification model.
        
        Returns:
            Built Keras model.
        """
        # Input layers for transformer
        input_ids = Input(shape=(self.max_sequence_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_sequence_length,), dtype=tf.int32, name='attention_mask')
        
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=False
        )
        
        # Get the pooled output (CLS token)
        pooled_output = transformer_outputs.pooler_output
        
        # Add dropout for regularization
        x = Dropout(self.dropout_rate)(pooled_output)
        
        # Add classification head
        x = Dense(256, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        if self.num_classes == 2:
            output = Dense(1, activation='sigmoid', name='output')(x)
        else:
            output = Dense(self.num_classes, activation='softmax', name='output')(x)
        
        # Create model
        model = Model(
            inputs=[input_ids, attention_mask],
            outputs=output
        )
        
        self.model = model
        return model
    
    def compile_model(self, 
                      learning_rate: float = 2e-5,
                      loss: Optional[str] = None,
                      metrics: List[str] = ['accuracy']):
        """
        Compile the model with optimizer, loss function, and metrics.
        
        Args:
            learning_rate: Learning rate for the optimizer.
            loss: Loss function. If None, binary or categorical crossentropy is used based on num_classes.
            metrics: List of metrics to track.
        """
        if self.model is None:
            self.build_model()
        
        # Set default loss based on number of classes
        if loss is None:
            loss = 'binary_crossentropy' if self.num_classes == 2 else 'categorical_crossentropy'
        
        # Create optimizer with weight decay
        optimizer = transformers.optimization_tf.AdamWeightDecay(
            learning_rate=learning_rate,
            weight_decay_rate=0.01,
            epsilon=1e-6,
            exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias']
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def preprocess_text(self, texts: List[str]) -> Dict[str, tf.Tensor]:
        """
        Preprocess texts for the transformer model.
        
        Args:
            texts: List of input texts.
            
        Returns:
            Dictionary of preprocessed inputs for the transformer model.
        """
        # Tokenize texts
        encoded_inputs = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors='tf'
        )
        
        return encoded_inputs
    
    def summary(self):
        """
        Print model summary.
        """
        if self.model is None:
            self.build_model()
        
        self.model.summary()
    
    def get_model(self) -> Model:
        """
        Get the Keras model instance.
        
        Returns:
            Keras model instance.
        """
        if self.model is None:
            self.build_model()
        
        return self.model
