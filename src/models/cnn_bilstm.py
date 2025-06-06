"""
CNN-BiLSTM model implementation for RADAR# project.

This module implements the CNN-BiLSTM component of the RADAR# model,
which combines convolutional layers for local pattern extraction with
bidirectional LSTM for sequential context.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D, Dropout, 
    Bidirectional, LSTM, Dense, Concatenate, GlobalMaxPooling1D
)
from typing import List, Dict, Optional, Union, Tuple

class CNNBiLSTM:
    """
    CNN-BiLSTM model for Arabic text classification.
    
    This class implements a hybrid deep learning architecture that combines
    Convolutional Neural Networks (CNNs) for local pattern extraction with
    Bidirectional Long Short-Term Memory (BiLSTM) for capturing sequential context.
    
    Attributes:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of word embeddings.
        max_sequence_length (int): Maximum sequence length.
        embedding_matrix (np.ndarray): Pre-trained word embedding matrix.
        num_filters (int): Number of filters in CNN layers.
        filter_sizes (List[int]): Sizes of filters in CNN layers.
        lstm_units (int): Number of units in LSTM layers.
        dropout_rate (float): Dropout rate for regularization.
        num_classes (int): Number of output classes.
        model (Model): Keras model instance.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 300,
                 max_sequence_length: int = 100,
                 embedding_matrix: Optional[tf.Tensor] = None,
                 num_filters: int = 128,
                 filter_sizes: List[int] = [3, 4, 5],
                 lstm_units: int = 128,
                 dropout_rate: float = 0.5,
                 num_classes: int = 2):
        """
        Initialize the CNN-BiLSTM model.
        
        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimension of word embeddings.
            max_sequence_length: Maximum sequence length.
            embedding_matrix: Pre-trained word embedding matrix.
            num_filters: Number of filters in CNN layers.
            filter_sizes: Sizes of filters in CNN layers.
            lstm_units: Number of units in LSTM layers.
            dropout_rate: Dropout rate for regularization.
            num_classes: Number of output classes.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.embedding_matrix = embedding_matrix
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self, with_attention: bool = True) -> Model:
        """
        Build the CNN-BiLSTM model.
        
        Args:
            with_attention: Whether to include attention mechanism.
            
        Returns:
            Built Keras model.
        """
        # Input layer
        sequence_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        
        # Embedding layer
        if self.embedding_matrix is not None:
            embedding_layer = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                weights=[self.embedding_matrix],
                input_length=self.max_sequence_length,
                trainable=False
            )
        else:
            embedding_layer = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length
            )
        
        embedded_sequences = embedding_layer(sequence_input)
        
        # CNN branch
        conv_blocks = []
        for filter_size in self.filter_sizes:
            conv = Conv1D(
                filters=self.num_filters,
                kernel_size=filter_size,
                padding='valid',
                activation='relu',
                strides=1
            )(embedded_sequences)
            conv = MaxPooling1D(pool_size=self.max_sequence_length - filter_size + 1)(conv)
            conv = tf.squeeze(conv, axis=1)
            conv_blocks.append(conv)
        
        # Concatenate CNN outputs
        cnn_features = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        cnn_features = Dropout(self.dropout_rate)(cnn_features)
        
        # BiLSTM branch
        lstm = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(embedded_sequences)
        
        # Add attention if requested
        if with_attention:
            from .attention import AttentionLayer
            lstm, attention_weights = AttentionLayer()(lstm)
        else:
            lstm = GlobalMaxPooling1D()(lstm)
        
        lstm = Dropout(self.dropout_rate)(lstm)
        
        # Concatenate CNN and BiLSTM features
        x = Concatenate()([cnn_features, lstm])
        
        # Dense layers
        x = Dense(256, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(128, activation='relu')(x)
        
        # Output layer
        if self.num_classes == 2:
            output = Dense(1, activation='sigmoid')(x)
        else:
            output = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=sequence_input, outputs=output)
        
        self.model = model
        return model
    
    def compile_model(self, 
                      learning_rate: float = 0.001,
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
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
    
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
