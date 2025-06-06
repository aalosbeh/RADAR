"""
Attention mechanism implementation for RADAR# project.

This module implements the attention mechanism used in the RADAR# model
to focus on the most relevant parts of the text for classification.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer
from typing import Tuple, Optional

class AttentionLayer(Layer):
    """
    Custom attention mechanism for sequence data.
    
    This layer implements a context-aware attention mechanism that allows the model
    to focus on the most relevant parts of the input sequence for classification.
    
    Attributes:
        W (tf.Variable): Weight matrix for attention calculation.
        b (tf.Variable): Bias vector for attention calculation.
        u (tf.Variable): Context vector for attention calculation.
    """
    
    def __init__(self, attention_dim: int = 100, **kwargs):
        """
        Initialize the attention layer.
        
        Args:
            attention_dim: Dimension of the attention space.
            **kwargs: Additional layer arguments.
        """
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        """
        Build the layer weights.
        
        Args:
            input_shape: Shape of the input tensor.
        """
        # Input shape: (batch_size, sequence_length, hidden_size)
        self.hidden_size = input_shape[-1]
        
        # Initialize weights
        self.W = self.add_weight(
            name="attention_weight",
            shape=(self.hidden_size, self.attention_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        
        self.b = self.add_weight(
            name="attention_bias",
            shape=(self.attention_dim,),
            initializer="zeros",
            trainable=True
        )
        
        self.u = self.add_weight(
            name="context_vector",
            shape=(self.attention_dim,),
            initializer="glorot_uniform",
            trainable=True
        )
        
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs, mask: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply attention mechanism to the input sequence.
        
        Args:
            inputs: Input sequence tensor of shape (batch_size, sequence_length, hidden_size).
            mask: Optional mask tensor of shape (batch_size, sequence_length).
            
        Returns:
            Tuple of (context_vector, attention_weights):
                - context_vector: Weighted sum of input sequence with shape (batch_size, hidden_size).
                - attention_weights: Attention weights with shape (batch_size, sequence_length, 1).
        """
        # Calculate attention scores
        # uit = tanh(W * hit + b)
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        
        # Calculate attention weights
        # at = softmax(uit * u)
        ait = tf.tensordot(uit, self.u, axes=1)
        
        # Apply mask if provided
        if mask is not None:
            # Add a large negative value to masked positions
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(mask, axis=-1)
            ait = ait * mask - 1e10 * (1 - mask)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(ait, axis=1)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)
        
        # Calculate context vector as weighted sum of input sequence
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        
        return context_vector, attention_weights
    
    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.
        
        Args:
            input_shape: Shape of the input tensor.
            
        Returns:
            Output shape of the layer.
        """
        return (input_shape[0], self.hidden_size), (input_shape[0], input_shape[1], 1)
    
    def get_config(self):
        """
        Get the layer configuration.
        
        Returns:
            Layer configuration dictionary.
        """
        config = super(AttentionLayer, self).get_config()
        config.update({
            'attention_dim': self.attention_dim
        })
        return config


class MultiHeadAttention(Layer):
    """
    Multi-head attention mechanism.
    
    This layer implements a multi-head attention mechanism that allows the model
    to jointly attend to information from different representation subspaces.
    
    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        dropout (float): Dropout rate.
    """
    
    def __init__(self, num_heads: int = 8, head_dim: int = 64, dropout: float = 0.1, **kwargs):
        """
        Initialize the multi-head attention layer.
        
        Args:
            num_heads: Number of attention heads.
            head_dim: Dimension of each attention head.
            dropout: Dropout rate.
            **kwargs: Additional layer arguments.
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.output_dim = num_heads * head_dim
        super(MultiHeadAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        """
        Build the layer weights.
        
        Args:
            input_shape: Shape of the input tensor.
        """
        # Input shape: (batch_size, sequence_length, hidden_size)
        self.hidden_size = input_shape[-1]
        
        # Query, Key, Value projections
        self.query_dense = tf.keras.layers.Dense(self.output_dim)
        self.key_dense = tf.keras.layers.Dense(self.output_dim)
        self.value_dense = tf.keras.layers.Dense(self.output_dim)
        
        # Output projection
        self.output_dense = tf.keras.layers.Dense(self.hidden_size)
        
        # Dropout layer
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)
        
        super(MultiHeadAttention, self).build(input_shape)
    
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, head_dim).
        
        Args:
            x: Input tensor.
            batch_size: Batch size.
            
        Returns:
            Tensor with shape (batch_size, num_heads, sequence_length, head_dim).
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs, mask=None, training=None):
        """
        Apply multi-head attention to the input sequence.
        
        Args:
            inputs: Input sequence tensor of shape (batch_size, sequence_length, hidden_size).
            mask: Optional mask tensor of shape (batch_size, sequence_length).
            training: Whether the call is in training mode.
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, hidden_size).
        """
        batch_size = tf.shape(inputs)[0]
        
        # Linear projections
        query = self.query_dense(inputs)  # (batch_size, seq_len, output_dim)
        key = self.key_dense(inputs)      # (batch_size, seq_len, output_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, output_dim)
        
        # Split heads
        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len, head_dim)
        key = self.split_heads(key, batch_size)      # (batch_size, num_heads, seq_len, head_dim)
        value = self.split_heads(value, batch_size)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Scaled dot-product attention
        # (batch_size, num_heads, seq_len_q, seq_len_k)
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        
        # Scale matmul_qk
        dk = tf.cast(self.head_dim, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Apply mask if provided
        if mask is not None:
            # Add a large negative value to masked positions
            mask = tf.cast(mask, tf.float32)
            mask = tf.expand_dims(tf.expand_dims(mask, 1), 1)  # (batch_size, 1, 1, seq_len)
            scaled_attention_logits += (1.0 - mask) * -1e9
        
        # Attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout_layer(attention_weights, training=training)
        
        # Output
        output = tf.matmul(attention_weights, value)  # (batch_size, num_heads, seq_len_q, head_dim)
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, head_dim)
        output = tf.reshape(output, (batch_size, -1, self.output_dim))  # (batch_size, seq_len_q, output_dim)
        
        output = self.output_dense(output)  # (batch_size, seq_len_q, hidden_size)
        
        return output
    
    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.
        
        Args:
            input_shape: Shape of the input tensor.
            
        Returns:
            Output shape of the layer.
        """
        return input_shape
    
    def get_config(self):
        """
        Get the layer configuration.
        
        Returns:
            Layer configuration dictionary.
        """
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'dropout': self.dropout
        })
        return config
