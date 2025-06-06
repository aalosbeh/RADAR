"""
Ensemble model implementation for RADAR# project.

This module implements the ensemble component of the RADAR# model,
which combines predictions from multiple models for improved performance.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Average, Multiply
from typing import List, Dict, Optional, Union, Tuple

class EnsembleModel:
    """
    Ensemble model for Arabic text classification.
    
    This class implements an ensemble approach that combines predictions from
    multiple models, including CNN-BiLSTM and transformer models, for improved
    radicalization detection performance.
    
    Attributes:
        models (List[Model]): List of base models to ensemble.
        ensemble_method (str): Method for combining model predictions.
        weights (List[float]): Weights for each model in weighted ensemble.
        num_classes (int): Number of output classes.
        model (Model): Keras model instance.
    """
    
    def __init__(self, 
                 models: List[Model],
                 ensemble_method: str = 'weighted',
                 weights: Optional[List[float]] = None,
                 num_classes: int = 2):
        """
        Initialize the ensemble model.
        
        Args:
            models: List of base models to ensemble.
            ensemble_method: Method for combining model predictions ('weighted', 'average', 'stacking').
            weights: Weights for each model in weighted ensemble.
            num_classes: Number of output classes.
        """
        self.models = models
        self.ensemble_method = ensemble_method
        self.num_classes = num_classes
        self.model = None
        
        # Set weights for weighted ensemble
        if weights is None and ensemble_method == 'weighted':
            # Default to equal weights
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
    
    def build_model(self) -> Model:
        """
        Build the ensemble model.
        
        Returns:
            Built Keras model.
        """
        # Get inputs from all base models
        all_inputs = []
        for model in self.models:
            all_inputs.extend(model.inputs)
        
        # Get outputs from all base models
        model_outputs = [model.outputs[0] for model in self.models]
        
        # Apply ensemble method
        if self.ensemble_method == 'weighted':
            # Weighted average of model outputs
            weighted_outputs = []
            for i, output in enumerate(model_outputs):
                weighted_output = Multiply()([output, tf.constant(self.weights[i])])
                weighted_outputs.append(weighted_output)
            
            ensemble_output = tf.add_n(weighted_outputs)
        
        elif self.ensemble_method == 'average':
            # Simple average of model outputs
            ensemble_output = Average()(model_outputs)
        
        elif self.ensemble_method == 'stacking':
            # Stacking: use a meta-learner to combine model outputs
            if len(model_outputs) > 1:
                concatenated = Concatenate()(model_outputs)
            else:
                concatenated = model_outputs[0]
            
            # Meta-learner (simple MLP)
            x = Dense(64, activation='relu')(concatenated)
            
            if self.num_classes == 2:
                ensemble_output = Dense(1, activation='sigmoid')(x)
            else:
                ensemble_output = Dense(self.num_classes, activation='softmax')(x)
        
        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")
        
        # Create ensemble model
        model = Model(inputs=all_inputs, outputs=ensemble_output)
        
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
    
    def predict(self, inputs: List[tf.Tensor], **kwargs) -> tf.Tensor:
        """
        Make predictions with the ensemble model.
        
        Args:
            inputs: List of input tensors for all base models.
            **kwargs: Additional arguments for the predict method.
            
        Returns:
            Prediction tensor.
        """
        if self.model is None:
            self.build_model()
        
        return self.model.predict(inputs, **kwargs)
    
    def evaluate(self, inputs: List[tf.Tensor], labels: tf.Tensor, **kwargs) -> List[float]:
        """
        Evaluate the ensemble model.
        
        Args:
            inputs: List of input tensors for all base models.
            labels: Ground truth labels.
            **kwargs: Additional arguments for the evaluate method.
            
        Returns:
            List of evaluation metrics.
        """
        if self.model is None:
            self.build_model()
        
        return self.model.evaluate(inputs, labels, **kwargs)
