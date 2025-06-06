"""
Model trainer implementation for RADAR# project.

This module implements the training loop and utilities for training
RADAR# models for Arabic radicalization detection.
"""

import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from typing import List, Dict, Optional, Union, Tuple, Any, Callable

class ModelTrainer:
    """
    Trainer for RADAR# models.
    
    This class implements utilities for training, validating, and saving
    RADAR# models for Arabic radicalization detection.
    
    Attributes:
        model: Model instance to train.
        model_name (str): Name of the model.
        output_dir (str): Directory to save model checkpoints and logs.
        callbacks (List): List of Keras callbacks for training.
    """
    
    def __init__(self, 
                 model,
                 model_name: str = 'radar_model',
                 output_dir: str = 'results/models'):
        """
        Initialize the model trainer.
        
        Args:
            model: Model instance to train.
            model_name: Name of the model.
            output_dir: Directory to save model checkpoints and logs.
        """
        self.model = model
        self.model_name = model_name
        self.output_dir = output_dir
        self.callbacks = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def setup_callbacks(self, 
                        patience: int = 5,
                        min_delta: float = 0.001,
                        save_best_only: bool = True,
                        log_dir: Optional[str] = None,
                        custom_callbacks: Optional[List] = None) -> List:
        """
        Set up training callbacks.
        
        Args:
            patience: Number of epochs with no improvement after which training will be stopped.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            save_best_only: Whether to save only the best model.
            log_dir: Directory to save TensorBoard logs.
            custom_callbacks: Additional custom callbacks.
            
        Returns:
            List of Keras callbacks.
        """
        callbacks = []
        
        # Model checkpoint callback
        checkpoint_path = os.path.join(self.output_dir, f"{self.model_name}_best.h5")
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            verbose=1,
            save_best_only=save_best_only,
            save_weights_only=False,
            mode='min'
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction callback
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1,
            mode='min'
        )
        callbacks.append(reduce_lr)
        
        # CSV logger callback
        csv_log_path = os.path.join(self.output_dir, f"{self.model_name}_training_log.csv")
        csv_logger = CSVLogger(
            filename=csv_log_path,
            separator=',',
            append=False
        )
        callbacks.append(csv_logger)
        
        # TensorBoard callback
        if log_dir is not None:
            tensorboard_callback = TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=False,
                update_freq='epoch',
                profile_batch=0
            )
            callbacks.append(tensorboard_callback)
        
        # Add custom callbacks
        if custom_callbacks is not None:
            callbacks.extend(custom_callbacks)
        
        self.callbacks = callbacks
        return callbacks
    
    def train(self, 
              train_data: Union[Tuple, tf.data.Dataset],
              val_data: Union[Tuple, tf.data.Dataset],
              epochs: int = 10,
              batch_size: int = 32,
              callbacks: Optional[List] = None,
              class_weights: Optional[Dict] = None,
              verbose: int = 1) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            train_data: Training data as a tuple of (inputs, labels) or a TensorFlow dataset.
            val_data: Validation data as a tuple of (inputs, labels) or a TensorFlow dataset.
            epochs: Number of epochs to train.
            batch_size: Batch size for training.
            callbacks: List of Keras callbacks for training.
            class_weights: Class weights for imbalanced datasets.
            verbose: Verbosity mode.
            
        Returns:
            Training history.
        """
        # Use provided callbacks or set up default ones
        if callbacks is None:
            if not self.callbacks:
                self.setup_callbacks()
            callbacks = self.callbacks
        
        # Start training
        start_time = time.time()
        print(f"Starting training of {self.model_name}...")
        
        # Check if data is a TensorFlow dataset
        if isinstance(train_data, tf.data.Dataset):
            train_dataset = train_data
            val_dataset = val_data
            
            # Train the model
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=verbose
            )
        else:
            # Unpack data
            X_train, y_train = train_data
            X_val, y_val = val_data
            
            # Train the model
            history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=verbose
            )
        
        # Calculate training time
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save training history
        history_path = os.path.join(self.output_dir, f"{self.model_name}_history.json")
        with open(history_path, 'w') as f:
            history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
            json.dump(history_dict, f, indent=4)
        
        return history
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model. If None, a default path is used.
            
        Returns:
            Path where the model was saved.
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, f"{self.model_name}.h5")
        
        # Save the model
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        return filepath
    
    def save_model_architecture(self, filepath: Optional[str] = None) -> str:
        """
        Save the model architecture as JSON.
        
        Args:
            filepath: Path to save the model architecture. If None, a default path is used.
            
        Returns:
            Path where the model architecture was saved.
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, f"{self.model_name}_architecture.json")
        
        # Save the model architecture
        model_json = self.model.to_json()
        with open(filepath, 'w') as f:
            f.write(model_json)
        
        print(f"Model architecture saved to {filepath}")
        
        return filepath
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model.
        """
        # Load the model
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict(self, 
                data: Union[np.ndarray, List[np.ndarray], tf.data.Dataset],
                batch_size: int = 32,
                verbose: int = 1) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            data: Input data for prediction.
            batch_size: Batch size for prediction.
            verbose: Verbosity mode.
            
        Returns:
            Predictions as a numpy array.
        """
        # Make predictions
        predictions = self.model.predict(data, batch_size=batch_size, verbose=verbose)
        
        return predictions
