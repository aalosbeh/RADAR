"""
Optimizer configuration for RADAR# project.

This module implements optimizer configurations and learning rate schedules
for training RADAR# models.
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PiecewiseConstantDecay
from typing import Dict, Any, Optional, Union, Callable

class OptimizerConfig:
    """
    Optimizer configuration for model training.
    
    This class provides utilities for creating and configuring optimizers
    and learning rate schedules for training RADAR# models.
    
    Attributes:
        optimizer_type (str): Type of optimizer to use.
        learning_rate (float): Initial learning rate.
        lr_schedule_type (str): Type of learning rate schedule.
        optimizer_params (Dict): Additional parameters for the optimizer.
        schedule_params (Dict): Parameters for the learning rate schedule.
    """
    
    def __init__(self, 
                 optimizer_type: str = 'adam',
                 learning_rate: float = 0.001,
                 lr_schedule_type: Optional[str] = None,
                 optimizer_params: Optional[Dict[str, Any]] = None,
                 schedule_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the optimizer configuration.
        
        Args:
            optimizer_type: Type of optimizer to use ('adam', 'sgd', 'rmsprop').
            learning_rate: Initial learning rate.
            lr_schedule_type: Type of learning rate schedule ('exponential', 'piecewise', None).
            optimizer_params: Additional parameters for the optimizer.
            schedule_params: Parameters for the learning rate schedule.
        """
        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.lr_schedule_type = lr_schedule_type
        self.optimizer_params = optimizer_params or {}
        self.schedule_params = schedule_params or {}
    
    def get_learning_rate_schedule(self) -> Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]:
        """
        Get the learning rate or learning rate schedule.
        
        Returns:
            Learning rate as a float or a learning rate schedule.
        """
        if self.lr_schedule_type is None:
            return self.learning_rate
        
        if self.lr_schedule_type == 'exponential':
            # Default parameters for exponential decay
            decay_rate = self.schedule_params.get('decay_rate', 0.9)
            decay_steps = self.schedule_params.get('decay_steps', 1000)
            staircase = self.schedule_params.get('staircase', False)
            
            return ExponentialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=staircase
            )
        
        elif self.lr_schedule_type == 'piecewise':
            # Default parameters for piecewise constant decay
            boundaries = self.schedule_params.get('boundaries', [1000, 2000])
            values = self.schedule_params.get('values', [self.learning_rate, self.learning_rate * 0.1, self.learning_rate * 0.01])
            
            return PiecewiseConstantDecay(
                boundaries=boundaries,
                values=values
            )
        
        else:
            raise ValueError(f"Unsupported learning rate schedule type: {self.lr_schedule_type}")
    
    def get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """
        Get the configured optimizer.
        
        Returns:
            Configured optimizer instance.
        """
        # Get learning rate or learning rate schedule
        lr = self.get_learning_rate_schedule()
        
        if self.optimizer_type == 'adam':
            # Default parameters for Adam optimizer
            beta_1 = self.optimizer_params.get('beta_1', 0.9)
            beta_2 = self.optimizer_params.get('beta_2', 0.999)
            epsilon = self.optimizer_params.get('epsilon', 1e-7)
            amsgrad = self.optimizer_params.get('amsgrad', False)
            
            return Adam(
                learning_rate=lr,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
                amsgrad=amsgrad
            )
        
        elif self.optimizer_type == 'sgd':
            # Default parameters for SGD optimizer
            momentum = self.optimizer_params.get('momentum', 0.0)
            nesterov = self.optimizer_params.get('nesterov', False)
            
            return SGD(
                learning_rate=lr,
                momentum=momentum,
                nesterov=nesterov
            )
        
        elif self.optimizer_type == 'rmsprop':
            # Default parameters for RMSprop optimizer
            rho = self.optimizer_params.get('rho', 0.9)
            momentum = self.optimizer_params.get('momentum', 0.0)
            epsilon = self.optimizer_params.get('epsilon', 1e-7)
            centered = self.optimizer_params.get('centered', False)
            
            return RMSprop(
                learning_rate=lr,
                rho=rho,
                momentum=momentum,
                epsilon=epsilon,
                centered=centered
            )
        
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'OptimizerConfig':
        """
        Create an optimizer configuration from a dictionary.
        
        Args:
            config: Dictionary containing optimizer configuration.
            
        Returns:
            OptimizerConfig instance.
        """
        return OptimizerConfig(
            optimizer_type=config.get('optimizer_type', 'adam'),
            learning_rate=config.get('learning_rate', 0.001),
            lr_schedule_type=config.get('lr_schedule_type', None),
            optimizer_params=config.get('optimizer_params', {}),
            schedule_params=config.get('schedule_params', {})
        )
    
    def to_config(self) -> Dict[str, Any]:
        """
        Convert the optimizer configuration to a dictionary.
        
        Returns:
            Dictionary containing optimizer configuration.
        """
        return {
            'optimizer_type': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'lr_schedule_type': self.lr_schedule_type,
            'optimizer_params': self.optimizer_params,
            'schedule_params': self.schedule_params
        }
    
    def __str__(self) -> str:
        """
        Get a string representation of the optimizer configuration.
        
        Returns:
            String representation.
        """
        return f"OptimizerConfig(type={self.optimizer_type}, lr={self.learning_rate}, schedule={self.lr_schedule_type})"
