"""
Hyperparameter optimization for RADAR# project.

This module implements hyperparameter optimization techniques for RADAR# models,
including grid search, random search, and Bayesian optimization.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from sklearn.model_selection import ParameterGrid, ParameterSampler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

class HyperparameterOptimizer:
    """
    Hyperparameter optimization for RADAR# models.
    
    This class implements various hyperparameter optimization techniques,
    including grid search, random search, and Bayesian optimization.
    
    Attributes:
        param_space (Dict): Parameter space to search.
        optimization_method (str): Optimization method to use.
        metric (str): Metric to optimize.
        direction (str): Direction of optimization ('minimize' or 'maximize').
        n_trials (int): Number of trials for random and Bayesian optimization.
        output_dir (str): Directory to save optimization results.
    """
    
    def __init__(self, 
                 param_space: Dict[str, Any],
                 optimization_method: str = 'grid',
                 metric: str = 'val_loss',
                 direction: str = 'minimize',
                 n_trials: int = 20,
                 output_dir: str = 'results/hyperparameters'):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            param_space: Parameter space to search.
            optimization_method: Optimization method ('grid', 'random', 'bayesian').
            metric: Metric to optimize.
            direction: Direction of optimization ('minimize' or 'maximize').
            n_trials: Number of trials for random and Bayesian optimization.
            output_dir: Directory to save optimization results.
        """
        self.param_space = param_space
        self.optimization_method = optimization_method.lower()
        self.metric = metric
        self.direction = direction.lower()
        self.n_trials = n_trials
        self.output_dir = output_dir
        self.results = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if Bayesian optimization is requested but not available
        if self.optimization_method == 'bayesian' and not OPTUNA_AVAILABLE:
            print("Warning: Optuna is not available. Falling back to random search.")
            self.optimization_method = 'random'
    
    def _create_model(self, params: Dict[str, Any]) -> tf.keras.Model:
        """
        Create a model with the given hyperparameters.
        
        This method should be overridden by subclasses to create a model
        with the specific architecture and hyperparameters.
        
        Args:
            params: Hyperparameters for the model.
            
        Returns:
            Created model.
        """
        raise NotImplementedError("Subclasses must implement _create_model")
    
    def _train_and_evaluate(self, 
                           model: tf.keras.Model,
                           train_data: Union[Tuple, tf.data.Dataset],
                           val_data: Union[Tuple, tf.data.Dataset],
                           params: Dict[str, Any]) -> Dict[str, float]:
        """
        Train and evaluate a model with the given hyperparameters.
        
        Args:
            model: Model to train and evaluate.
            train_data: Training data.
            val_data: Validation data.
            params: Hyperparameters for training.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Extract training parameters
        epochs = params.get('epochs', 10)
        batch_size = params.get('batch_size', 32)
        patience = params.get('patience', 5)
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            )
        ]
        
        # Train the model
        history = model.fit(
            train_data[0] if isinstance(train_data, tuple) else train_data,
            train_data[1] if isinstance(train_data, tuple) else None,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size if isinstance(train_data, tuple) else None,
            callbacks=callbacks,
            verbose=0
        )
        
        # Get the best metric value
        if self.direction == 'minimize':
            best_epoch = np.argmin(history.history[self.metric])
        else:
            best_epoch = np.argmax(history.history[self.metric])
        
        # Collect metrics
        metrics = {}
        for metric_name, values in history.history.items():
            metrics[metric_name] = float(values[best_epoch])
        
        return metrics
    
    def grid_search(self, 
                   train_data: Union[Tuple, tf.data.Dataset],
                   val_data: Union[Tuple, tf.data.Dataset]) -> pd.DataFrame:
        """
        Perform grid search for hyperparameter optimization.
        
        Args:
            train_data: Training data.
            val_data: Validation data.
            
        Returns:
            DataFrame of optimization results.
        """
        # Generate all parameter combinations
        param_grid = ParameterGrid(self.param_space)
        print(f"Grid search with {len(param_grid)} parameter combinations")
        
        # Iterate over all parameter combinations
        for i, params in enumerate(param_grid):
            print(f"Trial {i+1}/{len(param_grid)}: {params}")
            
            # Create and compile model
            model = self._create_model(params)
            
            # Train and evaluate model
            metrics = self._train_and_evaluate(model, train_data, val_data, params)
            
            # Store results
            result = {**params, **metrics}
            self.results.append(result)
            
            # Print current best result
            self._print_best_result()
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Save results
        self._save_results(results_df)
        
        return results_df
    
    def random_search(self, 
                     train_data: Union[Tuple, tf.data.Dataset],
                     val_data: Union[Tuple, tf.data.Dataset]) -> pd.DataFrame:
        """
        Perform random search for hyperparameter optimization.
        
        Args:
            train_data: Training data.
            val_data: Validation data.
            
        Returns:
            DataFrame of optimization results.
        """
        # Generate random parameter combinations
        param_samples = list(ParameterSampler(
            self.param_space, n_iter=self.n_trials, random_state=42
        ))
        print(f"Random search with {len(param_samples)} parameter combinations")
        
        # Iterate over parameter combinations
        for i, params in enumerate(param_samples):
            print(f"Trial {i+1}/{len(param_samples)}: {params}")
            
            # Create and compile model
            model = self._create_model(params)
            
            # Train and evaluate model
            metrics = self._train_and_evaluate(model, train_data, val_data, params)
            
            # Store results
            result = {**params, **metrics}
            self.results.append(result)
            
            # Print current best result
            self._print_best_result()
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Save results
        self._save_results(results_df)
        
        return results_df
    
    def bayesian_optimization(self, 
                             train_data: Union[Tuple, tf.data.Dataset],
                             val_data: Union[Tuple, tf.data.Dataset]) -> pd.DataFrame:
        """
        Perform Bayesian optimization for hyperparameter optimization.
        
        Args:
            train_data: Training data.
            val_data: Validation data.
            
        Returns:
            DataFrame of optimization results.
        """
        if not OPTUNA_AVAILABLE:
            print("Optuna is not available. Falling back to random search.")
            return self.random_search(train_data, val_data)
        
        # Define the objective function for Optuna
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_range in self.param_space.items():
                if isinstance(param_range, list):
                    if all(isinstance(x, int) for x in param_range):
                        params[param_name] = trial.suggest_int(
                            param_name, min(param_range), max(param_range)
                        )
                    elif all(isinstance(x, float) for x in param_range):
                        params[param_name] = trial.suggest_float(
                            param_name, min(param_range), max(param_range)
                        )
                    else:
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_range
                        )
                elif isinstance(param_range, tuple) and len(param_range) == 2:
                    if all(isinstance(x, int) for x in param_range):
                        params[param_name] = trial.suggest_int(
                            param_name, param_range[0], param_range[1]
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, param_range[0], param_range[1]
                        )
                else:
                    params[param_name] = trial.suggest_categorical(
                        param_name, [param_range]
                    )
            
            # Create and compile model
            model = self._create_model(params)
            
            # Train and evaluate model
            metrics = self._train_and_evaluate(model, train_data, val_data, params)
            
            # Store results
            result = {**params, **metrics}
            self.results.append(result)
            
            # Return the metric to optimize
            return metrics[self.metric]
        
        # Create Optuna study
        study_direction = 'minimize' if self.direction == 'minimize' else 'maximize'
        study = optuna.create_study(direction=study_direction)
        
        # Optimize
        print(f"Bayesian optimization with {self.n_trials} trials")
        study.optimize(objective, n_trials=self.n_trials)
        
        # Print best result
        print(f"Best trial: {study.best_trial.params}")
        print(f"Best {self.metric}: {study.best_trial.value}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Save results
        self._save_results(results_df)
        
        return results_df
    
    def optimize(self, 
                train_data: Union[Tuple, tf.data.Dataset],
                val_data: Union[Tuple, tf.data.Dataset]) -> pd.DataFrame:
        """
        Perform hyperparameter optimization using the specified method.
        
        Args:
            train_data: Training data.
            val_data: Validation data.
            
        Returns:
            DataFrame of optimization results.
        """
        if self.optimization_method == 'grid':
            return self.grid_search(train_data, val_data)
        elif self.optimization_method == 'random':
            return self.random_search(train_data, val_data)
        elif self.optimization_method == 'bayesian':
            return self.bayesian_optimization(train_data, val_data)
        else:
            raise ValueError(f"Unsupported optimization method: {self.optimization_method}")
    
    def _print_best_result(self):
        """
        Print the current best result.
        """
        if not self.results:
            return
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Get the best result
        if self.direction == 'minimize':
            best_idx = results_df[self.metric].idxmin()
        else:
            best_idx = results_df[self.metric].idxmax()
        
        best_result = results_df.iloc[best_idx].to_dict()
        
        print(f"Current best {self.metric}: {best_result[self.metric]}")
        print(f"Parameters: {', '.join([f'{k}={v}' for k, v in best_result.items() if k != self.metric])}")
    
    def _save_results(self, results_df: pd.DataFrame):
        """
        Save optimization results to disk.
        
        Args:
            results_df: DataFrame of optimization results.
        """
        # Save as CSV
        csv_path = os.path.join(self.output_dir, f"{self.optimization_method}_results.csv")
        results_df.to_csv(csv_path, index=False)
        
        # Save as JSON
        json_path = os.path.join(self.output_dir, f"{self.optimization_method}_results.json")
        results_df.to_json(json_path, orient='records', indent=4)
        
        print(f"Results saved to {csv_path} and {json_path}")
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best hyperparameters from the optimization results.
        
        Returns:
            Dictionary of best hyperparameters.
        """
        if not self.results:
            raise ValueError("No optimization results available")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Get the best result
        if self.direction == 'minimize':
            best_idx = results_df[self.metric].idxmin()
        else:
            best_idx = results_df[self.metric].idxmax()
        
        # Extract hyperparameters
        best_params = {}
        for param_name in self.param_space.keys():
            if param_name in results_df.columns:
                best_params[param_name] = results_df.iloc[best_idx][param_name]
        
        return best_params


class RADARHyperparameterOptimizer(HyperparameterOptimizer):
    """
    Hyperparameter optimizer specifically for RADAR# models.
    
    This class extends the base HyperparameterOptimizer with RADAR#-specific
    model creation and evaluation logic.
    """
    
    def __init__(self, 
                 param_space: Optional[Dict[str, Any]] = None,
                 optimization_method: str = 'grid',
                 metric: str = 'val_loss',
                 direction: str = 'minimize',
                 n_trials: int = 20,
                 output_dir: str = 'results/hyperparameters'):
        """
        Initialize the RADAR# hyperparameter optimizer.
        
        Args:
            param_space: Parameter space to search. If None, a default space is used.
            optimization_method: Optimization method ('grid', 'random', 'bayesian').
            metric: Metric to optimize.
            direction: Direction of optimization ('minimize' or 'maximize').
            n_trials: Number of trials for random and Bayesian optimization.
            output_dir: Directory to save optimization results.
        """
        # Default parameter space for RADAR# models
        if param_space is None:
            param_space = {
                'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
                'batch_size': [16, 32, 64],
                'dropout_rate': [0.2, 0.3, 0.4, 0.5],
                'lstm_units': [64, 128, 256],
                'num_filters': [64, 128, 256],
                'ensemble_weights': [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]
            }
        
        super().__init__(
            param_space=param_space,
            optimization_method=optimization_method,
            metric=metric,
            direction=direction,
            n_trials=n_trials,
            output_dir=output_dir
        )
    
    def _create_model(self, params: Dict[str, Any]) -> tf.keras.Model:
        """
        Create a RADAR# model with the given hyperparameters.
        
        Args:
            params: Hyperparameters for the model.
            
        Returns:
            Created RADAR# model.
        """
        from src.models.cnn_bilstm import CNNBiLSTM
        from src.models.transformer import TransformerModel
        from src.models.ensemble import EnsembleModel
        
        # Extract model hyperparameters
        vocab_size = params.get('vocab_size', 30000)
        embedding_dim = params.get('embedding_dim', 300)
        max_sequence_length = params.get('max_sequence_length', 100)
        lstm_units = params.get('lstm_units', 128)
        num_filters = params.get('num_filters', 128)
        dropout_rate = params.get('dropout_rate', 0.5)
        num_classes = params.get('num_classes', 2)
        ensemble_weights = params.get('ensemble_weights', (0.5, 0.5))
        
        # Create CNN-BiLSTM model
        cnn_bilstm = CNNBiLSTM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
            num_filters=num_filters,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            num_classes=num_classes
        )
        cnn_bilstm_model = cnn_bilstm.build_model()
        
        # Create transformer model
        transformer = TransformerModel(
            max_sequence_length=max_sequence_length,
            dropout_rate=dropout_rate,
            num_classes=num_classes
        )
        transformer_model = transformer.build_model()
        
        # Create ensemble model
        ensemble = EnsembleModel(
            models=[cnn_bilstm_model, transformer_model],
            ensemble_method='weighted',
            weights=list(ensemble_weights),
            num_classes=num_classes
        )
        ensemble_model = ensemble.build_model()
        
        # Compile model
        learning_rate = params.get('learning_rate', 0.001)
        ensemble_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return ensemble_model
