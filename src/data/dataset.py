"""
Dataset handling module for RADAR# project.

This module implements dataset loading, preparation, and splitting
for Arabic text radicalization detection.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

class RadicalizationDataset:
    """
    Dataset handler for Arabic radicalization detection.
    
    This class handles loading, preparation, and splitting of the dataset
    for training and evaluation of radicalization detection models.
    
    Attributes:
        data_path (str): Path to the dataset file.
        preprocessor: Instance of ArabicPreprocessor for text preprocessing.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        metadata (Dict): Additional metadata about the dataset.
    """
    
    def __init__(self, data_path: str, preprocessor=None):
        """
        Initialize the dataset handler.
        
        Args:
            data_path: Path to the dataset file.
            preprocessor: Instance of ArabicPreprocessor for text preprocessing.
        """
        self.data_path = data_path
        self.preprocessor = preprocessor
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.metadata = {}
        
        # Load dataset if path is provided
        if data_path and os.path.exists(data_path):
            self.load_dataset(data_path)
    
    def load_dataset(self, data_path: str) -> pd.DataFrame:
        """
        Load the dataset from a file.
        
        Args:
            data_path: Path to the dataset file.
            
        Returns:
            Loaded dataset as a pandas DataFrame.
        """
        # Determine file format from extension
        _, ext = os.path.splitext(data_path)
        
        if ext.lower() == '.csv':
            df = pd.read_csv(data_path)
        elif ext.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(data_path)
        elif ext.lower() == '.json':
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        print(f"Loaded dataset with {len(df)} samples")
        
        # Store dataset metadata
        self.metadata['num_samples'] = len(df)
        self.metadata['columns'] = list(df.columns)
        
        # Check for class distribution if label column exists
        if 'label' in df.columns:
            class_dist = df['label'].value_counts(normalize=True) * 100
            self.metadata['class_distribution'] = class_dist.to_dict()
            print(f"Class distribution: {class_dist}")
        
        return df
    
    def prepare_dataset(self, df: pd.DataFrame, 
                        text_column: str = 'text', 
                        label_column: str = 'label',
                        test_size: float = 0.2,
                        val_size: float = 0.1,
                        random_state: int = 42,
                        fit_preprocessor: bool = True) -> None:
        """
        Prepare the dataset for training and evaluation.
        
        Args:
            df: Input DataFrame containing the dataset.
            text_column: Name of the column containing text data.
            label_column: Name of the column containing labels.
            test_size: Proportion of the dataset to include in the test split.
            val_size: Proportion of the training dataset to include in the validation split.
            random_state: Random state for reproducibility.
            fit_preprocessor: Whether to fit the preprocessor on the training data.
        """
        # Ensure required columns exist
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset")
        
        # Extract features and labels
        X = df[text_column].values
        y = df[label_column].values
        
        # Convert labels to categorical if they are not already
        if len(y.shape) == 1:
            # Get unique classes
            classes = np.unique(y)
            self.metadata['num_classes'] = len(classes)
            self.metadata['classes'] = classes.tolist()
            
            # Convert string labels to integers if necessary
            if y.dtype == object:
                label_map = {label: i for i, label in enumerate(classes)}
                y = np.array([label_map[label] for label in y])
                self.metadata['label_map'] = label_map
            
            # Convert to one-hot encoding for multi-class classification
            if len(classes) > 2:
                y = to_categorical(y, num_classes=len(classes))
        
        # Split into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.shape) == 1 else None
        )
        
        # Further split training data into training and validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=random_state,
            stratify=y_train_val if len(y_train_val.shape) == 1 else None
        )
        
        print(f"Split dataset into {len(X_train)} training, {len(X_val)} validation, and {len(X_test)} test samples")
        
        # Preprocess text data if preprocessor is provided
        if self.preprocessor:
            print("Preprocessing training data...")
            X_train_processed = self.preprocessor.preprocess_pipeline(X_train, fit=fit_preprocessor)
            
            print("Preprocessing validation data...")
            X_val_processed = self.preprocessor.preprocess_pipeline(X_val, fit=False)
            
            print("Preprocessing test data...")
            X_test_processed = self.preprocessor.preprocess_pipeline(X_test, fit=False)
            
            # Store processed data
            self.X_train = X_train_processed
            self.X_val = X_val_processed
            self.X_test = X_test_processed
        else:
            # Store raw data
            self.X_train = X_train
            self.X_val = X_val
            self.X_test = X_test
        
        # Store labels
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # Update metadata
        self.metadata['train_samples'] = len(X_train)
        self.metadata['val_samples'] = len(X_val)
        self.metadata['test_samples'] = len(X_test)
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the training data.
        
        Returns:
            Tuple of (features, labels) for training.
        """
        return self.X_train, self.y_train
    
    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the validation data.
        
        Returns:
            Tuple of (features, labels) for validation.
        """
        return self.X_val, self.y_val
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the test data.
        
        Returns:
            Tuple of (features, labels) for testing.
        """
        return self.X_test, self.y_test
    
    def get_tf_dataset(self, batch_size: int = 32, 
                       shuffle: bool = True,
                       cache: bool = True) -> Dict[str, tf.data.Dataset]:
        """
        Create TensorFlow datasets for training, validation, and testing.
        
        Args:
            batch_size: Batch size for the datasets.
            shuffle: Whether to shuffle the training dataset.
            cache: Whether to cache the datasets.
            
        Returns:
            Dictionary containing TensorFlow datasets for 'train', 'val', and 'test'.
        """
        datasets = {}
        
        # Create training dataset
        train_ds = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        if shuffle:
            train_ds = train_ds.shuffle(buffer_size=len(self.X_train))
        train_ds = train_ds.batch(batch_size)
        if cache:
            train_ds = train_ds.cache()
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        datasets['train'] = train_ds
        
        # Create validation dataset
        val_ds = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
        val_ds = val_ds.batch(batch_size)
        if cache:
            val_ds = val_ds.cache()
        datasets['val'] = val_ds
        
        # Create test dataset
        test_ds = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))
        test_ds = test_ds.batch(batch_size)
        if cache:
            test_ds = test_ds.cache()
        datasets['test'] = test_ds
        
        return datasets
    
    def save_processed_data(self, output_dir: str) -> None:
        """
        Save processed data to disk.
        
        Args:
            output_dir: Directory to save the processed data.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed features
        np.save(os.path.join(output_dir, 'X_train.npy'), self.X_train)
        np.save(os.path.join(output_dir, 'X_val.npy'), self.X_val)
        np.save(os.path.join(output_dir, 'X_test.npy'), self.X_test)
        
        # Save labels
        np.save(os.path.join(output_dir, 'y_train.npy'), self.y_train)
        np.save(os.path.join(output_dir, 'y_val.npy'), self.y_val)
        np.save(os.path.join(output_dir, 'y_test.npy'), self.y_test)
        
        # Save metadata
        pd.DataFrame([self.metadata]).to_json(os.path.join(output_dir, 'metadata.json'))
        
        print(f"Saved processed data to {output_dir}")
    
    def load_processed_data(self, input_dir: str) -> None:
        """
        Load processed data from disk.
        
        Args:
            input_dir: Directory containing the processed data.
        """
        # Load processed features
        self.X_train = np.load(os.path.join(input_dir, 'X_train.npy'))
        self.X_val = np.load(os.path.join(input_dir, 'X_val.npy'))
        self.X_test = np.load(os.path.join(input_dir, 'X_test.npy'))
        
        # Load labels
        self.y_train = np.load(os.path.join(input_dir, 'y_train.npy'))
        self.y_val = np.load(os.path.join(input_dir, 'y_val.npy'))
        self.y_test = np.load(os.path.join(input_dir, 'y_test.npy'))
        
        # Load metadata
        self.metadata = pd.read_json(os.path.join(input_dir, 'metadata.json')).iloc[0].to_dict()
        
        print(f"Loaded processed data from {input_dir}")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Validation samples: {len(self.X_val)}")
        print(f"Test samples: {len(self.X_test)}")
