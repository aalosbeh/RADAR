"""
Arabic Text Preprocessing Module for RADAR#
This module implements enhanced preprocessing techniques for Arabic text analysis,
specifically optimized for radicalization detection in social media content.
"""

import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# For Arabic-specific processing
# Note: In a real implementation, you would need to install these packages
# pip install pyarabic farasapy camel-tools

class ArabicPreprocessor:
    """
    Enhanced Arabic text preprocessor for RADAR# model
    Implements specialized techniques for handling Arabic text in social media
    """
    
    def __init__(self, max_sequence_length=100, max_num_words=20000, use_farasa=True):
        """
        Initialize the preprocessor with configuration parameters
        
        Args:
            max_sequence_length: Maximum length of text sequences
            max_num_words: Maximum size of vocabulary
            use_farasa: Whether to use Farasa segmenter for tokenization
        """
        self.max_sequence_length = max_sequence_length
        self.max_num_words = max_num_words
        self.use_farasa = use_farasa
        self.tokenizer = None
        
        # Load Arabic stopwords (placeholder - would use actual library in implementation)
        self.arabic_stopwords = self._load_arabic_stopwords()
        
        # Initialize Farasa segmenter if enabled
        if self.use_farasa:
            try:
                # In actual implementation, would import and initialize Farasa
                # from farasa.segmenter import FarasaSegmenter
                # self.farasa_segmenter = FarasaSegmenter()
                pass
            except ImportError:
                print("Farasa not available. Falling back to standard tokenization.")
                self.use_farasa = False
    
    def _load_arabic_stopwords(self):
        """Load Arabic stopwords from resources"""
        # In actual implementation, would load from a file or library
        # Example stopwords (abbreviated list)
        return [
            "من", "إلى", "عن", "على", "في", "هو", "هي", "هم", "انت", "انا", 
            "نحن", "هذا", "هذه", "ذلك", "تلك", "اي", "كل", "بعض", "غير", "لا", 
            "ما", "مع", "عند", "عندما", "قد", "لقد", "كان", "كانت", "يكون"
        ]
    
    def clean_text(self, text):
        """
        Clean Arabic text by removing noise and standardizing characters
        
        Args:
            text: Input Arabic text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (keep the text without # symbol)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove non-Arabic characters except spaces
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        
        # Standardize Arabic characters
        # Replace آ with ا
        text = re.sub(r'آ', 'ا', text)
        # Replace إ with ا
        text = re.sub(r'إ', 'ا', text)
        # Replace أ with ا
        text = re.sub(r'أ', 'ا', text)
        # Replace ة with ه
        text = re.sub(r'ة', 'ه', text)
        # Replace ى with ي
        text = re.sub(r'ى', 'ي', text)
        
        # Remove diacritics (tashkeel)
        text = re.sub(r'[\u064B-\u065F]', '', text)
        
        # Remove elongation (tatweel)
        text = re.sub(r'ـ', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def remove_stopwords(self, text):
        """
        Remove Arabic stopwords from text
        
        Args:
            text: Input Arabic text
            
        Returns:
            Text with stopwords removed
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.arabic_stopwords]
        return ' '.join(filtered_words)
    
    def segment_with_farasa(self, text):
        """
        Segment Arabic text using Farasa segmenter
        
        Args:
            text: Input Arabic text
            
        Returns:
            Segmented text
        """
        if not self.use_farasa:
            return text
        
        # In actual implementation, would use Farasa
        # return self.farasa_segmenter.segment(text)
        return text  # Placeholder
    
    def preprocess_text(self, text, remove_stops=True):
        """
        Apply full preprocessing pipeline to Arabic text
        
        Args:
            text: Input Arabic text
            remove_stops: Whether to remove stopwords
            
        Returns:
            Fully preprocessed text
        """
        # Clean the text
        text = self.clean_text(text)
        
        # Segment with Farasa if enabled
        if self.use_farasa:
            text = self.segment_with_farasa(text)
        
        # Remove stopwords if enabled
        if remove_stops:
            text = self.remove_stopwords(text)
        
        return text
    
    def fit_tokenizer(self, texts):
        """
        Fit tokenizer on corpus of texts
        
        Args:
            texts: List of Arabic texts
            
        Returns:
            Fitted tokenizer
        """
        self.tokenizer = Tokenizer(num_words=self.max_num_words)
        self.tokenizer.fit_on_texts(texts)
        return self.tokenizer
    
    def texts_to_sequences(self, texts):
        """
        Convert texts to padded sequences
        
        Args:
            texts: List of Arabic texts
            
        Returns:
            Padded sequences
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted. Call fit_tokenizer first.")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)
        return padded_sequences
    
    def prepare_data(self, texts, labels, test_size=0.2, validation_size=0.1, random_state=42):
        """
        Prepare data for model training
        
        Args:
            texts: List of Arabic texts
            labels: List of corresponding labels
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train, validation, and test data
        """
        # Preprocess all texts
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            preprocessed_texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Split train into train and validation
        if validation_size > 0:
            val_size = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
            )
        else:
            X_val, y_val = [], []
        
        # Fit tokenizer on training data only
        self.fit_tokenizer(X_train)
        
        # Convert to sequences
        X_train_seq = self.texts_to_sequences(X_train)
        X_val_seq = self.texts_to_sequences(X_val) if len(X_val) > 0 else []
        X_test_seq = self.texts_to_sequences(X_test)
        
        return {
            'X_train': X_train_seq,
            'y_train': np.array(y_train),
            'X_val': X_val_seq,
            'y_val': np.array(y_val) if len(y_val) > 0 else [],
            'X_test': X_test_seq,
            'y_test': np.array(y_test),
            'tokenizer': self.tokenizer,
            'word_index': self.tokenizer.word_index,
            'preprocessed_texts': {
                'train': X_train,
                'val': X_val,
                'test': X_test
            }
        }


# Example usage
if __name__ == "__main__":
    # Sample data (would be loaded from dataset in actual implementation)
    sample_texts = [
        "هذا مثال لتغريدة عادية عن الطقس اليوم",
        "داعش يعلن مسؤوليته عن الهجوم الإرهابي",
        "أحب بلدي وأتمنى السلام للجميع"
    ]
    sample_labels = [0, 1, 0]  # 0: normal, 1: extremist
    
    # Initialize preprocessor
    preprocessor = ArabicPreprocessor(max_sequence_length=50, max_num_words=5000)
    
    # Preprocess sample texts
    for text in sample_texts:
        print(f"Original: {text}")
        print(f"Preprocessed: {preprocessor.preprocess_text(text)}")
        print("-" * 50)
    
    # Prepare data
    data = preprocessor.prepare_data(sample_texts, sample_labels)
    
    # Print some statistics
    print(f"Vocabulary size: {len(data['word_index'])}")
    print(f"Training samples: {len(data['X_train'])}")
    print(f"Test samples: {len(data['X_test'])}")
