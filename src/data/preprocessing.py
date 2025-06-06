"""
Arabic text preprocessing module for RADAR# project.

This module implements a comprehensive preprocessing pipeline for Arabic text,
including normalization, stopword removal, stemming, and tokenization.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pyarabic.araby as araby
from farasa.segmenter import FarasaSegmenter
from gensim.models import KeyedVectors
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

class ArabicPreprocessor:
    """
    A comprehensive preprocessing pipeline for Arabic text.
    
    This class implements various preprocessing techniques for Arabic text,
    including normalization, stopword removal, stemming, and tokenization.
    It also provides methods for converting text to numerical representations
    suitable for deep learning models.
    
    Attributes:
        max_sequence_length (int): Maximum sequence length for padding.
        tokenizer (Tokenizer): Keras tokenizer for converting text to sequences.
        word_index (Dict[str, int]): Mapping from words to indices.
        embedding_dim (int): Dimension of word embeddings.
        embedding_matrix (np.ndarray): Pre-trained word embedding matrix.
        segmenter (FarasaSegmenter): Farasa segmenter for Arabic text.
        stop_words (List[str]): List of Arabic stop words.
    """
    
    def __init__(self, 
                 max_sequence_length: int = 100, 
                 embedding_dim: int = 300,
                 embedding_path: Optional[str] = None,
                 vocab_size: int = 30000,
                 use_farasa: bool = True):
        """
        Initialize the Arabic preprocessor.
        
        Args:
            max_sequence_length: Maximum sequence length for padding.
            embedding_dim: Dimension of word embeddings.
            embedding_path: Path to pre-trained word embeddings.
            vocab_size: Maximum vocabulary size.
            use_farasa: Whether to use Farasa segmenter for advanced processing.
        """
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        self.word_index = None
        self.embedding_matrix = None
        
        # Initialize Farasa segmenter if requested
        self.use_farasa = use_farasa
        if use_farasa:
            try:
                self.segmenter = FarasaSegmenter()
            except Exception as e:
                print(f"Warning: Could not initialize Farasa segmenter: {e}")
                print("Falling back to basic preprocessing")
                self.use_farasa = False
        
        # Load Arabic stop words
        try:
            self.stop_words = set(stopwords.words('arabic'))
        except:
            print("Warning: NLTK Arabic stopwords not found. Using a minimal set.")
            # Minimal set of common Arabic stop words
            self.stop_words = {
                'في', 'من', 'على', 'إلى', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
                'هو', 'هي', 'هم', 'هن', 'أنا', 'أنت', 'أنتم', 'نحن', 'كان', 'كانت',
                'و', 'أو', 'ثم', 'لكن', 'إذا', 'إن', 'لا', 'ما', 'لم', 'لن'
            }
        
        # Load pre-trained embeddings if provided
        if embedding_path:
            self.load_embeddings(embedding_path)
    
    def normalize_arabic(self, text: str) -> str:
        """
        Normalize Arabic text by removing diacritics, elongation, and standardizing characters.
        
        Args:
            text: Input Arabic text.
            
        Returns:
            Normalized Arabic text.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove diacritics (tashkeel)
        text = araby.strip_tashkeel(text)
        
        # Remove tatweel (elongation)
        text = araby.strip_tatweel(text)
        
        # Standardize hamza forms
        text = re.sub(r'[إأآا]', 'ا', text)
        text = re.sub(r'[ىئ]', 'ي', text)
        text = re.sub(r'ؤ', 'و', text)
        text = re.sub(r'ة', 'ه', text)
        
        # Remove non-Arabic characters and extra spaces
        text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove Arabic stop words from a list of tokens.
        
        Args:
            tokens: List of Arabic tokens.
            
        Returns:
            List of tokens with stop words removed.
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_arabic(self, tokens: List[str]) -> List[str]:
        """
        Apply light stemming to Arabic tokens.
        
        Args:
            tokens: List of Arabic tokens.
            
        Returns:
            List of stemmed tokens.
        """
        stemmed_tokens = []
        
        for token in tokens:
            # Remove common prefixes
            if len(token) > 3:
                if token.startswith('ال'):
                    token = token[2:]
                elif token.startswith(('و', 'ف', 'ب', 'ل', 'ك')):
                    token = token[1:]
            
            # Remove common suffixes
            if len(token) > 3:
                if token.endswith(('ها', 'هم', 'هن', 'كم', 'كن', 'نا')):
                    token = token[:-2]
                elif token.endswith(('ه', 'ك', 'ي', 'ت')):
                    token = token[:-1]
            
            stemmed_tokens.append(token)
        
        return stemmed_tokens
    
    def segment_with_farasa(self, text: str) -> str:
        """
        Segment Arabic text using Farasa segmenter.
        
        Args:
            text: Input Arabic text.
            
        Returns:
            Segmented text.
        """
        if not self.use_farasa:
            return text
        
        try:
            return self.segmenter.segment(text)
        except Exception as e:
            print(f"Warning: Farasa segmentation failed: {e}")
            return text
    
    def preprocess_text(self, text: str, remove_stops: bool = True, 
                        apply_stemming: bool = True) -> List[str]:
        """
        Apply full preprocessing pipeline to Arabic text.
        
        Args:
            text: Input Arabic text.
            remove_stops: Whether to remove stop words.
            apply_stemming: Whether to apply stemming.
            
        Returns:
            List of preprocessed tokens.
        """
        if not text or not isinstance(text, str):
            return []
        
        # Normalize text
        text = self.normalize_arabic(text)
        
        # Apply Farasa segmentation if available
        if self.use_farasa:
            text = self.segment_with_farasa(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words if requested
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        
        # Apply stemming if requested
        if apply_stemming:
            tokens = self.stem_arabic(tokens)
        
        return tokens
    
    def fit_tokenizer(self, texts: List[str]):
        """
        Fit the tokenizer on a list of texts.
        
        Args:
            texts: List of preprocessed texts.
        """
        self.tokenizer.fit_on_texts(texts)
        self.word_index = self.tokenizer.word_index
        print(f"Vocabulary size: {len(self.word_index)}")
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        Convert a list of texts to sequences of indices.
        
        Args:
            texts: List of preprocessed texts.
            
        Returns:
            List of sequences of indices.
        """
        return self.tokenizer.texts_to_sequences(texts)
    
    def pad_sequences(self, sequences: List[List[int]]) -> np.ndarray:
        """
        Pad sequences to the same length.
        
        Args:
            sequences: List of sequences of indices.
            
        Returns:
            Padded sequences as a numpy array.
        """
        return pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
    
    def load_embeddings(self, embedding_path: str):
        """
        Load pre-trained word embeddings.
        
        Args:
            embedding_path: Path to pre-trained word embeddings.
        """
        try:
            print(f"Loading word embeddings from {embedding_path}...")
            word_vectors = KeyedVectors.load_word2vec_format(embedding_path, binary=False)
            
            # Initialize embedding matrix
            self.embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
            
            # Fill embedding matrix with pre-trained embeddings
            for word, i in self.word_index.items():
                if i >= self.vocab_size:
                    continue
                try:
                    embedding_vector = word_vectors[word]
                    self.embedding_matrix[i] = embedding_vector
                except KeyError:
                    # Word not in embedding vocabulary
                    pass
            
            print(f"Loaded {len(word_vectors.key_to_index)} word vectors.")
        except Exception as e:
            print(f"Error loading word embeddings: {e}")
    
    def get_embedding_matrix(self) -> np.ndarray:
        """
        Get the embedding matrix.
        
        Returns:
            Embedding matrix as a numpy array.
        """
        if self.embedding_matrix is None:
            print("Warning: Embedding matrix not initialized. Returning random embeddings.")
            self.embedding_matrix = np.random.uniform(-0.25, 0.25, 
                                                     (self.vocab_size, self.embedding_dim))
        return self.embedding_matrix
    
    def preprocess_pipeline(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """
        Apply full preprocessing pipeline to a list of texts.
        
        Args:
            texts: List of raw Arabic texts.
            fit: Whether to fit the tokenizer on these texts.
            
        Returns:
            Preprocessed texts as padded sequences.
        """
        # Preprocess each text
        preprocessed_texts = [' '.join(self.preprocess_text(text)) for text in texts]
        
        # Fit tokenizer if requested
        if fit:
            self.fit_tokenizer(preprocessed_texts)
        
        # Convert to sequences and pad
        sequences = self.texts_to_sequences(preprocessed_texts)
        padded_sequences = self.pad_sequences(sequences)
        
        return padded_sequences
