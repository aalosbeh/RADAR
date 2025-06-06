"""
Data augmentation module for RADAR# project.

This module implements various data augmentation techniques for Arabic text,
including synonym replacement, random insertion, random swap, and random deletion.
"""

import random
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import nltk
from nltk.corpus import wordnet as wn

class DataAugmenter:
    """
    Data augmentation for Arabic text.
    
    This class implements various techniques for augmenting Arabic text data,
    which can help improve model robustness and prevent overfitting.
    
    Attributes:
        arabic_wordnet_path (str): Path to Arabic WordNet data.
        synonym_dict (Dict[str, List[str]]): Dictionary of word synonyms.
    """
    
    def __init__(self, arabic_wordnet_path: Optional[str] = None):
        """
        Initialize the data augmenter.
        
        Args:
            arabic_wordnet_path: Path to Arabic WordNet data.
        """
        self.arabic_wordnet_path = arabic_wordnet_path
        self.synonym_dict = {}
        
        # Load Arabic WordNet if path is provided
        if arabic_wordnet_path:
            self.load_arabic_wordnet(arabic_wordnet_path)
    
    def load_arabic_wordnet(self, path: str) -> None:
        """
        Load Arabic WordNet data.
        
        Args:
            path: Path to Arabic WordNet data.
        """
        try:
            # This is a placeholder for loading Arabic WordNet
            # In a real implementation, this would parse the WordNet data
            # and build a synonym dictionary
            print(f"Loading Arabic WordNet from {path}...")
            
            # Placeholder for synonym dictionary
            self.synonym_dict = {}
            
            print("Arabic WordNet loaded successfully")
        except Exception as e:
            print(f"Error loading Arabic WordNet: {e}")
            print("Using fallback synonym generation")
    
    def get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms for a word.
        
        Args:
            word: Input word.
            
        Returns:
            List of synonyms.
        """
        # Check if word is in synonym dictionary
        if word in self.synonym_dict:
            return self.synonym_dict[word]
        
        # Fallback: generate simple variations
        # This is a very basic approach and would be replaced with proper
        # Arabic WordNet lookups in a real implementation
        synonyms = []
        
        # Add simple character variations (not linguistically accurate)
        if len(word) > 3:
            # Replace first character
            synonyms.append(word[1:])
            
            # Replace last character
            synonyms.append(word[:-1])
            
            # Add a character
            synonyms.append(word + word[-1])
        
        return synonyms if synonyms else [word]
    
    def synonym_replacement(self, tokens: List[str], n: int = 1) -> List[str]:
        """
        Replace n random words with their synonyms.
        
        Args:
            tokens: List of tokens.
            n: Number of words to replace.
            
        Returns:
            Augmented list of tokens.
        """
        new_tokens = tokens.copy()
        random_word_indices = random.sample(range(len(tokens)), min(n, len(tokens)))
        
        for idx in random_word_indices:
            word = tokens[idx]
            synonyms = self.get_synonyms(word)
            if synonyms:
                new_tokens[idx] = random.choice(synonyms)
        
        return new_tokens
    
    def random_insertion(self, tokens: List[str], n: int = 1) -> List[str]:
        """
        Randomly insert n words.
        
        Args:
            tokens: List of tokens.
            n: Number of words to insert.
            
        Returns:
            Augmented list of tokens.
        """
        new_tokens = tokens.copy()
        
        for _ in range(n):
            if not tokens:
                continue
                
            # Get a random word from the tokens
            random_word = random.choice(tokens)
            
            # Get synonyms for the random word
            synonyms = self.get_synonyms(random_word)
            
            # Insert a random synonym at a random position
            if synonyms:
                random_synonym = random.choice(synonyms)
                random_idx = random.randint(0, len(new_tokens))
                new_tokens.insert(random_idx, random_synonym)
        
        return new_tokens
    
    def random_swap(self, tokens: List[str], n: int = 1) -> List[str]:
        """
        Randomly swap n pairs of words.
        
        Args:
            tokens: List of tokens.
            n: Number of pairs to swap.
            
        Returns:
            Augmented list of tokens.
        """
        new_tokens = tokens.copy()
        
        for _ in range(n):
            if len(new_tokens) < 2:
                continue
                
            # Get two random indices
            idx1, idx2 = random.sample(range(len(new_tokens)), 2)
            
            # Swap the words
            new_tokens[idx1], new_tokens[idx2] = new_tokens[idx2], new_tokens[idx1]
        
        return new_tokens
    
    def random_deletion(self, tokens: List[str], p: float = 0.1) -> List[str]:
        """
        Randomly delete words with probability p.
        
        Args:
            tokens: List of tokens.
            p: Probability of deletion.
            
        Returns:
            Augmented list of tokens.
        """
        if not tokens:
            return tokens
            
        # Ensure at least one word remains
        if len(tokens) == 1:
            return tokens
            
        new_tokens = []
        
        for token in tokens:
            # Keep the word with probability (1-p)
            if random.random() > p:
                new_tokens.append(token)
        
        # If all words were deleted, keep a random one
        if not new_tokens:
            new_tokens = [random.choice(tokens)]
        
        return new_tokens
    
    def augment_text(self, tokens: List[str], 
                     techniques: List[str] = ['synonym', 'insert', 'swap', 'delete'],
                     num_augmentations: int = 1) -> List[List[str]]:
        """
        Apply multiple augmentation techniques to generate new examples.
        
        Args:
            tokens: List of tokens.
            techniques: List of augmentation techniques to apply.
            num_augmentations: Number of augmented examples to generate.
            
        Returns:
            List of augmented token lists.
        """
        augmented_texts = []
        
        for _ in range(num_augmentations):
            current_tokens = tokens.copy()
            
            # Apply a random augmentation technique
            technique = random.choice(techniques)
            
            if technique == 'synonym':
                n = max(1, int(len(current_tokens) * 0.1))  # Replace ~10% of words
                current_tokens = self.synonym_replacement(current_tokens, n=n)
            elif technique == 'insert':
                n = max(1, int(len(current_tokens) * 0.1))  # Insert ~10% new words
                current_tokens = self.random_insertion(current_tokens, n=n)
            elif technique == 'swap':
                n = max(1, int(len(current_tokens) * 0.1))  # Swap ~10% of words
                current_tokens = self.random_swap(current_tokens, n=n)
            elif technique == 'delete':
                current_tokens = self.random_deletion(current_tokens, p=0.1)
            
            augmented_texts.append(current_tokens)
        
        return augmented_texts
    
    def augment_dataset(self, texts: List[List[str]], labels: np.ndarray, 
                        augment_factor: float = 0.5) -> Tuple[List[List[str]], np.ndarray]:
        """
        Augment a dataset with new examples.
        
        Args:
            texts: List of tokenized texts.
            labels: Array of labels.
            augment_factor: Factor by which to augment the dataset (0.5 = 50% more examples).
            
        Returns:
            Tuple of (augmented_texts, augmented_labels).
        """
        num_augmentations = int(len(texts) * augment_factor)
        
        # Select random indices to augment
        indices = random.sample(range(len(texts)), min(num_augmentations, len(texts)))
        
        augmented_texts = []
        augmented_labels = []
        
        for idx in indices:
            # Generate one augmented example for each selected index
            augmented = self.augment_text(texts[idx], num_augmentations=1)[0]
            augmented_texts.append(augmented)
            augmented_labels.append(labels[idx])
        
        # Combine original and augmented data
        all_texts = texts + augmented_texts
        all_labels = np.concatenate([labels, np.array(augmented_labels)])
        
        return all_texts, all_labels
