"""
Text preprocessing pipeline for IMDb sentiment analysis.
Includes text cleaning, TF-IDF vectorization, and data validation.
"""

import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
try:
    import contractions
except ImportError:
    contractions = None
    print("Warning: contractions module not available. Contraction expansion will be skipped.")

try:
    import nltk
    from nltk.corpus import stopwords
    # Download required NLTK data
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
except ImportError:
    nltk = None
    stopwords = None
    print("Warning: NLTK not available. Stopword removal will be skipped.")

import warnings
warnings.filterwarnings('ignore')

class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for sentiment analysis.
    """
    
    def __init__(self, max_features=10000, min_df=5, ngram_range=(1, 2)):
        """
        Initialize the preprocessor with TF-IDF parameters.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
            min_df (int): Minimum document frequency for features
            ngram_range (tuple): Range of n-grams to extract
        """
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        
    def clean_text(self, text):
        """
        Clean and preprocess text data.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
            
        # Convert to string
        text = str(text)
        
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Expand contractions
        if contractions is not None:
            text = contractions.fix(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text, language='english'):
        """
        Remove stopwords from text.
        
        Args:
            text (str): Text to process
            language (str): Language for stopwords
            
        Returns:
            str: Text with stopwords removed
        """
        if stopwords is None:
            return text
        
        try:
            stop_words = set(stopwords.words(language))
            words = text.split()
            filtered_words = [word for word in words if word not in stop_words]
            return ' '.join(filtered_words)
        except:
            return text
    
    def preprocess_dataframe(self, df, text_column='review', label_column='sentiment', 
                           remove_stopwords_flag=True):
        """
        Preprocess entire dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            label_column (str): Name of label column
            remove_stopwords_flag (bool): Whether to remove stopwords
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        print("Starting data preprocessing...")
        
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Check for missing values
        missing_text = df_processed[text_column].isna().sum()
        missing_labels = df_processed[label_column].isna().sum()
        
        print(f"Missing text values: {missing_text}")
        print(f"Missing label values: {missing_labels}")
        
        # Drop rows with missing values
        if missing_text > 0 or missing_labels > 0:
            df_processed = df_processed.dropna(subset=[text_column, label_column])
            print(f"Dropped {missing_text + missing_labels} rows with missing values")
        
        # Clean text
        print("Cleaning text...")
        df_processed[text_column] = df_processed[text_column].apply(self.clean_text)
        
        # Remove stopwords if requested
        if remove_stopwords_flag:
            print("Removing stopwords...")
            df_processed[text_column] = df_processed[text_column].apply(
                lambda x: self.remove_stopwords(x)
            )
        
        # Check class balance
        class_counts = df_processed[label_column].value_counts()
        print(f"\nClass distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / len(df_processed)) * 100
            print(f"{class_name}: {count} ({percentage:.1f}%)")
        
        # Remove duplicates
        initial_length = len(df_processed)
        df_processed = df_processed.drop_duplicates(subset=[text_column])
        duplicates_removed = initial_length - len(df_processed)
        print(f"Removed {duplicates_removed} duplicate reviews")
        
        print(f"Final dataset size: {len(df_processed)}")
        
        return df_processed
    
    def create_tfidf_features(self, X_train, X_test=None, fit=True):
        """
        Create TF-IDF features from text data.
        
        Args:
            X_train (list/array): Training text data
            X_test (list/array): Test text data (optional)
            fit (bool): Whether to fit the vectorizer
            
        Returns:
            tuple: (X_train_tfidf, X_test_tfidf) or (X_train_tfidf, None)
        """
        print("Creating TF-IDF features...")
        
        if fit:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                ngram_range=self.ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
            print(f"Number of features: {len(self.vectorizer.get_feature_names_out())}")
        else:
            if self.vectorizer is None:
                raise ValueError("Vectorizer not fitted. Call with fit=True first.")
            X_train_tfidf = self.vectorizer.transform(X_train)
        
        if X_test is not None:
            X_test_tfidf = self.vectorizer.transform(X_test)
            return X_train_tfidf, X_test_tfidf
        
        return X_train_tfidf, None
    
    def prepare_data(self, df, text_column='review', label_column='sentiment', 
                    test_size=0.2, random_state=42, remove_stopwords_flag=True):
        """
        Complete data preparation pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            label_column (str): Name of label column
            test_size (float): Test set size
            random_state (int): Random seed
            remove_stopwords_flag (bool): Whether to remove stopwords
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, preprocessor)
        """
        # Preprocess dataframe
        df_processed = self.preprocess_dataframe(
            df, text_column, label_column, remove_stopwords_flag
        )
        
        # Split data
        X = df_processed[text_column]
        y = df_processed[label_column]
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, 
            stratify=y_encoded
        )
        
        print(f"\nData split:")
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Create TF-IDF features
        X_train_tfidf, X_test_tfidf = self.create_tfidf_features(
            X_train, X_test, fit=True
        )
        
        return X_train_tfidf, X_test_tfidf, y_train, y_test, self
    
    def get_feature_names(self):
        """
        Get feature names from the TF-IDF vectorizer.
        
        Returns:
            list: Feature names
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted yet.")
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_top_features(self, model, n_features=20, class_names=None):
        """
        Get top features for a trained model.
        
        Args:
            model: Trained model with feature_importances_ or coef_ attribute
            n_features (int): Number of top features to return
            class_names (list): Class names for multi-class
            
        Returns:
            dict: Top features for each class
        """
        if hasattr(model, 'coef_'):
            # For linear models like Logistic Regression
            if len(model.coef_.shape) == 1:
                # Binary classification
                feature_importance = np.abs(model.coef_[0])
            else:
                # Multi-class
                feature_importance = np.abs(model.coef_).mean(axis=0)
        elif hasattr(model, 'feature_importances_'):
            # For tree-based models
            feature_importance = model.feature_importances_
        else:
            raise ValueError("Model doesn't have coef_ or feature_importances_ attribute")
        
        feature_names = self.get_feature_names()
        
        # Get top features
        top_indices = np.argsort(feature_importance)[-n_features:][::-1]
        top_features = [(feature_names[i], feature_importance[i]) for i in top_indices]
        
        return top_features

def load_imdb_data(file_path):
    """
    Load IMDb dataset from CSV file.
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please ensure the IMDb dataset is available.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def validate_data(df, text_column='review', label_column='sentiment'):
    """
    Validate dataset for sentiment analysis.
    
    Args:
        df (pd.DataFrame): Dataset to validate
        text_column (str): Name of text column
        label_column (str): Name of label column
        
    Returns:
        bool: True if data is valid
    """
    print("Validating dataset...")
    
    # Check if required columns exist
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found")
        return False
    
    if label_column not in df.columns:
        print(f"Error: Column '{label_column}' not found")
        return False
    
    # Check for missing values
    missing_text = df[text_column].isna().sum()
    missing_labels = df[label_column].isna().sum()
    
    if missing_text > 0:
        print(f"Warning: {missing_text} missing values in text column")
    
    if missing_labels > 0:
        print(f"Warning: {missing_labels} missing values in label column")
    
    # Check class balance
    class_counts = df[label_column].value_counts()
    print(f"Class distribution: {dict(class_counts)}")
    
    # Check for very short reviews
    short_reviews = df[text_column].str.len() < 10
    if short_reviews.sum() > 0:
        print(f"Warning: {short_reviews.sum()} very short reviews (< 10 characters)")
    
    print("Data validation completed.")
    return True
