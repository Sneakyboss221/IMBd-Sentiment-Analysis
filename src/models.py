"""
Model training and ensemble implementation for IMDb sentiment analysis.
Includes Logistic Regression, LinearSVM, MultinomialNB, and ensemble methods.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from scipy.special import expit  # sigmoid function
import warnings
warnings.filterwarnings('ignore')

class LinearSVCWrapper:
    """
    Wrapper class for LinearSVC to make it compatible with VotingClassifier.
    Adds predict_proba method using decision_function and sigmoid conversion.
    """
    
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Convert decision_function to probabilities using sigmoid (expit).
        """
        decision_scores = self.model.decision_function(X)
        # Convert decision scores to probabilities using expit (sigmoid)
        prob_positive = expit(decision_scores)
        prob_negative = 1 - prob_positive
        return np.column_stack([prob_negative, prob_positive])
    
    def __getattr__(self, name):
        """
        Delegate all other attributes to the wrapped model.
        """
        if name in ['model', '__getattr__']:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        try:
            return getattr(self.model, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

class CustomVotingClassifier:
    """
    Custom VotingClassifier that handles LinearSVC properly.
    """
    
    def __init__(self, estimators, voting='soft', weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.fitted_estimators = []
        
    def fit(self, X, y):
        """
        Fit all estimators.
        """
        self.fitted_estimators = []
        for name, estimator in self.estimators:
            print(f"Fitting {name}...")
            fitted_estimator = estimator.fit(X, y)
            self.fitted_estimators.append((name, fitted_estimator))
        return self
    
    def predict(self, X):
        """
        Make predictions using voting.
        """
        if self.voting == 'soft':
            return self._predict_soft(X)
        else:
            return self._predict_hard(X)
    
    def predict_proba(self, X):
        """
        Get probability predictions.
        """
        if self.voting == 'soft':
            return self._predict_soft_proba(X)
        else:
            raise ValueError("predict_proba is not available when voting='hard'")
    
    def _predict_soft(self, X):
        """
        Soft voting prediction.
        """
        proba = self._predict_soft_proba(X)
        return np.argmax(proba, axis=1)
    
    def _predict_soft_proba(self, X):
        """
        Soft voting probability prediction.
        """
        probabilities = []
        
        for name, estimator in self.fitted_estimators:
            if hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X)
            else:
                # Handle LinearSVC without predict_proba
                decision_scores = estimator.decision_function(X)
                prob_positive = expit(decision_scores)
                prob_negative = 1 - prob_positive
                proba = np.column_stack([prob_negative, prob_positive])
            
            probabilities.append(proba)
        
        # Average probabilities
        if self.weights is None:
            avg_proba = np.mean(probabilities, axis=0)
        else:
            weighted_proba = np.average(probabilities, axis=0, weights=self.weights)
            avg_proba = weighted_proba
        
        return avg_proba
    
    def _predict_hard(self, X):
        """
        Hard voting prediction.
        """
        predictions = []
        for name, estimator in self.fitted_estimators:
            pred = estimator.predict(X)
            predictions.append(pred)
        
        # Majority vote
        predictions = np.array(predictions)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

class ModelTrainer:
    """
    Comprehensive model training and ensemble implementation.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the model trainer.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.ensemble = None
        
    def train_logistic_regression(self, X_train, y_train, X_test, y_test, 
                                 param_grid=None, cv=10, filepath_prefix='models/'):
        """
        Train Logistic Regression model with hyperparameter tuning.
        Loads existing model if available, otherwise trains new one.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            param_grid (dict): Parameter grid for tuning
            cv (int): Cross-validation folds (default 10 for StratifiedKFold)
            filepath_prefix (str): Prefix for model file paths
            
        Returns:
            dict: Training results
        """
        # Check if model already exists
        if self.load_single_model('logistic_regression', filepath_prefix):
            # Model was loaded, get results from loaded model
            if 'logistic_regression' in self.results:
                print("✅ Using existing Logistic Regression results")
                return self.results['logistic_regression']
            else:
                # Need to evaluate loaded model
                print("📊 Evaluating loaded Logistic Regression model...")
                model = self.models['logistic_regression']
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                results = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'best_params': 'loaded_model',
                    'cv_score': 'loaded_model'
                }
                
                self.results['logistic_regression'] = results
                print(f"✅ Loaded Logistic Regression - F1: {results['f1_score']:.3f}, Accuracy: {results['accuracy']:.3f}")
                return results
        
        print("Training Logistic Regression...")
        
        if param_grid is None:
            # Expanded hyperparameter grid as requested
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 50],
                'penalty': ['l2', 'elasticnet'],
                'solver': ['saga'],
                'class_weight': [None, 'balanced']
            }
        
        # Create model
        lr = LogisticRegression(random_state=self.random_state, max_iter=10000)
        
        # Use StratifiedKFold for cross-validation
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Grid search with StratifiedKFold
        grid_search = GridSearchCV(
            lr, param_grid, cv=cv_strategy, scoring='f1', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_lr = grid_search.best_estimator_
        self.models['logistic_regression'] = best_lr
        
        # Make predictions
        y_pred = best_lr.predict(X_test)
        y_pred_proba = best_lr.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results = {
            'model': best_lr,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_
        }
        
        self.results['logistic_regression'] = results
        
        print(f"✅ Logistic Regression - F1: {results['f1_score']:.3f}, Accuracy: {results['accuracy']:.3f}")
        print(f"Best parameters: {results['best_params']}")
        print(f"Cross-validation score: {results['cv_score']:.3f}")
        
        # Save the trained model
        import os
        os.makedirs(filepath_prefix, exist_ok=True)
        filename = f"{filepath_prefix}logistic_regression.joblib"
        joblib.dump(best_lr, filename)
        print(f"💾 Saved Logistic Regression model to {filename}")
        
        return results
    
    def train_svm(self, X_train, y_train, X_test, y_test, 
                  param_grid=None, cv=5, filepath_prefix='models/'):
        """
        Train LinearSVM model with hyperparameter tuning.
        Loads existing model if available, otherwise trains new one.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            param_grid (dict): Parameter grid for tuning
            cv (int): Cross-validation folds
            filepath_prefix (str): Prefix for model file paths
            
        Returns:
            dict: Training results
        """
        # Check if model already exists
        if self.load_single_model('svm', filepath_prefix):
            # Model was loaded, get results from loaded model
            if 'svm' in self.results:
                print("✅ Using existing LinearSVM results")
                return self.results['svm']
            else:
                # Need to evaluate loaded model
                print("📊 Evaluating loaded LinearSVM model...")
                model = self.models['svm']
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                results = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'best_params': 'loaded_model',
                    'cv_score': 'loaded_model'
                }
                
                self.results['svm'] = results
                print(f"✅ Loaded LinearSVM - F1: {results['f1_score']:.3f}, Accuracy: {results['accuracy']:.3f}")
                return results
        
        print("Training LinearSVM...")
        
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10],
                'loss': ['squared_hinge'],
                'dual': [False]
            }
        
        # Create model
        svm = LinearSVC(random_state=self.random_state, max_iter=10000)
        
        # Grid search
        grid_search = GridSearchCV(
            svm, param_grid, cv=cv, scoring='f1', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model and wrap it for ensemble compatibility
        best_svm = grid_search.best_estimator_
        # Wrap LinearSVC to add predict_proba method for ensemble
        wrapped_svm = LinearSVCWrapper(best_svm)
        self.models['svm'] = wrapped_svm
        
        # Make predictions using wrapped model
        y_pred = wrapped_svm.predict(X_test)
        y_pred_proba = wrapped_svm.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results = {
            'model': wrapped_svm,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_
        }
        
        self.results['svm'] = results
        
        print(f"✅ LinearSVM - F1: {results['f1_score']:.3f}, Accuracy: {results['accuracy']:.3f}")
        print(f"Best parameters: {results['best_params']}")
        
        # Save the trained model
        import os
        os.makedirs(filepath_prefix, exist_ok=True)
        filename = f"{filepath_prefix}svm.joblib"
        joblib.dump(wrapped_svm, filename)
        print(f"💾 Saved LinearSVM model to {filename}")
        
        return results
    
    def train_multinomial_nb(self, X_train, y_train, X_test, y_test, 
                             param_grid=None, cv=5, filepath_prefix='models/'):
        """
        Train Multinomial Naive Bayes model with hyperparameter tuning.
        Loads existing model if available, otherwise trains new one.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            param_grid (dict): Parameter grid for tuning
            cv (int): Cross-validation folds
            filepath_prefix (str): Prefix for model file paths
            
        Returns:
            dict: Training results
        """
        # Check if model already exists
        if self.load_single_model('multinomial_nb', filepath_prefix):
            # Model was loaded, get results from loaded model
            if 'multinomial_nb' in self.results:
                print("✅ Using existing MultinomialNB results")
                return self.results['multinomial_nb']
            else:
                # Need to evaluate loaded model
                print("📊 Evaluating loaded MultinomialNB model...")
                model = self.models['multinomial_nb']
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                results = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'best_params': 'loaded_model',
                    'cv_score': 'loaded_model'
                }
                
                self.results['multinomial_nb'] = results
                print(f"✅ Loaded MultinomialNB - F1: {results['f1_score']:.3f}, Accuracy: {results['accuracy']:.3f}")
                return results
        
        print("Training Multinomial Naive Bayes...")
        
        if param_grid is None:
            param_grid = {
                'alpha': [0.1, 1.0, 10.0]
            }
        
        # Create model
        nb_model = MultinomialNB()
        
        # Grid search
        grid_search = GridSearchCV(
            nb_model, param_grid, cv=cv, 
            scoring='f1', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_nb = grid_search.best_estimator_
        self.models['multinomial_nb'] = best_nb
        
        # Make predictions
        y_pred = best_nb.predict(X_test)
        y_pred_proba = best_nb.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results = {
            'model': best_nb,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_
        }
        
        self.results['multinomial_nb'] = results
        
        print(f"✅ MultinomialNB - F1: {results['f1_score']:.3f}, Accuracy: {results['accuracy']:.3f}")
        print(f"Best parameters: {results['best_params']}")
        
        # Save the trained model
        import os
        os.makedirs(filepath_prefix, exist_ok=True)
        filename = f"{filepath_prefix}multinomial_nb.joblib"
        joblib.dump(best_nb, filename)
        print(f"💾 Saved MultinomialNB model to {filename}")
        
        return results
    
    def create_ensemble(self, X_train, y_train, X_test, y_test, 
                       voting='soft', weights=None, filepath_prefix='models/'):
        """
        Create ensemble model using voting classifier.
        Loads existing ensemble if available, otherwise creates new one.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            voting (str): Voting strategy ('soft' or 'hard')
            weights (list): Weights for each model
            filepath_prefix (str): Prefix for model file paths
            
        Returns:
            dict: Ensemble results
        """
        # Always (re)create optimized ensemble per current configuration
        import os
        ensemble_filename = f"{filepath_prefix}ensemble.joblib"
        print("Creating optimized ensemble (LR + SVM, soft voting 50/50)...")
        
        # Prepare estimators: only keep Logistic Regression and SVM
        estimators = []
        for name in ['logistic_regression', 'svm']:
            if name in self.models:
                estimators.append((name, self.models[name]))
        
        if not estimators:
            raise ValueError("No eligible models found. Load or train Logistic Regression and SVM first.")

        # Default to equal weights for LR and SVM if not provided
        if weights is None:
            weights = [0.5] * len(estimators)
        
        # Create custom ensemble that handles LinearSVC properly
        self.ensemble = CustomVotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights
        )
        
        # Train ensemble
        self.ensemble.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.ensemble.predict(X_test)
        y_pred_proba = self.ensemble.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        results = {
            'model': self.ensemble,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'voting': voting,
            'weights': weights
        }
        
        self.results['ensemble'] = results
        
        print(f"✅ Ensemble - F1: {results['f1_score']:.3f}, Accuracy: {results['accuracy']:.3f}")
        
        # Save the ensemble model (overwrite existing)
        os.makedirs(filepath_prefix, exist_ok=True)
        joblib.dump(self.ensemble, ensemble_filename)
        print(f"💾 Saved Ensemble model to {ensemble_filename}")
        
        return results
    
    def compare_models(self):
        """
        Compare performance of all trained models.
        
        Returns:
            pd.DataFrame: Comparison results
        """
        if not self.results:
            print("No models trained yet.")
            return None
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': results['accuracy'],
                'F1-Score': results['f1_score'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'ROC-AUC': results['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        print("\n📊 Model Comparison:")
        print("=" * 80)
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        return comparison_df
    
    def get_feature_importance(self, model_name, feature_names, top_n=20):
        """
        Get feature importance for a specific model.
        
        Args:
            model_name (str): Name of the model
            feature_names (list): List of feature names
            top_n (int): Number of top features to return
            
        Returns:
            dict: Feature importance results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model = self.models[model_name]
        
        if hasattr(model, 'coef_'):
            # For linear models
            if len(model.coef_.shape) == 1:
                importance = np.abs(model.coef_[0])
            else:
                importance = np.abs(model.coef_).mean(axis=0)
        elif hasattr(model, 'feature_importances_'):
            # For tree-based models
            importance = model.feature_importances_
        else:
            raise ValueError(f"Model '{model_name}' doesn't support feature importance.")
        
        # Get top features
        top_indices = np.argsort(importance)[-top_n:][::-1]
        top_features = {
            'features': [feature_names[i] for i in top_indices],
            'importance': [importance[i] for i in top_indices]
        }
        
        return top_features
    
    def save_models(self, filepath_prefix='models/'):
        """
        Save all trained models automatically.
        
        Args:
            filepath_prefix (str): Prefix for model file paths
        """
        import os
        os.makedirs(filepath_prefix, exist_ok=True)
        
        # Save individual models with proper naming
        model_mappings = {
            'logistic_regression': 'logistic_regression.joblib',
            'svm': 'svm.joblib', 
            'multinomial_nb': 'multinomial_nb.joblib'
        }
        
        for name, model in self.models.items():
            if name in model_mappings:
                filename = f"{filepath_prefix}{model_mappings[name]}"
                joblib.dump(model, filename)
                print(f"✅ Saved {name} to {filename}")
            else:
                filename = f"{filepath_prefix}{name}.joblib"
                joblib.dump(model, filename)
                print(f"✅ Saved {name} to {filename}")
        
        # Save ensemble
        if self.ensemble is not None:
            filename = f"{filepath_prefix}ensemble.joblib"
            joblib.dump(self.ensemble, filename)
            print(f"✅ Saved ensemble to {filename}")
        
        # Save results
        results_filename = f"{filepath_prefix}results.joblib"
        joblib.dump(self.results, results_filename)
        print(f"✅ Saved results to {results_filename}")
        
        print(f"\n📁 All models saved to {filepath_prefix} directory")
    
    def save_all_models(self, trained_models_dict=None, filepath_prefix='models/'):
        """
        Helper function to save all models from a dictionary.
        
        Args:
            trained_models_dict (dict): Dictionary of trained models
            filepath_prefix (str): Prefix for model file paths
        """
        if trained_models_dict is None:
            trained_models_dict = self.models
            
        import os
        os.makedirs(filepath_prefix, exist_ok=True)
        
        print("💾 Saving all trained models...")
        
        for name, model in trained_models_dict.items():
            filename = f"{filepath_prefix}{name}.joblib"
            joblib.dump(model, filename)
            print(f"✅ Saved {name} to {filename}")
        
        print(f"📁 All models saved to {filepath_prefix} directory")
    
    def load_models(self, filepath_prefix='models/'):
        """
        Load previously saved models.
        
        Args:
            filepath_prefix (str): Prefix for model file paths
        """
        import os
        
        # Load individual models
        for name in ['logistic_regression', 'svm', 'multinomial_nb']:
            filename = f"{filepath_prefix}{name}.joblib"
            if os.path.exists(filename):
                self.models[name] = joblib.load(filename)
                print(f"✅ Loaded existing {name} model from {filename}")
        
        # Load ensemble
        ensemble_filename = f"{filepath_prefix}ensemble.joblib"
        if os.path.exists(ensemble_filename):
            self.ensemble = joblib.load(ensemble_filename)
            print(f"✅ Loaded existing ensemble from {ensemble_filename}")
        
        # Load results
        results_filename = f"{filepath_prefix}results.joblib"
        if os.path.exists(results_filename):
            self.results = joblib.load(results_filename)
            print(f"✅ Loaded existing results from {results_filename}")
    
    def check_model_exists(self, model_name, filepath_prefix='models/'):
        """
        Check if a model file exists.
        
        Args:
            model_name (str): Name of the model
            filepath_prefix (str): Prefix for model file paths
            
        Returns:
            bool: True if model exists, False otherwise
        """
        import os
        filename = f"{filepath_prefix}{model_name}.joblib"
        return os.path.exists(filename)
    
    def load_single_model(self, model_name, filepath_prefix='models/'):
        """
        Load a single model if it exists.
        
        Args:
            model_name (str): Name of the model
            filepath_prefix (str): Prefix for model file paths
            
        Returns:
            bool: True if model was loaded, False otherwise
        """
        import os
        filename = f"{filepath_prefix}{model_name}.joblib"
        
        if os.path.exists(filename):
            self.models[model_name] = joblib.load(filename)
            print(f"✅ Loaded existing {model_name} model from {filename}")
            return True
        else:
            print(f"📝 No existing {model_name} model found, will train new one")
            return False
    
    def predict_single(self, text, preprocessor):
        """
        Make prediction on a single text sample.
        
        Args:
            text (str): Text to predict
            preprocessor: Fitted preprocessor object
            
        Returns:
            dict: Prediction results
        """
        if not self.models and self.ensemble is None:
            raise ValueError("No models available for prediction.")
        
        # Preprocess text
        cleaned_text = preprocessor.clean_text(text)
        text_tfidf = preprocessor.vectorizer.transform([cleaned_text])
        
        predictions = {}
        
        # Individual model predictions
        for name, model in self.models.items():
            pred = model.predict(text_tfidf)[0]
            
            # Handle LinearSVC which doesn't have predict_proba
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(text_tfidf)[0]
            else:
                # For LinearSVC, use decision_function and convert to probabilities using expit
                decision_score = model.decision_function(text_tfidf)[0]
                prob_positive = expit(decision_score)
                prob_negative = 1 - prob_positive
                proba = [prob_negative, prob_positive]
            
            predictions[name] = {
                'prediction': pred,
                'probability': max(proba),
                'confidence': 'High' if max(proba) > 0.8 else 'Medium' if max(proba) > 0.6 else 'Low'
            }
        
        # Ensemble prediction
        if self.ensemble is not None:
            ensemble_pred = self.ensemble.predict(text_tfidf)[0]
            ensemble_proba = self.ensemble.predict_proba(text_tfidf)[0]
            predictions['ensemble'] = {
                'prediction': ensemble_pred,
                'probability': max(ensemble_proba),
                'confidence': 'High' if max(ensemble_proba) > 0.8 else 'Medium' if max(ensemble_proba) > 0.6 else 'Low'
            }
        
        return predictions
    
    def test_ensemble_functionality(self, X_test, y_test):
        """
        Test that the ensemble works correctly with all models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Test results
        """
        print("🧪 Testing ensemble functionality...")
        
        try:
            # Test ensemble prediction
            ensemble_pred = self.ensemble.predict(X_test[:5])  # Test on first 5 samples
            ensemble_proba = self.ensemble.predict_proba(X_test[:5])
            
            print("✅ Ensemble prediction successful")
            print(f"   Predictions shape: {ensemble_pred.shape}")
            print(f"   Probabilities shape: {ensemble_proba.shape}")
            
            # Test individual model predictions
            for name, model in self.models.items():
                pred = model.predict(X_test[:5])
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test[:5])
                else:
                    # Handle LinearSVC
                    decision_scores = model.decision_function(X_test[:5])
                    prob_positive = expit(decision_scores)
                    prob_negative = 1 - prob_positive
                    proba = np.column_stack([prob_negative, prob_positive])
                
                print(f"✅ {name} prediction successful")
                print(f"   Predictions shape: {pred.shape}")
                print(f"   Probabilities shape: {proba.shape}")
            
            print("🎉 All ensemble tests passed!")
            return True
            
        except Exception as e:
            print(f"❌ Ensemble test failed: {e}")
            return False
