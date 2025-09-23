"""
Complete IMDb Sentiment Analysis Pipeline
========================================

This script runs the complete sentiment analysis pipeline including:
- Data loading and preprocessing
- Model training (Logistic Regression, LinearSVM, MultinomialNB)
- Ensemble creation
- Comprehensive evaluation
- Model saving and visualization
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from preprocessing import TextPreprocessor, load_imdb_data, validate_data
from models import ModelTrainer
from evaluation import ModelEvaluator

def load_excel_dataset(file_path):
    """
    Load dataset from Excel file.
    
    Args:
        file_path (str): Path to Excel file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print(f"Loading dataset from: {file_path}")
    
    try:
        # Load Excel file
        df = pd.read_excel(file_path)
        print(f"Dataset loaded successfully: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Please ensure your Excel file is in the correct location.")
        return None
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

def create_sample_dataset():
    """
    Create a comprehensive sample dataset for demonstration.
    """
    print("Creating comprehensive sample dataset...")
    
    # Extended positive reviews
    positive_reviews = [
        "This movie was absolutely fantastic! The acting was superb and the plot was engaging.",
        "Amazing cinematography and great performances by all actors. Highly recommended!",
        "Outstanding film with brilliant storytelling and exceptional acting.",
        "Perfect movie! Everything about it was excellent - acting, direction, cinematography.",
        "Brilliant masterpiece! One of the best films I've ever seen.",
        "Excellent film with outstanding performances and great direction.",
        "Wonderful movie with amazing special effects and great acting.",
        "Fantastic film that kept me engaged from start to finish.",
        "Outstanding performances and brilliant storytelling. Highly recommended!",
        "Amazing movie with great cinematography and excellent acting.",
        "Incredible film! The story was captivating and the characters were well-developed.",
        "Superb movie with outstanding direction and brilliant performances.",
        "Magnificent! This is cinema at its finest.",
        "Exceptional film with wonderful cinematography and great storytelling.",
        "Outstanding movie! The plot was engaging and the acting was top-notch.",
        "Brilliant film with amazing special effects and great direction.",
        "Fantastic movie! One of the best I've seen this year.",
        "Excellent film with outstanding performances and great cinematography.",
        "Wonderful movie with brilliant storytelling and exceptional acting.",
        "Perfect film! Everything about it was outstanding.",
        "Amazing movie with great performances and wonderful direction.",
        "Outstanding film with brilliant cinematography and great acting.",
        "Fantastic movie! The story was engaging and the characters were well-developed.",
        "Excellent film with outstanding performances and great storytelling.",
        "Brilliant movie with amazing direction and wonderful acting.",
        "Superb film with outstanding cinematography and great performances.",
        "Magnificent movie! This is a masterpiece of cinema.",
        "Exceptional film with brilliant storytelling and outstanding acting.",
        "Outstanding movie with great direction and wonderful performances.",
        "Fantastic film with amazing cinematography and brilliant storytelling.",
        "Excellent movie with outstanding acting and great direction.",
        "Wonderful film with brilliant performances and amazing storytelling.",
        "Perfect movie with exceptional direction and outstanding cinematography.",
        "Amazing film with great acting and brilliant storytelling.",
        "Outstanding movie with wonderful performances and excellent direction.",
        "Fantastic film with brilliant cinematography and great acting.",
        "Excellent movie with outstanding storytelling and wonderful performances.",
        "Brilliant film with amazing direction and exceptional acting.",
        "Superb movie with outstanding cinematography and great storytelling.",
        "Magnificent film with brilliant performances and wonderful direction.",
        "Exceptional movie with amazing acting and outstanding cinematography."
    ]
    
    # Extended negative reviews
    negative_reviews = [
        "Terrible movie. Boring plot, bad acting, and poor direction. Would not recommend.",
        "Waste of time. The story made no sense and the characters were poorly developed.",
        "Disappointing. Expected much more from this director. The script was weak.",
        "Awful experience. The movie was confusing and poorly executed.",
        "Complete disaster. Nothing worked in this movie.",
        "Boring film with terrible acting and poor direction.",
        "Waste of money. The movie was confusing and poorly made.",
        "Terrible experience. Bad acting and boring plot.",
        "Disappointing film with poor storytelling and weak characters.",
        "Awful movie. Nothing about it was good.",
        "Horrible film. The plot was nonsensical and the acting was terrible.",
        "Waste of time. Boring and poorly executed.",
        "Terrible movie with bad direction and awful acting.",
        "Disappointing experience. The story was weak and confusing.",
        "Awful film with poor cinematography and terrible performances.",
        "Complete waste of time. Nothing about this movie was good.",
        "Terrible movie with bad storytelling and poor direction.",
        "Disappointing film with awful acting and weak plot.",
        "Horrible experience. The movie was confusing and poorly made.",
        "Waste of money. Terrible direction and bad performances.",
        "Awful movie with poor storytelling and terrible acting.",
        "Disappointing film with bad cinematography and weak characters.",
        "Terrible experience. The plot was nonsensical and poorly executed.",
        "Horrible movie with awful direction and bad performances.",
        "Complete disaster. Nothing worked in this film.",
        "Waste of time. Boring plot and terrible acting.",
        "Awful movie with poor direction and bad storytelling.",
        "Disappointing film with terrible cinematography and weak plot.",
        "Horrible experience. The movie was confusing and poorly made.",
        "Terrible movie with bad performances and awful direction.",
        "Waste of money. Disappointing storytelling and poor acting.",
        "Awful film with terrible direction and bad cinematography.",
        "Complete waste of time. Nothing about this movie was good.",
        "Horrible movie with poor storytelling and awful performances.",
        "Disappointing experience. The film was confusing and badly executed.",
        "Terrible movie with bad direction and poor cinematography.",
        "Awful film with terrible acting and weak plot development.",
        "Waste of time. Boring and poorly made movie.",
        "Horrible experience. Nothing worked in this film.",
        "Disappointing movie with awful direction and bad storytelling.",
        "Terrible film with poor performances and weak character development.",
        "Awful movie with bad cinematography and terrible plot.",
        "Complete disaster. The film was confusing and poorly executed.",
        "Waste of money. Terrible direction and awful acting.",
        "Horrible movie with poor storytelling and bad performances.",
        "Disappointing film with terrible cinematography and weak plot.",
        "Awful experience. Nothing about this movie was good.",
        "Terrible film with bad direction and poor character development.",
        "Waste of time. Boring plot and terrible execution."
    ]
    
    # Create balanced dataset
    reviews = []
    sentiments = []
    
    # Add positive reviews
    for review in positive_reviews:
        reviews.append(review)
        sentiments.append('positive')
    
    # Add negative reviews
    for review in negative_reviews:
        reviews.append(review)
        sentiments.append('negative')
    
    # Create DataFrame
    df = pd.DataFrame({
        'review': reviews,
        'sentiment': sentiments
    })
    
    print(f"Sample dataset created: {df.shape}")
    print(f"Class distribution: {df['sentiment'].value_counts().to_dict()}")
    
    return df

def main():
    """
    Main pipeline execution.
    """
    print("🎬 IMDb Sentiment Analysis - Complete Pipeline")
    print("=" * 60)
    
    # Step 1: Data Preparation
    print("\n📊 Step 1: Data Preparation")
    print("-" * 30)
    
    # Load your IMDb dataset
    dataset_path = "IMDB Dataset.csv"
    
    # Try to load IMDb dataset first, fallback to sample if not found
    df = load_imdb_data(dataset_path)
    if df is None:
        print("Falling back to sample dataset...")
        df = create_sample_dataset()
    
    # Validate data
    if not validate_data(df):
        print("❌ Data validation failed!")
        return
    
    # Step 2: Preprocessing
    print("\n🔧 Step 2: Text Preprocessing")
    print("-" * 30)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        max_features=5000,
        min_df=2,
        ngram_range=(1, 2)
    )
    
    # Prepare data
    X_train, X_test, y_train, y_test, fitted_preprocessor = preprocessor.prepare_data(
        df, test_size=0.2, random_state=42
    )
    
    print(f"✅ Preprocessing completed!")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {X_train.shape[1]}")
    
    # Step 3: Model Training
    print("\n🤖 Step 3: Model Training")
    print("-" * 30)
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Train Logistic Regression (auto-saves if new, loads if exists)
    print("\nTraining Logistic Regression...")
    lr_results = trainer.train_logistic_regression(
        X_train, y_train, X_test, y_test,
        param_grid={'C': [0.1, 1, 10], 'max_iter': [1000]},
        filepath_prefix='models/'
    )
    
    # Train LinearSVM (auto-saves if new, loads if exists)
    print("\nTraining LinearSVM...")
    svm_results = trainer.train_svm(
        X_train, y_train, X_test, y_test,
        param_grid={'C': [0.1, 1, 10], 'loss': ['squared_hinge'], 'dual': [False]},
        filepath_prefix='models/'
    )
    
    # Train Multinomial Naive Bayes (auto-saves if new, loads if exists)
    print("\nTraining Multinomial Naive Bayes...")
    nb_results = trainer.train_multinomial_nb(
        X_train, y_train, X_test, y_test,
        param_grid={
            'alpha': [0.1, 1.0, 10.0]
        },
        filepath_prefix='models/'
    )
    
    # Create ensemble (auto-saves if new, loads if exists)
    print("\nCreating ensemble...")
    ensemble_results = trainer.create_ensemble(
        X_train, y_train, X_test, y_test,
        filepath_prefix='models/'
    )
    
    # Test ensemble functionality
    print("\n🧪 Testing ensemble functionality...")
    ensemble_test_passed = trainer.test_ensemble_functionality(X_test, y_test)
    if ensemble_test_passed:
        print("✅ Ensemble functionality test passed!")
    else:
        print("❌ Ensemble functionality test failed!")
    
    # Step 4: Model Comparison
    print("\n📈 Step 4: Model Comparison")
    print("-" * 30)
    
    comparison_df = trainer.compare_models()
    
    # Step 5: Evaluation and Visualization
    print("\n📊 Step 5: Evaluation and Visualization")
    print("-" * 30)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate all models
    evaluation_results = {}
    for model_name, results in trainer.results.items():
        evaluation_results[model_name] = evaluator.evaluate_model(
            y_test, results['predictions'], results['probabilities'], model_name
        )
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Model comparison plot
    evaluator.create_model_comparison_plot(evaluation_results, metric='f1_score')
    
    # Confusion matrices
    for model_name, results in trainer.results.items():
        evaluator.plot_confusion_matrix(
            y_test, results['predictions'], model_name
        )
    
    # ROC curves
    for model_name, results in trainer.results.items():
        if results['probabilities'] is not None:
            evaluator.plot_roc_curve(
                y_test, results['probabilities'], model_name
            )
    
    # Feature importance (for Logistic Regression)
    if 'logistic_regression' in trainer.models:
        feature_names = fitted_preprocessor.get_feature_names()
        lr_model = trainer.models['logistic_regression']
        
        if hasattr(lr_model, 'coef_'):
            importance_scores = np.abs(lr_model.coef_[0])
            evaluator.plot_feature_importance(
                feature_names, importance_scores, 'Logistic Regression'
            )
    
    # Step 6: Model Persistence Summary
    print("\n💾 Step 6: Model Persistence Summary")
    print("-" * 30)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # All models are automatically saved during training
    print("✅ All models automatically saved during training")
    print("📁 Models saved to: models/ directory")
    
    # Save evaluation report
    report = evaluator.generate_report(evaluation_results, 'results/evaluation_report.md')
    
    # Step 7: Interactive Prediction Demo
    print("\n🔮 Step 7: Interactive Prediction Demo")
    print("-" * 30)
    
    # Sample new reviews for prediction
    new_reviews = [
        "This movie was absolutely amazing! Best film I've seen this year!",
        "Terrible movie. Boring and poorly made. Would not recommend.",
        "Great acting and wonderful cinematography. Highly recommended!",
        "Waste of time. Confusing plot and bad direction.",
        "Outstanding performances and brilliant storytelling. A masterpiece!",
        "Awful film with terrible acting and poor direction.",
        "Fantastic movie with amazing special effects and great acting.",
        "Disappointing experience. The movie was confusing and poorly executed."
    ]
    
    print("Analyzing new reviews:")
    print("-" * 40)
    
    for i, review in enumerate(new_reviews, 1):
        predictions = trainer.predict_single(review, fitted_preprocessor)
        
        # Get ensemble prediction (preferred)
        if 'ensemble' in predictions:
            pred = predictions['ensemble']['prediction']
            prob = predictions['ensemble']['probability']
            confidence = predictions['ensemble']['confidence']
        else:
            # Fallback to first available model
            pred = list(predictions.values())[0]['prediction']
            prob = list(predictions.values())[0]['probability']
            confidence = list(predictions.values())[0]['confidence']
        
        sentiment = "Positive" if pred == 1 else "Negative"
        print(f"{i}. {sentiment:8s} ({confidence:5s} confidence: {prob:.3f}) - {review}")
    
    # Final Summary
    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\n📊 Final Results Summary:")
    print("-" * 30)
    for model_name, results in evaluation_results.items():
        print(f"{model_name.replace('_', ' ').title():20s} - F1: {results['f1_score']:.3f}, Accuracy: {results['accuracy']:.3f}")
    
    # Best model
    best_model = max(evaluation_results.keys(), 
                    key=lambda x: evaluation_results[x]['f1_score'])
    print(f"\n🏆 Best Model: {best_model.replace('_', ' ').title()}")
    print(f"   F1-Score: {evaluation_results[best_model]['f1_score']:.3f}")
    print(f"   Accuracy: {evaluation_results[best_model]['accuracy']:.3f}")
    
    print("\n📁 Files Created:")
    print("- models/ - Saved trained models")
    print("- results/ - Evaluation results and reports")
    print("- Visualizations displayed above")
    
    print("\n🚀 Next Steps:")
    print("1. Review the evaluation report in results/evaluation_report.md")
    print("2. Use the saved models for predictions on new data")
    print("3. Experiment with different parameters and models")
    print("4. Try the Jupyter notebooks for detailed analysis")
    
    print("\n✨ Thank you for using the IMDb Sentiment Analysis Pipeline!")

if __name__ == "__main__":
    main()
