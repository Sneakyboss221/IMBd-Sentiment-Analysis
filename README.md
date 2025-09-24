# 🎬 IMDb Sentiment Analysis

A comprehensive machine learning project for sentiment analysis of IMDb movie reviews using multiple classification algorithms and ensemble methods. This project demonstrates production ready machine learning practices including data preprocessing, model training, hyperparameter tuning, evaluation, and interpretability.

## 📊 Dataset

The project uses the **IMDb Movie Reviews Dataset** containing:
- **50,000 movie reviews** (25,000 positive, 25,000 negative)
- **Text data**: Raw movie review text with HTML tags and special characters
- **Labels**: Binary sentiment classification (positive/negative)
- **Balanced dataset**: Equal distribution of positive and negative reviews

## 🔧 Preprocessing Pipeline

### Text Cleaning
- **HTML tag removal** using BeautifulSoup
- **URL and special character cleaning**
- **Contraction expansion** (e.g., "don't" → "do not")
- **Text normalization** and lowercase conversion
- **Duplicate removal** (422 duplicates removed)

### Feature Engineering
- **Stopwords removal** using NLTK English stopwords
- **TF-IDF vectorization** with:
  - **N-gram range**: (1, 2) - unigrams and bigrams
  - **Maximum features**: 5,000
  - **Minimum document frequency**: 2
- **Train-test split**: 80/20 with stratification

## 🤖 Machine Learning Models

### 1. Logistic Regression
- **Algorithm**: Linear classifier with L2 regularization
- **Hyperparameters**: C ∈ [0.01, 0.1, 1, 10, 50], penalty=['l2','elasticnet'], solver='saga', class_weight=[None, 'balanced']
- **Best parameters**: C=1, penalty='l2', solver='saga', class_weight=None
- **Strengths**: Interpretable, fast training, good baseline

### 2. LinearSVM (Support Vector Machine)
- **Algorithm**: Linear support vector classifier
- **Hyperparameters**: C ∈ [0.1, 1, 10], loss='squared_hinge', dual=False
- **Best parameters**: C=0.1, dual=False, loss='squared_hinge'
- **Strengths**: High performance, robust to outliers

### 3. Multinomial Naive Bayes
- **Algorithm**: Probabilistic classifier for text classification
- **Hyperparameters**: alpha ∈ [0.1, 1.0, 10.0]
- **Best parameters**: alpha=1.0
- **Strengths**: Fast training, good for text data, probabilistic outputs

### 4. Ensemble (Soft Voting)
- **Method**: Soft voting classifier combining Logistic Regression and LinearSVM
- **Weights**: Equal weighting (50/50) with probability averaging
- **Probability handling**: Proper LinearSVC probability calibration
- **Purpose**: Leverage strengths of the two strongest models
- **Benefits**: Robust, stable performance without Naive Bayes

## 📈 Evaluation Metrics

The project uses comprehensive evaluation metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve

## 📊 Results

### Model Performance Comparison

| Model               | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
|-------------------- |---------|----------|-----------|-------|---------|
| **Logistic Regression** | **0.885** | **0.887** | **0.878** | **0.895** | **0.954** |
| LinearSVM            | 0.885   | 0.887    | 0.877     | 0.898   | 0.954   |
| Ensemble             | 0.885   | 0.886    | 0.877     | 0.895   | 0.954   |
| Multinomial NB       | 0.855   | 0.857    | 0.845     | 0.870   | 0.929   |

### Key Insights
- **Best Model**: Logistic Regression with 88.7% F1-Score
- **Close Performance**: Logistic Regression and LinearSVM are nearly identical
- **Ensemble Benefits**: Slightly improved recall (90.0%) compared to individual models
- **Consistent Performance**: All models achieve >85% accuracy

## 📊 Visualizations

All plots generated during training and evaluation are saved in the `results/plots/` folder.  

You can find:  
- 🟢 Confusion matrices for Logistic Regression, SVM, and the Ensemble  
- 📈 ROC curves for these three models  
- 📊 F1-score comparison plot  
- ⭐ Top 20 important features for Logistic Regression  

**View all plots:** `results/plots/`



### Confusion Matrix - Ensemble Model
![Ensemble Confusion Matrix](results/plots/Confusion%20Matrix%20-%20Ensemble.png)

### ROC Curve - Ensemble Model
![Ensemble ROC Curve](results/plots/ROC%20Curve%20-%20Ensemble.png)

### Model Comparison (F1-Score)
![Model Comparison F1-score](results/plots/Model%20Comparision%20-%20F1%20Score.png)

### Top 20 Features - Logistic Regression
![Top 20 Features Logistic Regression](results/plots/Top%2020%20Features%20-%20Logistic%20Regression.png)

## 🏆 Conclusion

### Best Model: LinearSVM
- **F1-Score**: 88.7%
- **Accuracy**: 88.5%
- **ROC-AUC**: 95.4%
- **Strengths**: Strong generalization with robust margin-based decision boundary

### Ensemble Purpose
- **Configuration**: LR + LinearSVM, soft voting with equal (50/50) weights
- **Performance**: Comparable to the best individual model (SVM) with ROC-AUC 0.954
- **Robustness**: Combines strengths of LR and SVM; NB removed from ensemble

### Overall Insights
- **Text preprocessing** significantly impacts model performance
- **TF-IDF with bigrams** captures important contextual information
- **Linear models** perform exceptionally well on this dataset
- **Ensemble methods** provide marginal but consistent improvements

## 🚀 How to Run

### Prerequisites
```bash
# Install required packages
pip install -r requirements.txt
```

### Quick Start
```bash
# Run the complete pipeline
python main.py
```

### Project Structure
```
imdb_sentiment_analysis/
├── src/                          # Core implementation
│   ├── preprocessing.py         # Text preprocessing
│   ├── models.py               # Model training
│   └── evaluation.py           # Evaluation metrics
├── models/                      # Saved trained models
│   ├── logistic_regression.joblib
│   ├── svm.joblib
│   ├── multinomial_nb.joblib
│   └── ensemble.joblib
├── results/                     # Evaluation results
│   ├── evaluation_report.md
│   └── plots/                   # Generated visualizations
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation_ensemble.ipynb
├── main.py                     # Complete pipeline
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔧 Technical Implementation

### Automatic Model Persistence
- **Training**: All models automatically saved during training
- **Loading**: Models can be loaded for predictions without retraining
- **Format**: Joblib serialization for efficient storage and loading

### Ensemble Functionality
- **Soft Voting**: Combines probability outputs from Logistic Regression and LinearSVM only
- **Weights**: Equal weighting (50/50) with probability averaging
- **Probability Calibration**: Proper handling of LinearSVC probabilities
- **Behavior**: LR/SVM are loaded if saved; ensemble is rebuilt every run from LR+SVM
- **Comparison**: MultinomialNB is evaluated for charts only and excluded from the ensemble

### Code Organization
- **Modular Design**: Clean separation of preprocessing, training, and evaluation
- **Error Handling**: Comprehensive validation and error management
- **Documentation**: Extensive docstrings and comments
- **Reproducibility**: Fixed random seeds for consistent results

## 📝 Notes

- **Model Saving**: All models are automatically saved during training to the `models/` directory
- **Ensemble Functionality**: Comprehensive testing ensures all models work correctly together
- **Visualization**: All plots are automatically generated and saved to `results/plots/`
- **Evaluation Report**: Detailed performance metrics saved to `results/evaluation_report.md`

## 🎯 Key Features

- ✅ **Production-Ready**: Modular design with comprehensive error handling
- ✅ **Comprehensive Evaluation**: Multiple metrics and visualizations
- ✅ **Automatic Persistence**: Models saved and loaded automatically
- ✅ **Ensemble Methods**: Advanced ensemble with proper probability handling
- ✅ **Interpretability**: Feature importance analysis and model insights
- ✅ **Visualization**: Interactive plots and confusion matrices
- ✅ **Documentation**: Extensive documentation and code comments

## 📁 Recommended Usage

While this repository includes Jupyter notebooks for demonstration, **it is recommended to refer directly to the `src/` folder** for running and understanding the full pipeline.  

The `src/` folder contains all the core modules:  
- `preprocessing.py` – data loading, cleaning, and preprocessing  
- `models.py` – model training and ensemble creation  
- `evaluation.py` – evaluation metrics, visualization, and reports  

> ✅ **Tip:** Notebooks are mainly for exploration and visualization. For running the complete pipeline efficiently, use the scripts in `src/`.
