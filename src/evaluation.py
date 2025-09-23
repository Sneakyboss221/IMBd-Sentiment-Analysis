"""
Evaluation and visualization module for IMDb sentiment analysis.
Includes comprehensive metrics, visualizations, and interpretability analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('default')
sns.set_palette("husl")

class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization.
    """
    
    def __init__(self, class_names=['Negative', 'Positive']):
        """
        Initialize the evaluator.
        
        Args:
            class_names (list): Names of the classes
        """
        self.class_names = class_names
        self.results = {}
        
    def evaluate_model(self, y_true, y_pred, y_pred_proba=None, model_name='Model'):
        """
        Evaluate a single model and return comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation results
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # ROC-AUC if probabilities available
        roc_auc = None
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        self.results[model_name] = results
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name='Model', save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name (str): Name of the model
            save_path (str): Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name='Model', save_path=None):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name (str): Name of the model
            save_path (str): Path to save the plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name='Model', save_path=None):
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name (str): Name of the model
            save_path (str): Path to save the plot
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = np.trapz(precision, recall)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names, importance_scores, model_name='Model', 
                               top_n=20, save_path=None):
        """
        Plot feature importance.
        
        Args:
            feature_names (list): List of feature names
            importance_scores (array): Importance scores
            model_name (str): Name of the model
            top_n (int): Number of top features to show
            save_path (str): Path to save the plot
        """
        # Get top features
        top_indices = np.argsort(importance_scores)[-top_n:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_scores = [importance_scores[i] for i in top_indices]
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(top_features)), top_scores)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Features - {model_name}')
        plt.gca().invert_yaxis()
        
        # Color bars by importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_model_comparison_plot(self, results_dict, metric='f1_score', save_path=None):
        """
        Create comparison plot for multiple models.
        
        Args:
            results_dict (dict): Dictionary of model results
            metric (str): Metric to compare
            save_path (str): Path to save the plot
        """
        models = list(results_dict.keys())
        scores = [results_dict[model][metric] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, scores)
        plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
        plt.xlabel('Models')
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, results_dict, save_path=None):
        """
        Create interactive dashboard using Plotly.
        
        Args:
            results_dict (dict): Dictionary of model results
            save_path (str): Path to save the HTML file
        """
        # Prepare data
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'F1-Score', 'Precision', 'Recall'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add traces
        for i, metric in enumerate(['accuracy', 'f1_score', 'precision', 'recall']):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            scores = [results_dict[model][metric] for model in models]
            
            fig.add_trace(
                go.Bar(x=models, y=scores, name=metric.replace('_', ' ').title()),
                row=row, col=col
            )
        
        # Update layout
        fig.update_layout(
            title_text="Model Performance Dashboard",
            showlegend=False,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def analyze_misclassifications(self, X_test, y_true, y_pred, model_name='Model', 
                               preprocessor=None, top_n=10):
        """
        Analyze misclassified samples.
        
        Args:
            X_test: Test features
            y_true: True labels
            y_pred: Predicted labels
            model_name (str): Name of the model
            preprocessor: Preprocessor object
            top_n (int): Number of misclassifications to show
        """
        # Find misclassified samples
        misclassified = y_true != y_pred
        misclassified_indices = np.where(misclassified)[0]
        
        print(f"\n🔍 Misclassification Analysis - {model_name}")
        print("=" * 50)
        print(f"Total misclassifications: {misclassified.sum()}")
        print(f"Misclassification rate: {misclassified.mean():.3f}")
        
        if len(misclassified_indices) == 0:
            print("No misclassifications found!")
            return
        
        # Show top misclassifications
        print(f"\nTop {min(top_n, len(misclassified_indices))} misclassifications:")
        print("-" * 50)
        
        for i, idx in enumerate(misclassified_indices[:top_n]):
            true_label = self.class_names[y_true[idx]]
            pred_label = self.class_names[y_pred[idx]]
            
            if preprocessor is not None and hasattr(preprocessor, 'vectorizer'):
                # Get original text if available
                print(f"{i+1}. True: {true_label}, Predicted: {pred_label}")
            else:
                print(f"{i+1}. True: {true_label}, Predicted: {pred_label}")
    
    def generate_report(self, results_dict, save_path=None):
        """
        Generate comprehensive evaluation report.
        
        Args:
            results_dict (dict): Dictionary of model results
            save_path (str): Path to save the report
        """
        report = []
        report.append("# Model Evaluation Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary table
        report.append("## Model Performance Summary")
        report.append("")
        
        # Create comparison table
        comparison_data = []
        for model_name, results in results_dict.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{results['accuracy']:.3f}",
                'Precision': f"{results['precision']:.3f}",
                'Recall': f"{results['recall']:.3f}",
                'F1-Score': f"{results['f1_score']:.3f}",
                'ROC-AUC': f"{results['roc_auc']:.3f}" if results['roc_auc'] else 'N/A'
            })
        
        # Convert to markdown table
        df = pd.DataFrame(comparison_data)
        report.append(df.to_string(index=False))
        report.append("")
        
        # Best model
        best_model = max(results_dict.keys(), 
                        key=lambda x: results_dict[x]['f1_score'])
        report.append(f"## Best Model: {best_model.replace('_', ' ').title()}")
        report.append(f"F1-Score: {results_dict[best_model]['f1_score']:.3f}")
        report.append("")
        
        # Detailed analysis for each model
        for model_name, results in results_dict.items():
            report.append(f"## {model_name.replace('_', ' ').title()}")
            report.append("")
            report.append(f"**Accuracy:** {results['accuracy']:.3f}")
            report.append(f"**Precision:** {results['precision']:.3f}")
            report.append(f"**Recall:** {results['recall']:.3f}")
            report.append(f"**F1-Score:** {results['f1_score']:.3f}")
            if results['roc_auc']:
                report.append(f"**ROC-AUC:** {results['roc_auc']:.3f}")
            report.append("")
        
        # Join report
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")
        
        return report_text
    
    def plot_learning_curves(self, model, X_train, y_train, cv=5, save_path=None):
        """
        Plot learning curves for a model.
        
        Args:
            model: Model to evaluate
            X_train: Training features
            y_train: Training labels
            cv (int): Cross-validation folds
            save_path (str): Path to save the plot
        """
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self, results_dict, save_path=None):
        """
        Plot correlation heatmap of metrics across models.
        
        Args:
            results_dict (dict): Dictionary of model results
            save_path (str): Path to save the plot
        """
        # Prepare data
        data = []
        for model_name, results in results_dict.items():
            data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc'] if results['roc_auc'] else 0
            })
        
        df = pd.DataFrame(data)
        correlation_matrix = df.set_index('Model').corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        plt.title('Metric Correlation Heatmap')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
