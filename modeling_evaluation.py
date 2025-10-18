import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.models.model import (
    LogisticRegressionGD, KNNClassifier, DecisionTreeClassifierManual,
    RandomForestManual, GradientBoostingClassifierManual,
    accuracy_score, precision_score_manual, recall_score_manual, 
    f1_score_manual, roc_auc_score_manual, stratified_train_test_split
)


class ModelingEvaluationTab:
    def __init__(self):
        self.name = "Modeling & Evaluation"
        self.results = {}
    
    def load_data(self):
        """Load featured dataset for modeling"""
        FEATURED_PATH = os.path.join("data", "cart_abandonment_featured.csv")
        try:
            df = pd.read_csv(FEATURED_PATH)
            return df
        except FileNotFoundError:
            st.error("📊 Featured dataset not found. Please run Feature Engineering first.")
            return None

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score_manual(y_test, y_pred),
                'recall': recall_score_manual(y_test, y_pred),
                'f1_score': f1_score_manual(y_test, y_pred),
                'roc_auc': roc_auc_score_manual(y_test, y_proba)
            }
            
            # Store results
            self.results[model_name] = {
                'model': model,
                'metrics': metrics,
                'y_true': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            
            return metrics
            
        except Exception as e:
            st.error(f"❌ Error evaluating {model_name}: {str(e)}")
            return None

    def plot_roc_curves(self):
        """Plot ROC curves for all trained models"""
        if not self.results:
            return
        
        fig = go.Figure()
        
        for model_name, result in self.results.items():
            y_true = result['y_true']
            y_proba = result['y_proba']
            
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_score = result['metrics']['roc_auc']
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'{model_name} (AUC = {auc_score:.3f})',
                line=dict(width=2)
            ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            line=dict(dash='dash', color='gray'),
            name='Random Classifier',
            showlegend=False
        ))
        
        fig.update_layout(
            title='📈 ROC Curves - Model Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=700,
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig)

    def plot_pr_curves(self):
        """Plot Precision-Recall curves for all trained models"""
        if not self.results:
            return
        
        fig = go.Figure()
        
        for model_name, result in self.results.items():
            y_true = result['y_true']
            y_proba = result['y_proba']
            
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            f1_score = result['metrics']['f1_score']
            
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                name=f'{model_name} (F1 = {f1_score:.3f})',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='📊 Precision-Recall Curves',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=700,
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig)

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        if not self.results:
            return
        
        n_models = len(self.results)
        cols = min(2, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=list(self.results.keys()),
            horizontal_spacing=0.15,
            vertical_spacing=0.2
        )
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            
            cm = confusion_matrix(result['y_true'], result['y_pred'])
            
            heatmap = go.Heatmap(
                z=cm,
                x=['Not Abandoned', 'Abandoned'],
                y=['Not Abandoned', 'Abandoned'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                colorscale='Blues',
                showscale=False
            )
            
            fig.add_trace(heatmap, row=row, col=col)
            
            accuracy = result['metrics']['accuracy']
            fig.layout.annotations[idx].update(text=f'{model_name}<br>Acc: {accuracy:.3f}')
        
        fig.update_layout(
            title_text="🎯 Confusion Matrices",
            height=400 * rows,
            showlegend=False
        )
        
        st.plotly_chart(fig)

    def display_metrics_comparison(self):
        """Display comparison table of all model metrics"""
        if not self.results:
            return
        
        metrics_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            metrics_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        st.subheader("📋 Model Performance Comparison")
        st.dataframe(metrics_df.style.highlight_max(axis=0), width='stretch')
        
        st.subheader("🏆 Best Performing Models by Metric")
        best_models = {}
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
            best_idx = metrics_df[metric].astype(float).idxmax()
            best_models[metric] = metrics_df.loc[best_idx, 'Model']
        
        best_df = pd.DataFrame(list(best_models.items()), columns=['Metric', 'Best Model'])
        st.dataframe(best_df, width='stretch')

    def calculate_feature_importance(self, model, feature_names, model_name):
        """Calculate feature importance for tree-based models"""
        try:
            if hasattr(model, 'trees') and model.trees:
                feature_importance = np.zeros(len(feature_names))
                total_splits = 0
                
                for tree in model.trees:
                    feature_usage = self._count_feature_usage(tree.tree, len(feature_names))
                    feature_importance += feature_usage
                    total_splits += np.sum(feature_usage > 0)
                
                if total_splits > 0:
                    feature_importance = feature_importance / total_splits
                return feature_importance
            
            elif hasattr(model, 'tree'):
                feature_importance = self._count_feature_usage(model.tree, len(feature_names))
                total_splits = np.sum(feature_importance > 0)
                if total_splits > 0:
                    feature_importance = feature_importance / total_splits
                return feature_importance
            
            else:
                return None
                
        except Exception as e:
            st.warning(f"Could not calculate feature importance for {model_name}: {e}")
            return None

    def _count_feature_usage(self, node, n_features):
        """Recursively count feature usage in decision tree nodes"""
        feature_usage = np.zeros(n_features)
        
        if not node.get("is_leaf", True):
            feature_idx = node.get("feature")
            if feature_idx is not None and feature_idx < n_features:
                feature_usage[feature_idx] += 1
            
            if "left" in node:
                feature_usage += self._count_feature_usage(node["left"], n_features)
            if "right" in node:
                feature_usage += self._count_feature_usage(node["right"], n_features)
        
        return feature_usage

    def plot_feature_importance(self, model, feature_names, model_name):
        """Plot feature importance for tree-based models"""
        try:
            importance_scores = self.calculate_feature_importance(model, feature_names, model_name)
            
            if importance_scores is None or np.sum(importance_scores) == 0:
                st.info(f"📊 Feature importance not available for {model_name}")
                return
            
            feat_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=True)
            
            non_zero_features = feat_imp_df[feat_imp_df['importance'] > 0]
            if len(non_zero_features) == 0:
                st.info(f"📊 No feature importance data available for {model_name}")
                return
            
            top_features = non_zero_features.tail(15)  # Top 15 features
            
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title=f'🔍 {model_name} - Feature Importance (Top 15)',
                color='importance',
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig)
            
        except Exception as e:
            st.info(f"📊 Feature importance analysis not available for {model_name}")

    def run(self):
        st.header("🤖 Modeling & Evaluation")
        
        df = self.load_data()
        if df is None:
            st.info("💡 Please run the Feature Engineering tab first to generate the dataset for modeling.")
            return
        
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)  # Excluding target
        with col3:
            abandonment_rate = df['abandoned'].mean()
            st.metric("Abandonment Rate", f"{abandonment_rate:.2%}")
        with col4:
            positive_count = df['abandoned'].sum()
            positive_ratio = df['abandoned'].mean()
            st.metric("Positive Class", f"{positive_count} ({positive_ratio:.1%})")
        
        st.subheader("⚙️ Data Preparation")
        
        exclude_cols = ['abandoned', 'session_id', 'user_id']
        selected_features = [col for col in df.columns if col not in exclude_cols]
        
        st.info(f"🔧 Using all {len(selected_features)} features for modeling")
        
        with st.expander("View Features Used for Modeling"):
            st.write("Selected features:", selected_features)
        
        X = df[selected_features]
        y = df['abandoned'].values
        feature_names = selected_features
        
        st.write("**Data Split Configuration**")
        test_size = 0.2  
        random_state = 42 
        
        st.info(f"📊 Using fixed split: {test_size*100}% test size, random_state={random_state}")
        
        train_idx, test_idx = stratified_train_test_split(X.values, y, test_size=test_size, random_state=random_state)
        X_train, X_test = X.values[train_idx], X.values[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        st.success(f"✅ Data split: {len(X_train)} training samples, {len(X_test)} test samples")
        
        st.subheader("🎯 Model Selection & Training")
        
        st.write("**Choose models to train:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Models**")
            train_lr = st.checkbox("Logistic Regression", value=True)
            train_knn = st.checkbox("K-Nearest Neighbors", value=True)
            train_dt = st.checkbox("Decision Tree", value=True)
            
        with col2:
            st.write("**Ensemble Models**")
            train_rf = st.checkbox("Random Forest", value=True)
            train_gb = st.checkbox("Gradient Boosting", value=True)
        
        save_models = st.checkbox("💾 Save trained models to disk", value=True)
        if save_models:
            st.info("Models will be saved as .pkl files in the 'saved_models' directory for later use")
        
        if st.button("🚀 Train Selected Models", type="primary", width='stretch'):
            self.results = {} 
            with st.spinner("Training models... This may take a few moments."):
                progress_bar = st.progress(0)
                models_to_train = []
                
                if train_lr: models_to_train.append(("Logistic Regression", LogisticRegressionGD))
                if train_knn: models_to_train.append(("K-Nearest Neighbors", KNNClassifier))
                if train_dt: models_to_train.append(("Decision Tree", DecisionTreeClassifierManual))
                if train_rf: models_to_train.append(("Random Forest", RandomForestManual))
                if train_gb: models_to_train.append(("Gradient Boosting", GradientBoostingClassifierManual))
                
                for i, (model_name, model_class) in enumerate(models_to_train):
                    progress_bar.progress((i) / len(models_to_train))
                    
                    with st.spinner(f"Training {model_name}..."):
                        try:
                            if model_name == "Logistic Regression":
                                model = model_class(lr=0.1, n_iter=1000, l2=1e-4)
                            elif model_name == "K-Nearest Neighbors":
                                model = model_class(k=5)
                            elif model_name == "Decision Tree":
                                model = model_class(max_depth=6, min_samples_split=10)
                            elif model_name == "Random Forest":
                                model = model_class(n_estimators=20, max_depth=6, min_samples_split=10)
                            elif model_name == "Gradient Boosting":
                                model = model_class(n_estimators=50, learning_rate=0.1, max_depth=3)
                            else:
                                model = model_class()
                            
                            if model_name == "Gradient Boosting":
                                model.fit_with_baseline(X_train, y_train)
                            else:
                                model.fit(X_train, y_train)
                            
                            metrics = self.evaluate_model(model, X_test, y_test, model_name)
                            
                            if metrics:
                                st.success(f"✅ {model_name} trained successfully!")
                                
                            if save_models:
                                model_dir = "saved_models"
                                os.makedirs(model_dir, exist_ok=True)
                                model_path = os.path.join(model_dir, f"{model_name.replace(' ', '_').lower()}.pkl")
                                with open(model_path, "wb") as f:
                                    pickle.dump({
                                        'model': model,
                                        'feature_names': feature_names,
                                        'metrics': metrics,
                                        'model_name': model_name
                                    }, f)
                                st.info(f"💾 Saved {model_name} to {model_path}")
                        
                        except Exception as e:
                            st.error(f"❌ Failed to train {model_name}: {str(e)}")
                
                progress_bar.progress(1.0)
                st.success("🎉 All models trained successfully!")
        
        if self.results:
            st.subheader("📊 Evaluation Results")
            
            self.display_metrics_comparison()
            
            eval_tabs = st.tabs(["📈 ROC Curves", "📊 PR Curves", "🎯 Confusion Matrices", "🔍 Feature Importance"])
            
            with eval_tabs[0]:
                self.plot_roc_curves()
                
            with eval_tabs[1]:
                self.plot_pr_curves()
                
            with eval_tabs[2]:
                self.plot_confusion_matrices()
                
            with eval_tabs[3]:
                st.subheader("Feature Importance Analysis")
                tree_models = [name for name in self.results.keys() 
                             if any(x in name.lower() for x in ['tree', 'forest', 'boost'])]
                
                if tree_models:
                    for model_name in tree_models:
                        model_result = self.results[model_name]
                        self.plot_feature_importance(model_result['model'], feature_names, model_name)
                else:
                    st.info("📊 No tree-based models available for feature importance analysis.")
            
            st.subheader("💾 Model Export")
            if st.button("Download Best Model", width='content'):
                if self.results:
                    best_model_name = max(self.results.keys(), 
                                        key=lambda x: self.results[x]['metrics']['roc_auc'])
                    best_model = self.results[best_model_name]
                    
                    model_data = {
                        'model': best_model['model'],
                        'feature_names': feature_names,
                        'metrics': best_model['metrics'],
                        'model_name': best_model_name
                    }
                    
                    model_bytes = pickle.dumps(model_data)
                    
                    st.download_button(
                        label=f"📥 Download {best_model_name}",
                        data=model_bytes,
                        file_name=f"best_model_{best_model_name.replace(' ', '_').lower()}.pkl",
                        mime="application/octet-stream",
                        width='content'
                    )