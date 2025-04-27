import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import shap
import os

# Create result1 directory
os.makedirs('result1', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def create_interaction_features(X):
    """Create interaction features"""
    interaction_features = pd.DataFrame()
    
    # Imaging feature interactions
    imaging_features = ['volume_m', 'surface_mr', 'max_diamt']
    for i in range(len(imaging_features)):
        for j in range(i+1, len(imaging_features)):
            interaction_features[f'{imaging_features[i]}_{imaging_features[j]}'] = X[imaging_features[i]] * X[imaging_features[j]]
    
    # Clinical feature interactions
    clinical_features = ['age', 'gender_male']
    for i in range(len(clinical_features)):
        for j in range(len(imaging_features)):
            interaction_features[f'{clinical_features[i]}_{imaging_features[j]}'] = X[clinical_features[i]] * X[imaging_features[j]]
    
    return interaction_features

def evaluate_model(y_true, y_pred, y_prob=None):
    """Evaluate model performance"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    return metrics

def load_and_preprocess_data():
    """Load and preprocess data"""
    try:
        # Load data
        tumor_features = pd.read_csv('result1/p1_3_processed_tumor_features.csv')
        clinical_data = pd.read_csv('result1/p1_1_clinical1_processed.csv')
        
        # Ensure PatientID is string type
        tumor_features['PatientID'] = tumor_features['PatientID'].astype(str)
        clinical_data['PatientID'] = clinical_data['PatientID'].astype(str)
        
        # Merge
        merged_data = pd.merge(tumor_features, clinical_data, on='PatientID', how='inner')
        logging.info(f"Merged data shape: {merged_data.shape}")
        
        # Check target variable distribution
        logging.info("Target variable distribution:")
        logging.info(merged_data['deadstatus.event'].value_counts(normalize=True))
        
        # Fill missing values
        numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
        merged_data[numeric_cols] = merged_data[numeric_cols].fillna(merged_data[numeric_cols].median())
        
        # Define features to exclude
        excluded_features = [
            'PatientID', 'folder',  # Identifiers
            'deadstatus.event', 'deadstatus.event_0', 'deadstatus.event_1',  # Target variables
            'Survival.time', 'Survival.time_original',  # Survival time
            'age_original'  # Duplicate features
        ]
        
        # Check OneHot encoded columns
        onehot_cols = [col for col in merged_data.columns if '_' in col]
        logging.info("OneHot encoded columns:")
        logging.info(onehot_cols)
        
        # Check numeric columns
        numeric_cols = ['age']  # Only keep age as numeric feature
        
        # Final selected features
        selected_features = [col for col in numeric_cols + onehot_cols if col not in excluded_features]
        
        # Create interaction features
        X = merged_data[selected_features]
        interaction_features = create_interaction_features(X)
        X = pd.concat([X, interaction_features], axis=1)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        y = merged_data['deadstatus.event']
        
        logging.info(f"Final number of features for modeling: {X_scaled.shape[1]}")
        logging.info("Feature list used:")
        logging.info(X_scaled.columns.tolist())
        
        return X_scaled, y, X_scaled.columns.tolist()
    
    except Exception as e:
        logging.error(f"Error during data loading and preprocessing: {str(e)}")
        raise

def single_modality_analysis(X, y, feature_names):
    """Single modality analysis"""
    results = {}
    
    try:
        # Distinguish imaging features vs clinical features
        imaging_features = [col for col in feature_names if col in [
            'contrast', 'correlation', 'dissimilarity', 'homogeneity',
            'volume_m', 'surface_mr', 'max_diamt'
        ]]
        
        clinical_features = [col for col in feature_names if col not in imaging_features]
        
        # Use stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Define models
        models = {
            'XGBoost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                scale_pos_weight=len(y[y==0])/len(y[y==1])
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        # Imaging features
        if imaging_features:
            X_imaging = X[imaging_features]
            
            # Use SMOTE for class imbalance
            smote = SMOTE(random_state=42)
            X_imaging_resampled, y_resampled = smote.fit_resample(X_imaging, y)
            
            results['imaging'] = {}
            
            for model_name, model in models.items():
                logging.info(f"Starting {model_name} imaging feature analysis...")
                
                # Cross-validation evaluation
                imaging_metrics = {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'auc': []
                }
                
                for train_idx, test_idx in cv.split(X_imaging_resampled, y_resampled):
                    X_train, X_test = X_imaging_resampled.iloc[train_idx], X_imaging_resampled.iloc[test_idx]
                    y_train, y_test = y_resampled.iloc[train_idx], y_resampled.iloc[test_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                    
                    fold_metrics = evaluate_model(y_test, y_pred, y_prob)
                    for metric, value in fold_metrics.items():
                        if metric != 'confusion_matrix':
                            imaging_metrics[metric].append(value)
                
                results['imaging'][model_name] = {
                    metric: {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
                    for metric, values in imaging_metrics.items()
                }
                
                logging.info(f"{model_name} imaging feature evaluation results:")
                for metric, stats in results['imaging'][model_name].items():
                    logging.info(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Clinical features
        if clinical_features:
            X_clinical = X[clinical_features]
            
            # Use SMOTE for class imbalance
            X_clinical_resampled, y_resampled = smote.fit_resample(X_clinical, y)
            
            results['clinical'] = {}
            
            for model_name, model in models.items():
                logging.info(f"Starting {model_name} clinical feature analysis...")
                
                # Cross-validation evaluation
                clinical_metrics = {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'auc': []
                }
                
                for train_idx, test_idx in cv.split(X_clinical_resampled, y_resampled):
                    X_train, X_test = X_clinical_resampled.iloc[train_idx], X_clinical_resampled.iloc[test_idx]
                    y_train, y_test = y_resampled.iloc[train_idx], y_resampled.iloc[test_idx]
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                    
                    fold_metrics = evaluate_model(y_test, y_pred, y_prob)
                    for metric, value in fold_metrics.items():
                        if metric != 'confusion_matrix':
                            clinical_metrics[metric].append(value)
                
                results['clinical'][model_name] = {
                    metric: {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
                    for metric, values in clinical_metrics.items()
                }
                
                logging.info(f"{model_name} clinical feature evaluation results:")
                for metric, stats in results['clinical'][model_name].items():
                    logging.info(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return results
    
    except Exception as e:
        logging.error(f"Error in single modality analysis: {str(e)}")
        raise

def multimodal_fusion(X, y):
    """Multimodal fusion analysis"""
    try:
        # Use SMOTE for class imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Define models
        models = {
            'XGBoost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                scale_pos_weight=len(y[y==0])/len(y[y==1])
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        results = {}
        
        for model_name, model in models.items():
            logging.info(f"Starting {model_name} multimodal fusion analysis...")
            
            # Cross-validation evaluation
            fusion_metrics = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'auc': []
            }
            
            for train_idx, test_idx in cv.split(X_resampled, y_resampled):
                X_train, X_test = X_resampled.iloc[train_idx], X_resampled.iloc[test_idx]
                y_train, y_test = y_resampled.iloc[train_idx], y_resampled.iloc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                fold_metrics = evaluate_model(y_test, y_pred, y_prob)
                for metric, value in fold_metrics.items():
                    if metric != 'confusion_matrix':
                        fusion_metrics[metric].append(value)
            
            results[model_name] = {
                metric: {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
                for metric, values in fusion_metrics.items()
            }
            
            logging.info(f"{model_name} multimodal fusion evaluation results:")
            for metric, stats in results[model_name].items():
                logging.info(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return results
    
    except Exception as e:
        logging.error(f"Error in multimodal fusion analysis: {str(e)}")
        raise

def feature_importance_analysis(X, y, feature_names):
    """Feature importance analysis"""
    try:
        # Use SMOTE for class imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Use XGBoost for feature selection
        xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            scale_pos_weight=len(y[y==0])/len(y[y==1])
        )
        
        # Fit model first
        xgb.fit(X_resampled, y_resampled)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb.feature_importances_
        })
        
        # Sort
        importance = importance.sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importance.head(20))
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig('result1/p1_4_feature_importance.png')
        
        # Save feature importance data
        importance.to_csv('result1/p1_4_feature_importance.csv', index=False)
        
        logging.info("Feature importance analysis completed")
        logging.info("Top 10 important features:")
        logging.info(importance.head(10))
        
        return importance
    
    except Exception as e:
        logging.error(f"Error in feature importance analysis: {str(e)}")
        raise

def model_interpretability_analysis(X, y, feature_names):
    """Model interpretability analysis"""
    try:
        # Use SMOTE for class imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Use XGBoost
        xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            scale_pos_weight=len(y[y==0])/len(y[y==1])
        )
        
        # Fit model
        xgb.fit(X_resampled, y_resampled)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(xgb)
        shap_values = explainer.shap_values(X_resampled)
        
        # Plot SHAP summary
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_resampled, feature_names=feature_names, show=False)
        plt.title('SHAP Summary Plot')
        plt.tight_layout()
        plt.savefig('result1/p1_4_shap_summary.png')
        
        # Save SHAP values
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.to_csv('result1/p1_4_shap_values.csv', index=False)
        
        # Calculate feature importance
        importance = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': np.abs(shap_values).mean(0)
        }).sort_values('shap_importance', ascending=False)
        
        logging.info("Model interpretability analysis completed")
        logging.info("Top 10 important features (based on SHAP values):")
        logging.info(importance.head(10))
        
        return importance
    
    except Exception as e:
        logging.error(f"Error in model interpretability analysis: {str(e)}")
        raise

def model_comparison(X, y, feature_names):
    """Model comparison analysis"""
    try:
        # Use SMOTE for class imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Define models
        models = {
            'XGBoost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                scale_pos_weight=len(y[y==0])/len(y[y==1])
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        # Use stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        results = {}
        
        for model_name, model in models.items():
            logging.info(f"Starting {model_name} model analysis...")
            
            # Cross-validation evaluation
            metrics = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'auc': []
            }
            
            for train_idx, test_idx in cv.split(X_resampled, y_resampled):
                X_train, X_test = X_resampled.iloc[train_idx], X_resampled.iloc[test_idx]
                y_train, y_test = y_resampled.iloc[train_idx], y_resampled.iloc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                fold_metrics = evaluate_model(y_test, y_pred, y_prob)
                for metric, value in fold_metrics.items():
                    if metric != 'confusion_matrix':
                        metrics[metric].append(value)
            
            results[model_name] = {
                metric: {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
                for metric, values in metrics.items()
            }
            
            logging.info(f"{model_name} evaluation results:")
            for metric, stats in results[model_name].items():
                logging.info(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            
            # Calculate SHAP values (only for XGBoost)
            if model_name == 'XGBoost':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_resampled)
                
                # Plot SHAP summary
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_resampled, feature_names=feature_names, show=False)
                plt.title(f'SHAP Summary Plot - {model_name}')
                plt.tight_layout()
                plt.savefig(f'result1/p1_4_shap_summary_{model_name}.png')
            
            # Calculate feature importance
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save feature importance data
                importance.to_csv(f'result1/p1_4_feature_importance_{model_name}.csv', index=False)
                
                logging.info(f"{model_name} Top 10 important features:")
                logging.info(importance.head(10))
        
        return results
    
    except Exception as e:
        logging.error(f"Error in model comparison analysis: {str(e)}")
        raise

def main():
    """Main function"""
    try:
        logging.info("Starting data loading and preprocessing...")
        X, y, feature_names = load_and_preprocess_data()
        
        logging.info("Performing single modality analysis...")
        single_results = single_modality_analysis(X, y, feature_names)
        logging.info(f"Single modality analysis results: {single_results}")
        
        logging.info("Performing multimodal fusion analysis...")
        fusion_results = multimodal_fusion(X, y)
        logging.info(f"Multimodal fusion results: {fusion_results}")
        
        logging.info("Performing feature importance analysis...")
        feature_importance = feature_importance_analysis(X, y, feature_names)
        
        logging.info("Performing model interpretability analysis...")
        model_interpretability = model_interpretability_analysis(X, y, feature_names)
        
        logging.info("Performing model comparison analysis...")
        comparison_results = model_comparison(X, y, feature_names)
        
        # Save results
        results = {
            'single_modality': single_results,
            'multimodal_fusion': fusion_results,
            'feature_importance': feature_importance.to_dict(),
            'model_interpretability': model_interpretability.to_dict(),
            'model_comparison': comparison_results
        }
        
        pd.DataFrame(results).to_csv('result1/p1_4_analysis_results.csv')
        logging.info("Analysis completed, results saved to result1/p1_4_analysis_results.csv")
    
    except Exception as e:
        logging.error(f"Error during program execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 