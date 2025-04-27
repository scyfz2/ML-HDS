import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif

# Create result2 directory
os.makedirs('result2', exist_ok=True)

def create_interaction_features(X):
    """Create interaction features"""
    interaction_features = pd.DataFrame()
    
    # Imaging feature interactions
    imaging_features = ['volume_m', 'surface_mr', 'max_diamt']
    available_imaging = [f for f in imaging_features if f in X.columns]
    
    for i in range(len(available_imaging)):
        for j in range(i+1, len(available_imaging)):
            interaction_features[f'{available_imaging[i]}_{available_imaging[j]}'] = X[available_imaging[i]] * X[available_imaging[j]]
    
    # Clinical feature interactions
    clinical_features = ['Age at Histological Diagnosis']
    available_clinical = [f for f in clinical_features if f in X.columns]
    gender_features = [col for col in X.columns if col.startswith('Gender_')]
    
    for feature in available_clinical:
        for img_feature in available_imaging:
            interaction_features[f'{feature}_{img_feature}'] = X[feature] * X[img_feature]
    
    for gender_feature in gender_features:
        for img_feature in available_imaging:
            interaction_features[f'{gender_feature}_{img_feature}'] = X[gender_feature] * X[img_feature]
    
    print(f"Created {len(interaction_features.columns)} interaction features")
    if len(interaction_features.columns) > 0:
        print("Interaction features:", interaction_features.columns.tolist())
    
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
        print("Loading data...")
        tumor_features = pd.read_csv('result2/p2_4_segmentation_features.csv')
        clinical_data = pd.read_csv('result2/p2_2_cleaned_clinical_data.csv')
        rnaseq_data = pd.read_csv('result2/p2_5_rnaseq_selected.txt', sep='\t')
        
        # Record original data information
        print("Data shapes:")
        print(f"Tumor features: {tumor_features.shape}")
        print(f"Clinical data: {clinical_data.shape}")
        print(f"RNA-seq data: {rnaseq_data.shape}")
        
        # Standardize ID column names
        tumor_features = tumor_features.rename(columns={'subject_id': 'PatientID'})
        
        # Process PatientID format
        tumor_features['PatientID'] = tumor_features['PatientID'].str.replace('R01-1-', 'R01-')
        clinical_data['Case ID'] = clinical_data['Case ID'].str.replace('R01-1-1-', 'R01-')
        rnaseq_data['PatientID'] = rnaseq_data['PatientID'].str.replace('R01-1-1-', 'R01-')
        
        # Merge data based on tumor features
        print("Merging data...")
        # 1. First merge tumor features and clinical data
        merged_data = pd.merge(tumor_features, clinical_data, left_on='PatientID', right_on='Case ID', how='left')
        print(f"After merging tumor and clinical: {merged_data.shape}")
        
        # 2. Then merge RNA-seq data
        merged_data = pd.merge(merged_data, rnaseq_data, on='PatientID', how='left')
        print(f"Final merged shape: {merged_data.shape}")
        
        # Check for missing values
        missing_stats = merged_data.isnull().sum()
        missing_stats = missing_stats[missing_stats > 0]
        if not missing_stats.empty:
            print("Filling missing values...")
            # Fill missing values in numeric features with median
            numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
            merged_data[numeric_cols] = merged_data[numeric_cols].fillna(merged_data[numeric_cols].median())
        
        # Process target variable
        merged_data['deadstatus.event'] = (merged_data['Survival Status'] == 'Dead').astype(int)
        
        # Define features to exclude
        excluded_features = [
            'PatientID', 'Case ID',  # Identifiers
            'deadstatus.event',  # Target variable
            'Survival Status',  # Survival related
            'Time to Death (days)', 'Date of Death', 'Date of Last Known Alive',  # Time related
            'Recurrence', 'Recurrence Location', 'Date of Recurrence'  # Recurrence related
        ]
        
        # Keep only existing columns
        existing_excluded_features = [col for col in excluded_features if col in merged_data.columns]
        
        # Identify feature types
        numeric_features = merged_data.select_dtypes(include=[np.number]).columns
        categorical_features = merged_data.select_dtypes(include=['object']).columns
        
        # OneHot encode categorical features
        print("Encoding categorical features...")
        for feature in categorical_features:
            if feature not in existing_excluded_features:
                dummies = pd.get_dummies(merged_data[feature], prefix=feature)
                # Clean feature names
                dummies.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_') for col in dummies.columns]
                merged_data = pd.concat([merged_data, dummies], axis=1)
                merged_data = merged_data.drop(feature, axis=1)
        
        # Standardize numeric features
        print("Standardizing numeric features...")
        numeric_features = [col for col in numeric_features if col not in existing_excluded_features]
        if len(numeric_features) > 0:
            scaler = StandardScaler()
            merged_data[numeric_features] = scaler.fit_transform(merged_data[numeric_features])
        
        # Create interaction features
        X = merged_data.drop(existing_excluded_features, axis=1)
        interaction_features = create_interaction_features(X)
        
        # Standardize interaction features if they exist
        if not interaction_features.empty:
            X = pd.concat([X, interaction_features], axis=1)
            interaction_scaler = StandardScaler()
            X[interaction_features.columns] = interaction_scaler.fit_transform(X[interaction_features.columns])
        
        y = merged_data['deadstatus.event']
        
        # Clean feature names
        X.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_') for col in X.columns]
        
        print(f"Final feature count: {X.shape[1]}")
        print(f"Target distribution: {y.value_counts(normalize=True)}")
        
        return X, y, X.columns.tolist()
    
    except Exception as e:
        print(f"Error during data loading and preprocessing: {str(e)}")
        raise

def single_modality_analysis(X, y, feature_names):
    """Single modality analysis"""
    results = {}
    
    try:
        # Distinguish imaging features, clinical features, and RNA-seq features
        imaging_features = [col for col in feature_names if col in [
            'volume_m', 'surface_mr', 'max_diamt', 'compactn',
            'volume_m_surface_mr', 'volume_m_max_diamt', 'surface_mr_max_diamt'
        ]]
        
        clinical_features = [col for col in feature_names if col not in imaging_features and not col.startswith('ENSG')]
        rnaseq_features = [col for col in feature_names if col.startswith('ENSG')]
        
        # Use stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Calculate class weights
        class_weights = {}
        for label in [0, 1]:
            count = len(y[y == label])
            if count > 0:
                class_weights[label] = len(y) / (2 * count)
            else:
                class_weights[label] = 1.0
        
        # Define XGBoost model (with regularization)
        xgb_model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.01,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            scale_pos_weight=class_weights.get(1, 1.0)
        )
        
        # Imaging features
        if imaging_features:
            print("\nEvaluating imaging features...")
            X_imaging = X[imaging_features]
            results['imaging'] = evaluate_modality(X_imaging, y, xgb_model, cv)
            print("Imaging features results:")
            for metric, stats in results['imaging'].items():
                print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Clinical features
        if clinical_features:
            print("\nEvaluating clinical features...")
            X_clinical = X[clinical_features]
            results['clinical'] = evaluate_modality(X_clinical, y, xgb_model, cv)
            print("Clinical features results:")
            for metric, stats in results['clinical'].items():
                print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # RNA-seq features
        if rnaseq_features:
            print("\nEvaluating RNA-seq features...")
            X_rnaseq = X[rnaseq_features]
            results['rnaseq'] = evaluate_modality(X_rnaseq, y, xgb_model, cv)
            print("RNA-seq features results:")
            for metric, stats in results['rnaseq'].items():
                print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return results
    
    except Exception as e:
        print(f"Error in single modality analysis: {str(e)}")
        raise

def evaluate_modality(X, y, model, cv):
    """Evaluate single modality performance"""
    try:
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
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
        
        return {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for metric, values in metrics.items()
        }
    
    except Exception as e:
        print(f"Error in modality evaluation: {str(e)}")
        raise

def multimodal_fusion(X, y):
    """Multimodal fusion analysis"""
    try:
        print("\nPerforming multimodal fusion analysis...")
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Calculate class weights
        class_weights = {}
        for label in [0, 1]:
            count = len(y[y == label])
            if count > 0:
                class_weights[label] = len(y) / (2 * count)
            else:
                class_weights[label] = 1.0
        
        # Define XGBoost model (with regularization)
        xgb_model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.01,  # Lower learning rate
            max_depth=3,  # Reduce tree depth
            min_child_weight=5,  # Increase minimum leaf node samples
            subsample=0.8,  # Use subsampling
            colsample_bytree=0.8,  # Use feature sampling
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=42,
            scale_pos_weight=class_weights.get(1, 1.0)
        )
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        results = {}
        
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
            
            xgb_model.fit(X_train, y_train)
            y_pred = xgb_model.predict(X_test)
            y_prob = xgb_model.predict_proba(X_test)[:, 1]
            
            fold_metrics = evaluate_model(y_test, y_pred, y_prob)
            for metric, value in fold_metrics.items():
                if metric != 'confusion_matrix':
                    fusion_metrics[metric].append(value)
        
        results = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for metric, values in fusion_metrics.items()
        }
        
        print("Multimodal fusion results:")
        for metric, stats in results.items():
            print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return results
    
    except Exception as e:
        print(f"Error in multimodal fusion analysis: {str(e)}")
        raise

def feature_importance_analysis(X, y, feature_names):
    """Feature importance analysis"""
    try:
        print("\nPerforming feature importance analysis...")
        # Handle class imbalance with SMOTE
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
        plt.savefig('result2/p2_6_feature_importance.png')
        
        # Save feature importance data
        importance.to_csv('result2/p2_6_feature_importance.csv', index=False)
        
        print("Top 10 important features:")
        print(importance.head(10))
        
        return importance
    
    except Exception as e:
        print(f"Error in feature importance analysis: {str(e)}")
        raise

def model_interpretability_analysis(X, y, feature_names):
    """Model interpretability analysis"""
    try:
        print("\nPerforming model interpretability analysis...")
        # Handle class imbalance with SMOTE
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
        plt.savefig('result2/p2_6_shap_summary.png')
        
        # Save SHAP values
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.to_csv('result2/p2_6_shap_values.csv', index=False)
        
        # Calculate feature importance
        importance = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': np.abs(shap_values).mean(0)
        }).sort_values('shap_importance', ascending=False)
        
        print("Top 10 important features (based on SHAP values):")
        print(importance.head(10))
        
        return importance
    
    except Exception as e:
        print(f"Error in model interpretability analysis: {str(e)}")
        raise

def main():
    """Main function"""
    try:
        print("Starting analysis...")
        X, y, feature_names = load_and_preprocess_data()
        
        print("\nPerforming single modality analysis...")
        single_results = single_modality_analysis(X, y, feature_names)
        
        print("\nPerforming multimodal fusion analysis...")
        fusion_results = multimodal_fusion(X, y)
        
        print("\nPerforming feature importance analysis...")
        feature_importance = feature_importance_analysis(X, y, feature_names)
        
        print("\nPerforming model interpretability analysis...")
        model_interpretability = model_interpretability_analysis(X, y, feature_names)
        
        # Save results
        results = {
            'single_modality': single_results,
            'multimodal_fusion': fusion_results,
            'feature_importance': feature_importance.to_dict(),
            'model_interpretability': model_interpretability.to_dict()
        }
        
        pd.DataFrame(results).to_csv('result2/p2_6_analysis_results.csv')
        print("\nAnalysis completed, results saved to result2/p2_6_analysis_results.csv")
    
    except Exception as e:
        print(f"Error during program execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 