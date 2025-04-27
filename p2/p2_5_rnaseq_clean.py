import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def clean_rnaseq_data(file_path):
    """Clean RNA-seq data"""
    try:
        print("Loading RNA-seq data...")
        rnaseq_data = pd.read_csv(file_path, sep='\t', index_col=0, na_values=['NA', ''])
        
        # Calculate NA ratio for each row
        na_ratio = rnaseq_data.isna().sum(axis=1) / len(rnaseq_data.columns)
        
        # Remove rows with NA ratio > 50%
        rnaseq_data = rnaseq_data[na_ratio <= 0.5]
        print(f"Removed rows with >50% NA values. Remaining shape: {rnaseq_data.shape}")
        
        # Remove columns that are all NA
        rnaseq_data = rnaseq_data.dropna(axis=1, how='all')
        print(f"Removed all-NA columns. Remaining shape: {rnaseq_data.shape}")
        
        # Fill remaining missing values
        rnaseq_data = rnaseq_data.fillna(rnaseq_data.mean())
        
        # Save cleaned data
        rnaseq_data.to_csv('result2/rnaseq_cleaned.txt', sep='\t')
        print("Saved cleaned data to result2/rnaseq_cleaned.txt")
        
        return rnaseq_data
    
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        raise

def select_features(rnaseq_data, clinical_data_path, max_features=20):
    """Select most important features"""
    try:
        print("Loading clinical data...")
        # Load clinical data
        clinical_data = pd.read_csv(clinical_data_path)
        clinical_data = clinical_data.rename(columns={'Case ID': 'PatientID'})
        clinical_data['PatientID'] = clinical_data['PatientID'].str.replace('R0', 'R01-')
        clinical_data['PatientID'] = clinical_data['PatientID'].str.replace('R01-1-', 'R01-')  # Remove -1-
        clinical_data['deadstatus.event'] = (clinical_data['Survival Status'] == 'Dead').astype(int)
        
        # Prepare data
        rnaseq_data = rnaseq_data.T
        rnaseq_data = rnaseq_data.reset_index()
        rnaseq_data = rnaseq_data.rename(columns={'index': 'PatientID'})
        rnaseq_data['PatientID'] = rnaseq_data['PatientID'].str.replace('R0', 'R01-')
        rnaseq_data['PatientID'] = rnaseq_data['PatientID'].str.replace('R01-1-', 'R01-')  # Remove -1-
        
        # Merge data
        merged_data = pd.merge(rnaseq_data, clinical_data[['PatientID', 'deadstatus.event']], 
                             on='PatientID', how='inner')
        print(f"Merged data shape: {merged_data.shape}")
        
        # Separate features and target variable
        X = merged_data.drop(['PatientID', 'deadstatus.event'], axis=1)
        y = merged_data['deadstatus.event']
        
        # 1. Initial feature selection using variance threshold
        print("Performing variance threshold selection...")
        selector = VarianceThreshold(threshold=0.1)
        X_selected = selector.fit_transform(X)
        selected_features = X.columns[selector.get_support()].tolist()
        print(f"Selected {len(selected_features)} features after variance threshold")
        
        # 2. Further feature selection using mutual information
        if len(selected_features) > max_features:
            print("Performing mutual information selection...")
            mi_scores = mutual_info_classif(X[selected_features], y)
            mi_df = pd.DataFrame({
                'feature': selected_features,
                'mi_score': mi_scores
            })
            mi_df = mi_df.sort_values('mi_score', ascending=False)
            selected_features = mi_df.head(max_features)['feature'].tolist()
            print(f"Selected top {len(selected_features)} features by mutual information")
            
            # Plot feature importance
            plt.figure(figsize=(12, 6))
            sns.barplot(x='mi_score', y='feature', data=mi_df.head(max_features))
            plt.title('Top 20 Features by Mutual Information')
            plt.tight_layout()
            plt.savefig('result2/p2_5_rna_feature_importance.png')
            print("Saved feature importance plot")
            
            # SHAP analysis
            print("Performing SHAP analysis...")
            from xgboost import XGBClassifier
            model = XGBClassifier(random_state=42)
            model.fit(X[selected_features], y)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[selected_features])
            
            # Plot SHAP summary
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X[selected_features], feature_names=selected_features, show=False)
            plt.title('SHAP Summary Plot')
            plt.tight_layout()
            plt.savefig('result2/p2_5_shap_summary.png')
            print("Saved SHAP summary plot")
        
        # Save selected features
        selected_data = rnaseq_data[['PatientID'] + selected_features]
        selected_data.to_csv('result2/p2_5_rnaseq_selected.txt', sep='\t', index=False)
        print("Saved selected features to result2/p2_5_rnaseq_selected.txt")
        
        # Save feature importance information
        if len(selected_features) > max_features:
            mi_df.to_csv('result2/p2_5_feature_importance.csv', index=False)
            print("Saved feature importance information")
        
        return selected_features
    
    except Exception as e:
        print(f"Error during feature selection: {str(e)}")
        raise

def main():
    """Main function"""
    try:
        print("Starting RNA-seq data processing...")
        # Clean data
        cleaned_data = clean_rnaseq_data('p2/rnaseq.txt')
        
        # Select features
        selected_features = select_features(cleaned_data, 'p2/clinical2.csv')
        print(f"Processing completed. Selected {len(selected_features)} features.")
    
    except Exception as e:
        print(f"Error during program execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 