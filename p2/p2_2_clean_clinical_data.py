import pandas as pd
import os
import numpy as np

def clean_clinical_data():
    """Clean clinical data, keeping only specified PatientIDs"""
    try:
        # Load data
        patient_ids = pd.read_csv('result2/p2_1_patient_ids.csv', header=None, names=['PatientID'])
        clinical_data = pd.read_csv('p2/clinical2.csv')
        
        # Check required columns
        if 'Case ID' not in clinical_data.columns:
            raise ValueError("Clinical data missing Case ID column")
        
        # Process PatientID format
        clinical_data['Case ID'] = clinical_data['Case ID'].str.replace('AMC-', 'R01-')
        patient_ids['PatientID'] = patient_ids['PatientID'].str.replace('R01-1-1-', 'R01-')
        
        # Check PatientID format
        common_ids = set(patient_ids['PatientID']).intersection(set(clinical_data['Case ID']))
        
        # Filter data and make a copy to avoid SettingWithCopyWarning
        filtered_clinical_data = clinical_data[clinical_data['Case ID'].isin(patient_ids['PatientID'])].copy()
        
        # Check for complete match
        if len(filtered_clinical_data) != len(patient_ids):
            missing_ids = set(patient_ids['PatientID']) - set(filtered_clinical_data['Case ID'])
        
        # Handle missing values
        # 1. Identify feature types
        numeric_features = filtered_clinical_data.select_dtypes(include=[np.number]).columns
        categorical_features = filtered_clinical_data.select_dtypes(include=['object']).columns
        
        # 2. Process missing values in numeric features
        for feature in numeric_features:
            if filtered_clinical_data[feature].isnull().sum() > 0:
                median_value = filtered_clinical_data[feature].median()
                filtered_clinical_data.loc[:, feature] = filtered_clinical_data[feature].fillna(median_value)
        
        # 3. Process missing values in categorical features
        for feature in categorical_features:
            if filtered_clinical_data[feature].isnull().sum() > 0:
                mode_value = filtered_clinical_data[feature].mode()[0]
                filtered_clinical_data.loc[:, feature] = filtered_clinical_data[feature].fillna(mode_value)
        
        # Save results
        output_file = 'result2/p2_2_cleaned_clinical_data.csv'
        filtered_clinical_data.to_csv(output_file, index=False)
        
        return filtered_clinical_data
    
    except Exception as e:
        raise

if __name__ == "__main__":
    clean_clinical_data() 