import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def preprocess_dataset1():
    # Read data
    data_path = Path("p1/clinical1.csv")
    df = pd.read_csv(data_path)
    
    # Print original data information
    print("Original data information:")
    print(f"Total rows: {len(df)}")
    print("\nMissing values statistics:")
    print(df.isnull().sum())
    
    # Process continuous variables
    continuous_vars = ['age', 'Survival.time']
    for var in continuous_vars:
        median_value = df[var].median()
        df[var] = df[var].fillna(median_value)
        print(f"\nMedian value of {var}: {median_value}")
    
    # Process categorical variables
    categorical_vars = ['clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage', 
                       'Overall.Stage', 'Histology', 'gender', 'deadstatus.event']
    for var in categorical_vars:
        # Convert values to string and handle missing values
        df[var] = df[var].astype(str).replace('nan', 'Unknown')
    
    # Z-score standardization for continuous variables
    scaler = StandardScaler()
    for var in continuous_vars:
        # Save original values
        original_col = f"{var}_original"
        df[original_col] = df[var]
        # Standardize
        df[var] = scaler.fit_transform(df[[var]])
        print(f"\nStandardized statistics for {var}:")
        print(f"Mean: {df[var].mean():.4f}")
        print(f"Standard deviation: {df[var].std():.4f}")
    
    # One-hot encoding for categorical variables
    print("\nCategorical variables encoding information:")
    encoded_columns = []
    for var in categorical_vars:
        # Get unique values
        unique_values = sorted(df[var].unique())
        print(f"\n{var}:")
        print(f"Number of unique values: {len(unique_values)}")
        print(f"Unique values: {unique_values}")
        
        # Perform one-hot encoding
        dummies = pd.get_dummies(df[var], prefix=var)
        encoded_columns.extend(dummies.columns.tolist())
        
        # Add encoded columns to dataframe
        df = pd.concat([df, dummies], axis=1)
    
    # Save processed data
    output_path = Path("result1/p1_1_clinical1_processed.csv")
    df.to_csv(output_path, index=False)
    
    # Print processed data information
    print("\nProcessed data information:")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Number of new encoded columns: {len(encoded_columns)}")
    print("\nEncoded column names:")
    for col in encoded_columns:
        print(f"- {col}")
    
    return df

if __name__ == "__main__":
    processed_df = preprocess_dataset1() 