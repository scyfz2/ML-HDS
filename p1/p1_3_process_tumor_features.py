import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def process_tumor_features():
    """Process tumor feature data"""
    try:
        # Get the absolute path of the current script
        current_dir = Path(__file__).parent.absolute()
        workspace_root = current_dir.parent.absolute()
        
        # Define paths using absolute paths
        input_path = workspace_root / 'result1' / 'p1_2_tumor_features.csv'
        output_path = workspace_root / 'result1' / 'p1_3_processed_tumor_features.csv'
        
        # Check if input file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Check if output directory exists, create if not
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load original data
        logging.info(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path)
        
        # Select features to keep
        selected_features = [
            'contrast',
            'correlation',
            'dissimilarity',
            'homogeneity',
            'volume_m',
            'surface_mr',
            'max_diamt'
        ]
        
        # Check if these features exist
        missing_features = [f for f in selected_features if f not in df.columns]
        if missing_features:
            logging.warning(f"The following features do not exist in the data: {missing_features}")
            selected_features = [f for f in selected_features if f in df.columns]
        
        # Add PatientID for subsequent analysis
        analysis_features = selected_features + ['subject_id']
        
        # Create new DataFrame
        processed_df = df[analysis_features].copy()
        
        # Rename subject_id to PatientID
        processed_df = processed_df.rename(columns={'subject_id': 'PatientID'})
        
        # Check and convert data types
        for feature in selected_features:
            if processed_df[feature].dtype == 'object':
                try:
                    processed_df[feature] = pd.to_numeric(processed_df[feature], errors='coerce')
                    logging.info(f"Converted feature {feature} to numeric type")
                except Exception as e:
                    logging.error(f"Error converting feature {feature}: {str(e)}")
                    raise
        
        # Check for missing values
        missing_values = processed_df.isnull().sum()
        if missing_values.any():
            logging.warning("Missing values found:")
            logging.warning(missing_values[missing_values > 0])
            # Fill missing values with median
            processed_df = processed_df.fillna(processed_df.median())
        
        # Save processed data
        processed_df.to_csv(output_path, index=False)
        logging.info(f"Processed data saved to {output_path}, containing {len(selected_features)} features")
        
        return processed_df
    
    except Exception as e:
        logging.error(f"Error occurred during processing: {str(e)}")
        raise

if __name__ == "__main__":
    process_tumor_features() 