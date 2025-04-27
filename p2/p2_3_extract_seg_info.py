import pandas as pd
import os

def extract_seg_info():
    # Read patient_ids.csv
    patient_ids = pd.read_csv('result2/p2_1_patient_ids.csv', header=None, names=['Subject ID'])
    
    # Read metadata.csv
    metadata = pd.read_csv('p2/metadata.csv')
    
    # Filter rows where Modality is SEG
    seg_data = metadata[metadata['Modality'] == 'SEG']
    
    # Filter Subject IDs that are in patient_ids
    filtered_data = seg_data[seg_data['Subject ID'].isin(patient_ids['Subject ID'])]
    
    # Select required columns
    result = filtered_data[['Subject ID', 'Series Description', 'Manufacturer', 'File Size', 'File Location']]
    
    # Save results to CSV file
    output_file = 'result2/p2_3_seg_info.csv'
    result.to_csv(output_file, index=False)
    
    print(f"Processing completed! Found {len(result)} SEG records")
    print(f"Results saved to {output_file}")
    
    # Print some statistics
    print("\nStatistics:")
    print(f"Total number of patients: {len(patient_ids)}")
    print(f"Number of patients with SEG records: {len(result['Subject ID'].unique())}")
    print("\nManufacturer distribution:")
    print(result['Manufacturer'].value_counts())
    print("\nFile size statistics:")
    print(result['File Size'].describe())

if __name__ == "__main__":
    extract_seg_info() 