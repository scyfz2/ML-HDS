# Lung Cancer Prognosis Prediction

This project provides a comprehensive data preprocessing and analysis pipeline for lung cancer prognosis prediction using multi-modal data, including clinical, imaging, and RNA sequencing data.

## Project Overview

The project consists of two main datasets:
1. LUNG1 dataset (dataset1)
2. AMC and R01 datasets (dataset2)

Each dataset contains:
- Clinical data (demographics, diagnosis, treatment)
- CT imaging data with tumor segmentation
- RNA sequencing data (for dataset2)
- Comprehensive metadata

## Key Features

- Multi-modal data integration
- Automated feature extraction from medical images
- Advanced machine learning analysis
- Comprehensive clinical data processing
- RNA sequencing data analysis

## Data Processing Pipeline

1. Clinical Data Processing:
   - Data cleaning and standardization
   - Missing value handling
   - Feature engineering
   - Categorical variable encoding

2. Imaging Feature Extraction:
   - DICOM image processing
   - Tumor segmentation analysis
   - Morphological feature calculation
   - Radiomic feature extraction

3. RNA Sequencing Analysis:
   - Expression data normalization
   - Gene filtering
   - Differential expression analysis
   - Pathway analysis

## Machine Learning Analysis

- Feature selection and importance analysis
- Model training and validation
- Performance evaluation
- SHAP value analysis for interpretability
- Cross-validation and hyperparameter tuning

## Requirements

- Python 3.8+
- Required packages listed in requirements.txt
- Sufficient disk space for medical images
- Adequate RAM for processing large datasets

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Organize data according to the specified directory structure

3. Run the preprocessing pipeline:
```bash
python data_preprocessing.py
```

4. Execute analysis scripts:
```bash
python p2_6_xgboost_analysis.py
```

## Output

The pipeline generates:
- Processed clinical data
- Extracted imaging features
- Processed RNA sequencing data
- Machine learning model results
- Feature importance analysis
- Performance metrics

## Notes

- Ensure proper organization of input data files
- Follow the specified directory structure
- Process data in the correct sequence
- For detailed information about each part of the project, please refer to:
  - [P1 README](p1/README.md) - Clinical data and imaging feature analysis
  - [P2 README](p2/README.md) - RNA sequencing data analysis


