import csv
import os
from pathlib import Path

# Input file name
input_file = './p2/rnaseq.txt'
# Output file name
output_file = './result2/p2_1_patient_ids.csv'

# Create directories if they don't exist
os.makedirs(os.path.dirname(input_file), exist_ok=True)
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Open input file and read the first row
with open(input_file, 'r') as infile:
    reader = csv.reader(infile, delimiter='\t')
    # Read the first row
    first_row = next(reader)

# Extract patient IDs, skip the first element (usually column name)
patient_ids = first_row[1:]

# Write patient IDs to CSV file, one ID per row
with open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    for patient_id in patient_ids:
        writer.writerow([patient_id])

print(f'Patient IDs have been extracted and saved to {output_file}')