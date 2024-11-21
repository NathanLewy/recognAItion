import os
import pandas as pd
import numpy as np

# Source and target folders
folders = {
    "ExtractedFeatures_HAP_ANG_train": "GroupedFeatures_HAP_ANG_train",
    "ExtractedFeatures_HAP_ANG_Val": "GroupedFeatures_HAP_ANG_Val"
}

# Process each source folder
for source_folder, target_folder in folders.items():
    # Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # List all CSV files in the source folder
    csv_files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]

    for csv_file in csv_files:
        source_path = os.path.join(source_folder, csv_file)
        target_path = os.path.join(target_folder, csv_file)

        # Load the CSV file
        df = pd.read_csv(source_path)

        # Ensure numeric conversion
        df = df.apply(pd.to_numeric, errors='coerce')

        # Group rows by chunks of 8 and compute the mean for each group
        grouped = df.groupby(np.arange(len(df)) // 8).mean()

        # Save the grouped data to the target folder
        grouped.to_csv(target_path, index=False)
        print(f"Grouped data saved to {target_path}")

print("Processing complete for all folders.")
