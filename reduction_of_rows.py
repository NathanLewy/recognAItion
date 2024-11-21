import os
import pandas as pd
import numpy as np

# Source and target folders
folders = {
    "GroupedFeatures_HAP_ANG_train": "TransformedFeatures_HAP_ANG_train",
    "GroupedFeatures_HAP_ANG_Val": "TransformedFeatures_HAP_ANG_Val"
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

        # Extract the coefficient names from the first row
        coefficient_names = df.columns.tolist()

        # Drop the first row to work with numeric data
        df = df.apply(pd.to_numeric, errors='coerce')

        # Initialize the transformed data
        transformed_data = []

        # Generate the transformed matrix b_(i, j)
        for i in range(1, 49):  # i ranges from 1 to 48 (48 rows)
            for j in range(1, 3):  # j ranges from 1 to 2 (2 columns)
                # Calculate the row index in a_(1, i * j)
                a_row_idx = (i * j) - 1

                # Ensure the index is within bounds
                if a_row_idx < len(df):
                    value = df.iloc[a_row_idx, 0]  # Take value from a_(1, i * j)
                    new_name = f"{coefficient_names[0]}_{i}_{j}"  # Append i and j to the name
                else:
                    value = np.nan  # Assign NaN if out of bounds
                    new_name = f"{coefficient_names[0]}_{i}_{j}"

                # Append the transformed value
                transformed_data.append([new_name, value])

        # Convert the transformed data into a DataFrame
        transformed_df = pd.DataFrame(transformed_data, columns=["Coefficient_Name", "Value"])

        # Save the transformed data to the target folder
        transformed_df.to_csv(target_path, index=False)
        print(f"Transformed data saved to {target_path}")

print("Processing complete for all folders.")
