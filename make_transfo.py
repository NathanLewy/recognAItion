import os
import pandas as pd

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

        # Initialize the transformed DataFrame
        transformed_data = {}

        # Iterate through each column in the original DataFrame
        for column in df.columns:
            for i in range(1, len(df) + 1):  # Iterate through the row indices
                new_column_name = f"{column}_{i}"  # Add the suffix `_i`
                transformed_data[new_column_name] = [df.iloc[i - 1][column]]  # Add the corresponding value

        # Convert the transformed dictionary into a new DataFrame
        transformed_df = pd.DataFrame(transformed_data)

        # Save the transformed DataFrame to the target folder
        transformed_df.to_csv(target_path, index=False)
        print(f"Transformed data saved to {target_path}")

print("Processing complete for all folders.")
