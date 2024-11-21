import os
import pandas as pd

# Source and target folders
folders = {
    "TransformedFeatures_HAP_ANG_train": "CombinedFeatures_HAP_ANG_train.csv",
    "TransformedFeatures_HAP_ANG_Val": "CombinedFeatures_HAP_ANG_Val.csv"
}

# Process each folder
for source_folder, combined_file in folders.items():
    # List all CSV files in the source folder
    csv_files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]

    # Initialize an empty list to store dataframes
    all_dataframes = []

    for csv_file in csv_files:
        source_path = os.path.join(source_folder, csv_file)

        # Load the CSV file
        df = pd.read_csv(source_path)

        # Determine the emotion based on the file name
        emotion = 0 if "HAP" in csv_file else 1 if "ANG" in csv_file else None

        # Add the "Emotion" column to the DataFrame
        if emotion is not None:
            df["Emotion"] = emotion
        else:
            raise ValueError(f"File {csv_file} does not contain 'HAP' or 'ANG' in its name.")

        # Append the dataframe to the list
        all_dataframes.append(df)

    # Concatenate all dataframes into one big dataframe
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Save the combined dataframe to the target file
    combined_df.to_csv(combined_file, index=False)
    print(f"Combined CSV file with 'Emotion' column saved as {combined_file}")

print("All folders processed and combined files created.")
