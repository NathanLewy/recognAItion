import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
features_folder = "ExtractedFeatures"  # Folder containing the feature CSV files
num_files_to_plot = 5  # Number of files to plot

# List all CSV files in the features folder
csv_files = [f for f in os.listdir(features_folder) if f.endswith('.csv')]

# Ensure there are at least `num_files_to_plot` CSV files
if len(csv_files) < num_files_to_plot:
    raise ValueError(f"Not enough CSV files in {features_folder}. Found {len(csv_files)}, need {num_files_to_plot}.")

# Select the first `num_files_to_plot` files (or choose randomly)
selected_csv_files = csv_files[:num_files_to_plot]

# Process and plot each selected file in a separate graph
for csv_file in selected_csv_files:
    # Load the CSV file
    file_path = os.path.join(features_folder, csv_file)
    df = pd.read_csv(file_path)

    # Select the first 200 rows and specific columns (adjust as needed)
    df_subset = df.iloc[:200, :]  # Assuming all relevant columns are in this range

    # Convert all columns to numeric, handling non-numeric data
    df_subset = df_subset.apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values
    df_subset = df_subset.dropna()

    # Ensure the subset has data to plot
    if df_subset.empty:
        print(f"No valid data to plot in {csv_file}. Skipping.")
        continue

    # Create a new figure for each file
    plt.figure(figsize=(10, 6))
    
    # Plot each column
    for column in df_subset.columns:
        plt.plot(
            df_subset.index.to_numpy(),
            df_subset[column].to_numpy(),
            label=column,
            marker='o'
        )

    # Customize the plot
    plt.title(f"Features from {csv_file}")
    plt.xlabel("Row Index")
    plt.ylabel("Feature Values")
    plt.legend(title="Columns", loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

