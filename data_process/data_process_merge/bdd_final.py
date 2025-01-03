import os
import pandas as pd

class CsvEquiCombiner:
    def __init__(self):
        self.data_emodb_folder = "/home/pierres/Projet_S7/recognAItion/data_emodb"
        self.data_folder = "/home/pierres/Projet_S7/recognAItion/data"
        self.data_final_folder = "/home/pierres/Projet_S7/recognAItion/data_final"

        # Input file paths
        self.sample_emodb_file = os.path.join(self.data_emodb_folder, "sample.csv")
        self.label_emodb_file = os.path.join(self.data_emodb_folder, "labels.csv")
        self.sample_data_file = os.path.join(self.data_folder, "sample.csv")
        self.label_data_file = os.path.join(self.data_folder, "labels.csv")

        # Output file paths
        self.sample_equi_file = os.path.join(self.data_final_folder, "sample_equi.csv")
        self.label_equi_file = os.path.join(self.data_final_folder, "label_equi.csv")

        # Create the output folder if it doesn't exist
        os.makedirs(self.data_final_folder, exist_ok=True)

    def combine_and_shuffle_csv_files(self):
        # Load the input files
        sample_emodb_df = pd.read_csv(self.sample_emodb_file, header=None,skiprows = 1)
        label_emodb_df = pd.read_csv(self.label_emodb_file, header=None,skiprows = 1)
        sample_data_df = pd.read_csv(self.sample_data_file, header=None,skiprows = 1)
        label_data_df = pd.read_csv(self.label_data_file, header=None,skiprows = 1)

        # Determine the size of the smallest file
        min_size = min(len(sample_emodb_df), len(sample_data_df))

        # Limit the size of each dataset to twice the smallest size
        sample_emodb_limited = sample_emodb_df.iloc[:min_size]
        label_emodb_limited = label_emodb_df.iloc[:min_size]
        sample_data_limited = sample_data_df.iloc[:min_size]
        label_data_limited = label_data_df.iloc[:min_size]

        # Combine the datasets
        sample_equi_df = pd.concat([sample_emodb_limited, sample_data_limited], ignore_index=True)
        label_equi_df = pd.concat([label_emodb_limited, label_data_limited], ignore_index=True)

        # Shuffle the combined dataset while keeping the association
        shuffled_indices = sample_equi_df.sample(frac=1, random_state=42).index
        sample_equi_df = sample_equi_df.iloc[shuffled_indices].reset_index(drop=True)
        label_equi_df = label_equi_df.iloc[shuffled_indices].reset_index(drop=True)

        # Save the combined and shuffled datasets to the output files
        sample_equi_df.to_csv(self.sample_equi_file, index=False, header=False)
        label_equi_df.to_csv(self.label_equi_file, index=False, header=False)

        print(f"'sample_equi.csv' created at: {self.sample_equi_file}")
        print(f"'label_equi.csv' created at: {self.label_equi_file}")

# Example usage
if __name__ == "__main__":
    combiner = CsvEquiCombiner()
    combiner.combine_and_shuffle_csv_files()

