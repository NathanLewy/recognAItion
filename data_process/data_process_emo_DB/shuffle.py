import pandas as pd
import numpy as np

class CSVProcessor:
    def __init__(self):
        """
        Initialize with paths to the sample and label files.
        """
        self.sample_file = "/home/pierres/Projet_S7/recognAItion/data_emodb/sample.csv"
        self.label_file = "/home/pierres/Projet_S7/recognAItion/data_emodb/labels.csv"
        self.sample_df = pd.read_csv(self.sample_file)
        self.label_df = pd.read_csv(self.label_file)
        
        # Ensure both DataFrames have the same number of rows
        assert len(self.sample_df) == len(self.label_df), "The files do not have the same number of lines."

    def shuffle_and_extract(self, output_sample_file, output_label_file, fraction=0.2):
        """
        Shuffle both files, extract a fraction of rows to new files, and delete these rows from the original files.

        Parameters:
        output_sample_file (str): Path to save the extracted sample file.
        output_label_file (str): Path to save the extracted label file.
        fraction (float): The fraction of rows to extract (default is 0.2).
        """
        # Shuffle the indices
        shuffled_indices = np.random.permutation(len(self.sample_df))

        # Apply the shuffle to both DataFrames
        sample_df_shuffled = self.sample_df.iloc[shuffled_indices].reset_index(drop=True)
        label_df_shuffled = self.label_df.iloc[shuffled_indices].reset_index(drop=True)

        # Calculate the number of rows to extract
        num_rows = int(fraction * len(self.sample_df))

        # Extract the data
        extracted_sample_df = sample_df_shuffled.iloc[:num_rows]
        extracted_label_df = label_df_shuffled.iloc[:num_rows]

        # Save the extracted data to new files
        extracted_sample_df.to_csv(output_sample_file, index=False)
        extracted_label_df.to_csv(output_label_file, index=False)

        print(f"Extracted {num_rows} rows to {output_sample_file} and {output_label_file}.")

        # Remove the extracted rows and save the remaining data back to the original files
        remaining_sample_df = sample_df_shuffled.iloc[num_rows:].reset_index(drop=True)
        remaining_label_df = label_df_shuffled.iloc[num_rows:].reset_index(drop=True)
        

        remaining_sample_df.to_csv(self.sample_file, index=False)
        remaining_label_df.to_csv(self.label_file, index=False)

       

        print(f"Remaining data saved back to {self.sample_file} and {self.label_file}.")

