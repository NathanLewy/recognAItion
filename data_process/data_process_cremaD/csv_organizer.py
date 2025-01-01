
import pandas as pd
import os

class CsvBatchTransformer:
    def __init__(self):
        self.input_folder = "/home/pierres/Projet_S7/recognAItion/data/filtered_audio_to_csv_not_organized"
        self.target_folder = "/home/pierres/Projet_S7/recognAItion/data/filtered_audio_to_csv"

        # Create target folder if it doesn't exist
        os.makedirs(self.target_folder, exist_ok=True)

    def transform_csv(self, input_folder, target_folder):
        """
        Transforms a single CSV file into a 1xnm format with renamed columns.
        """
        # Load the CSV file
        df = pd.read_csv(input_folder)

        # Get the original column names
        original_columns = df.columns.tolist()

        # Flatten the data into a single row
        flattened_data = df.values.flatten()

        # Create new column names following the format "name_j"
        new_columns = [
            f"{original_columns[col_idx]}_{row_idx+1}"
            for row_idx in range(df.shape[0])
            for col_idx in range(df.shape[1])
        ]

        # Create a DataFrame with one row and new column names
        transformed_df = pd.DataFrame([flattened_data], columns=new_columns)

        # Save to the output file
        transformed_df.to_csv(target_folder, index=False)

    def process_all_files(self):
        """
        Processes all CSV files in the source folder and saves transformed files to the target folder.
        """
        csv_files = [f for f in os.listdir(self.input_folder) if f.endswith('.csv')]

        for csv_file in csv_files:
            input_path = os.path.join(self.input_folder, csv_file)
            output_path = os.path.join(self.target_folder, f"transformed_{csv_file}")
            print(f"Processing {csv_file}...")
            self.transform_csv(input_path, output_path)
            print(f"Transformed file saved to {output_path}")

