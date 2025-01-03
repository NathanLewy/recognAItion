import os
import pandas as pd

class CsvDataOrganizer:
    def __init__(self):
        self.source_folder = "/home/pierres/Projet_S7/recognAItion/data_emodb/features_organized"
        self.samples_file = "/home/pierres/Projet_S7/recognAItion/data_emodb/sample.csv"
        self.labels_file = "/home/pierres/Projet_S7/recognAItion/data_emodb/labels.csv"

        # Ensure the output files are cleared or replaced
        self.clear_output_files()

    def clear_output_files(self):
        """Deletes existing output files if they exist."""
        if os.path.exists(self.samples_file):
            os.remove(self.samples_file)
        if os.path.exists(self.labels_file):
            os.remove(self.labels_file)

    def create_samples_and_labels(self):
        samples = []
        labels = []

        # Iterate through all CSV files in the source folder
        csv_files = [f for f in os.listdir(self.source_folder) if f.endswith('.csv')]

        for csv_file in csv_files:

            file_path = os.path.join(self.source_folder, csv_file)
            

            # Read the second line (excluding the header) of the current CSV file
            df = pd.read_csv(file_path, skiprows=1, header=None)
           

            if not df.empty:
                samples.append(df.iloc[0].tolist())  # Add the second line as a list to samples

            print(df.iloc[0].tolist())
            
            # Assign a label based on the presence of specific letters in the file name
            if 'W' in csv_file:
                label = 0
            elif 'E' in csv_file:
                label = 1
            elif 'A' in csv_file:
                label = 2
            elif 'F' in csv_file:
                label = 3
            elif 'T' in csv_file:
                label = 4
            elif 'N' in csv_file:
                label = 5
            else:
                label = -1  # Default case if none of the letters are found (optional)

            labels.append(label)


        # Save samples and labels to their respective CSV files
        samples_df = pd.DataFrame(samples)
        labels_df = pd.DataFrame(labels, columns=['Label'])
        samples_df.to_csv(self.samples_file, index=False, header=False)
        labels_df.to_csv(self.labels_file, index=False, header=False)
        
       

        print(f"Samples saved to {self.samples_file}")
        print(f"Labels saved to {self.labels_file}")
