import os
import opensmile
import pandas as pd
import numpy as np
import librosa

class WavToCsvExtractor:
    def __init__(self, num_segments, columns_to_keep=None):
        self.source_folder = "/home/pierres/Projet_S7/recognAItion/data/FilteredAudio"
        self.target_folder = "/home/pierres/Projet_S7/recognAItion/data/filtered_audio_to_csv_not_organized"
        self.num_segments = num_segments
        self.columns_to_keep = columns_to_keep

        # Initialize OpenSMILE
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors  # Frame-level features
        )

        # Create the target folder if it doesn't exist
        os.makedirs(self.target_folder, exist_ok=True)

    def process_audio(self, wav_file):
        """
        Process a single .wav file and extract its features into a .csv file.

        :param wav_file: Name of the .wav file to process.
        """
        source_path = os.path.join(self.source_folder, wav_file)
        target_csv = os.path.join(self.target_folder, f"{os.path.splitext(wav_file)[0]}_features.csv")

        # Load audio and get duration
        y, sr = librosa.load(source_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        segment_duration = duration / self.num_segments

        print(f"Processing {wav_file}...")
        features = self.smile.process_file(source_path)
        features = features.reset_index()

        # Print out the columns of features to inspect


        segment_features = []
        for i in range(self.num_segments):
            start_time = np.timedelta64(int(i * segment_duration * 1e9), 'ns')
            end_time = np.timedelta64(int((i + 1) * segment_duration * 1e9), 'ns')

            segment_df = features[(features['start'] >= start_time) & (features['start'] < end_time)]

            # Aggregate features for the segment
            segment_agg = segment_df.mean(numeric_only=True)

            # Check if all requested columns exist, and only keep those that do
            if self.columns_to_keep is not None:
                missing_columns = [col for col in self.columns_to_keep if col not in features.columns]
                if missing_columns:
                    print(f"Warning: The following columns are missing and will be ignored: {missing_columns}")
                
                # Keep only the columns that exist in both the DataFrame and the requested columns
                valid_columns = [col for col in self.columns_to_keep if col in features.columns]
                segment_agg = segment_agg[valid_columns]

            segment_features.append(segment_agg)

        aggregated_features = pd.DataFrame(segment_features)
        aggregated_features.to_csv(target_csv, index=False)
        print(f"Features for {wav_file} saved to {target_csv}.")

    def process_all_files(self):
        """
        Process all .wav files in the source folder.
        """
        wav_files = [f for f in os.listdir(self.source_folder) if f.endswith('.wav')]

        for wav_file in wav_files:
            self.process_audio(wav_file)
