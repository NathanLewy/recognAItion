import os
import shutil

class AudioFilter:
    def __init__(self, emotions):
        """
        Initialize the AudioFilter class with the base path to the project.

        :param emotions: List of emotion identifiers to filter files by.
        """
        self.base_path = "/home/pierres/Projet_S7/recognAItion/"
        self.wav_folder = os.path.join(self.base_path, "data_emodb", "wav")
        self.emotions = emotions
        self.filtered_audio_folder = os.path.join(self.base_path, "data_emodb", "FilteredAudio")

    def create_filtered_folder(self):
        """Create the 'FilteredAudio' folder if it doesn't exist."""
        os.makedirs(self.filtered_audio_folder, exist_ok=True)

    def clear_filtered_folder(self):
        """Delete all files in the 'FilteredAudio' folder."""
        if os.path.exists(self.filtered_audio_folder):
            for file_name in os.listdir(self.filtered_audio_folder):
                file_path = os.path.join(self.filtered_audio_folder, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def filter_and_copy_files(self):
        """
        Filter the .wav files containing specified emotions in their names
        and copy them to the 'FilteredAudio' folder.
        """
        # Create the folder if it doesn't exist
        self.create_filtered_folder()

        # Clear the folder before copying new files
        self.clear_filtered_folder()

        # Iterate over files in the "wav" folder
        for file_name in os.listdir(self.wav_folder):
            if file_name.endswith(".wav"):
                for emo in self.emotions:
                    if emo in file_name:
                        source = os.path.join(self.wav_folder, file_name)
                        destination = os.path.join(self.filtered_audio_folder, file_name)
                        shutil.copy2(source, destination)

        print(f"Filtered files have been copied to {self.filtered_audio_folder}.")

