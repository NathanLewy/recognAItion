import os
import shutil

class AudioFilter:
    def __init__(self, base_path):
        """
        Initialize the AudioFilter class with the base path to the project.

        :param base_path: The path to the "recognAltion" folder.
        """
        self.base_path = base_path
        self.wav_folder = os.path.join(self.base_path, "data_emodb", "wav")
        self.filtered_audio_folder = os.path.join(self.base_path, "data_emodb", "FilteredAudio")

    def create_filtered_folder(self):
        """Create the 'FilteredAudio' folder if it doesn't exist."""
        os.makedirs(self.filtered_audio_folder, exist_ok=True)

    def filter_and_copy_files(self):
        """
        Filter the .wav files containing 'F' or 'T' in their names
        and copy them to the 'FilteredAudio' folder.
        """
        # Create the folder if it doesn't exist
        self.create_filtered_folder()

        # Iterate over files in the "wav" folder
        for file_name in os.listdir(self.wav_folder):
            if file_name.endswith(".wav") and ("F" in file_name or "T" in file_name):
                source = os.path.join(self.wav_folder, file_name)
                destination = os.path.join(self.filtered_audio_folder, file_name)
                shutil.copy2(source, destination)

        print(f"Files containing 'F' or 'T' have been copied to {self.filtered_audio_folder}.")

# Example usage
if __name__ == "__main__":
    # Replace this with the actual path to the "recognAltion" folder
    base_path = "/home/pierres/Projet_S7/recognAItion/"
    
    # Create an instance of the class and perform the operation
    audio_filter = AudioFilter(base_path)
    audio_filter.filter_and_copy_files()
