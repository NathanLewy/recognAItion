import os
import shutil

class AudioFilter:
    def __init__(self,emotions):
        """
        Initializes the AudioFilter with hard-coded source and target folders and keywords.
        """
        self.source_folder = "/home/pierres/Projet_S7/recognAItion/data/AudioWAV"
        self.target_folder = "/home/pierres/Projet_S7/recognAItion/data/FilteredAudio"
        self.keywords = emotions

    def ensure_target_folder(self):
        """Ensures the target folder exists."""
        os.makedirs(self.target_folder, exist_ok=True)

    def filter_files(self):
        """Filters and copies files containing specified keywords."""
        print(f"Source folder: {self.source_folder}")
        print(f"Target folder: {self.target_folder}")
        
        # Check if the source folder exists
        if not os.path.exists(self.source_folder):
            print(f"Error: Source folder not found: {self.source_folder}")
            return
        
        # Ensure the target folder exists
        self.ensure_target_folder()
        
        # Process and filter files
        for filename in os.listdir(self.source_folder):
            if any(keyword in filename for keyword in self.keywords):
                source_path = os.path.join(self.source_folder, filename)
                target_path = os.path.join(self.target_folder, filename)
                shutil.copy(source_path, target_path)

        print(f"Filtered files copied to: {self.target_folder}")

