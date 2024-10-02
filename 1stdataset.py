import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from collections import Counter

# Function to extract MFCC features from an audio file
def extract_features(file_path):
    try:
        # Load the audio file
        signal, sr = librosa.load(file_path, sr=16000)  # Set sample rate to 16 kHz

        # Extract MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        return mfccs_mean
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Path to the directory containing audio files
audio_directory = '/Users/paullemaire/Documents/wav'  # Modify this path based on your actual location

# Prepare data and labels lists
data = []
labels = []

# Emotion mapping based on your classification scheme
emotion_mapping = {
    'W': 'anger',
    'L': 'boredom',
    'E': 'disgust',
    'A': 'anxiety',
    'F': 'happiness',
    'T': 'sadness',
    'N': 'neutral'
}

# Iterate through each file in the audio directory
for file_name in os.listdir(audio_directory):
    if file_name.lower().endswith('.wav'):  # Check for .wav files (case-insensitive)
        file_path = os.path.join(audio_directory, file_name)
        
        # Extract features
        features = extract_features(file_path)
        if features is not None:  # Ensure feature extraction succeeded
            data.append(features)

            # Extract the emotion label from the filename
            emotion_label = file_name[5]  # Assuming the emotion label is the 7th character (index 6)
            if emotion_label in emotion_mapping:  # Check if the label is in the mapping
                labels.append(emotion_mapping[emotion_label])
            else:
                print (f"{emotion_mapping}")
                print(f"Label {emotion_label} not recognized in file {file_name}. Skipping.")

# Create a DataFrame from the extracted features
df = pd.DataFrame(data)
if not df.empty:
    df['label'] = labels
else:
    print("No features extracted. Exiting.")
    exit()

# Check the data preparation
print("Data extracted and DataFrame created.")
print(df.head())
print(f"Total samples: {df.shape[0]}")
print(f"Unique labels: {df['label'].nunique()}")
print(f"Labels: {df['label'].value_counts()}")  # Count of each label

# Convert to numpy arrays
X = df.drop('label', axis=1).values
y = df['label'].values

# Check for empty features
if X.shape[0] == 0:
    print("No features extracted. Exiting.")
    exit()

# Filter classes with less than 2 samples
class_counts = Counter(y)

min_samples = 2
classes_to_keep = [label for label, count in class_counts.items() if count >= min_samples]
X_filtered = X[np.isin(y, classes_to_keep)]
y_filtered = y[np.isin(y, classes_to_keep)]

# Check if there are any samples left after filtering
if X_filtered.shape[0] == 0:
    print("No samples left after filtering. Exiting.")
    exit()

# Split data into training and test sets
if len(set(y_filtered)) < 2:
    print("Not enough unique classes for stratified splitting. Proceeding with a regular split.")
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Debugging print statements
print("Labels in y_train:", label_encoder.classes_)
print("Labels in y_test:", set(y_test))  # Print unique labels in y_test

# Create and train the SVM classifier
svm_model = SVC(kernel='linear', random_state=42)  # Use a linear kernel
svm_model.fit(X_train, y_train_encoded)

# Make predictions
y_pred = svm_model.predict(X_test)

# Classification report
print("Classification Report:")
print(classification_report(
    y_test_encoded, y_pred,
    target_names=label_encoder.classes_  # Use original class names
))
