import librosa
import numpy as np
# haha lol
# Fonction pour extraire les caractéristiques MFCC

def extract_features(file_path):
    # Charger le fichier audio
    signal, sr = librosa.load(file_path, sr=None)

    # Extraire les MFCC
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    return mfccs_mean

# Test de la fonction
file_path = '/Users/paullemaire/Documents/IEMOCAP_full_release_withoutVideos.tar.gz'
features = extract_features(file_path)
print(features)
