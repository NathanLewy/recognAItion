import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import librosa
from pydub import AudioSegment
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv

# Classe du modèle LSTM (identique à ce que vous avez déjà)
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Model, self).__init__()
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, input):
        hidden, cell_state = self.LSTM(input)
        output = hidden[:, -1, :]
        output = self.fc(output)  # Applique la couche dense après LSTM
        return output

# Convertir les fichiers MP3 en tableau de samples audio
def mp3toaudio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(22050)  # Conversion mono et à 22050 Hz
    audio = np.array(audio.get_array_of_samples())
    return audio

# Extraire les MFCCs d'un fichier audio
def extract_mfcc(samples, sr=22050, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=samples.astype(float), sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # Retournons la matrice transposée pour avoir une séquence temporelle

# Préparation des données à partir du dossier audio et du fichier CSV
def prepare_data(BASE_DIR, fragments_dir, emotion_summary_file, n_mfcc=13):
    # Charger le fichier CSV des émotions
    emotion_summary_df = pd.read_csv(emotion_summary_file)

    # Dictionnaire pour stocker les caractéristiques et les labels
    X = []
    y = []
    
    # Extraire les labels émotionnels
    emotion_columns = ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']
    
    # Encoder les émotions si nécessaire
    label_encoder = LabelEncoder()
    for col in emotion_columns:
        emotion_summary_df[col] = label_encoder.fit_transform(emotion_summary_df[col])
    
    # Parcourir le dossier des fichiers audio (MP3)
    for file_name in os.listdir(fragments_dir):
        if file_name.endswith(".mp3"):
            video_id = file_name.split('_',1)[1][:-4]  # extraire le video_id du nom du fichier
            # Vérifier si le video_id existe dans le fichier CSV
            if video_id in emotion_summary_df['id'].values:
                # Extraire les caractéristiques audio (MFCCs)
                audio_path = os.path.join(fragments_dir, file_name)
                samples = mp3toaudio(audio_path)
                mfcc_features = extract_mfcc(samples, n_mfcc=n_mfcc)
                
                # Ajouter les caractéristiques et les labels
                X.append(mfcc_features)
                
                # Extraire l'émotion pour ce `video_id` (moyenne des émotions)
                emotion_data = emotion_summary_df[emotion_summary_df['id'] == video_id][emotion_columns].values
                y.append(emotion_data.flatten())  # C'est un tableau 1D des émotions
    
    # Convertir en numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Convertir les données en tensors PyTorch
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    return X_tensor, y_tensor


# Entraînement du modèle
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # Mode entraînement
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Réinitialiser les gradients
            outputs = model(inputs)  # Passer les données dans le modèle
            loss = criterion(outputs, labels)  # Calculer la perte
            loss.backward()  # Calculer les gradients
            optimizer.step()  # Mettre à jour les poids
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Validation du modèle
def validate_model(model, val_loader, criterion):
    model.eval()  # Mode évaluation
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    print(f"Validation Loss: {val_loss/len(val_loader)}")


# Dataset personnalisé pour PyTorch
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Exemple d'utilisation
load_dotenv()
BASE_DIR = os.getenv("BASE_DIR")
fragments_dir = os.path.join(BASE_DIR, 'fragments')
emotion_summary_file = os.path.join(BASE_DIR, 'emotion_summary.csv')

X, y = prepare_data(BASE_DIR, fragments_dir, emotion_summary_file)

# Séparation des données en ensemble d'entraînement et de validation (80/20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer des DataLoaders pour l'entraînement et la validation
train_dataset = EmotionDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = EmotionDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Hyperparamètres
input_size = X_train.shape[2]  # Nombre de MFCC (13 par défaut)
hidden_size = 64  # Taille des couches cachées
output_size = y_train.shape[1]  # Nombre de classes (émotions)
num_epochs = 60
learning_rate = 0.001

# Initialiser le modèle
model = Model(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
criterion = nn.MSELoss()  # Utilisation de la perte MSE pour régression des émotions
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Entraîner le modèle
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Valider le modèle
validate_model(model, val_loader, criterion)