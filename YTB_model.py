import torch
import torch.nn as nn
import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Charger les données avec les séquences temporelles
def load_data_with_temporal_features(path):
    data = []
    summary = pd.read_csv(os.path.join(path, 'emotion_summary.csv'))
    emotions = ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']
    
    count = 0
    for root, _, files in os.walk(os.path.join(path, 'fragments')):
        for file in files:
            if file.endswith('.mp3'):
                filepath = os.path.join(root, file)
                video_id = file[:-4].split('_', maxsplit=1)[1]
                
                # Charger l'audio
                y, sr = librosa.load(filepath, sr=None)
                
                # Filtrer les informations sur la vidéo
                filtered_summary = summary[summary['id'] == video_id]
                emotions_dict = filtered_summary.iloc[0].to_dict()
                
                # Extraire les MFCC
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
                
                # Créer le vecteur de probabilités pour les émotions
                emotions_vector = np.array([emotions_dict[emotion] for emotion in emotions])
                
                # Ajouter les données
                data.append({
                    'features': mfcc.T,  # Dimensions : (time_steps, n_mfcc)
                    'label': emotions_vector  # Vecteur de probabilités pour chaque émotion
                })
                
            count += 1
            print(f"{count} out of 21500")
            if count > 10000:
                return data
    return data

# Préparer les données
load_dotenv()
path = os.getenv('YTB_DIR')
data = load_data_with_temporal_features(path)

# Récupération des caractéristiques et des étiquettes
X = [torch.tensor(item['features'], dtype=torch.float32) for item in data]
y = [torch.tensor(item['label'], dtype=torch.float32) for item in data]
# Trier les séquences par longueur (requis pour pack_padded_sequence)
sequence_lengths = [seq.shape[0] for seq in X]
sorted_indices = np.argsort(-np.array(sequence_lengths))  # Tri décroissant
X = [X[i] for i in sorted_indices]
y = [y[i] for i in sorted_indices]
sequence_lengths = [sequence_lengths[i] for i in sorted_indices]

# Padding des séquences
max_seq_length = max(sequence_lengths)
X_padded = torch.stack([torch.cat([seq, torch.zeros(max_seq_length - seq.shape[0], seq.shape[1])]) for seq in X])

# Séparer les données en entraînement et test
X_train, X_test, y_train, y_test, lengths_train, lengths_test = train_test_split(
    X_padded, y, sequence_lengths, test_size=0.4, random_state=42
)


# Dataset et DataLoader
class AudioTemporalDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X = X
        self.y = y
        self.lengths = lengths
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]

train_dataset = AudioTemporalDataset(X_train, y_train, lengths_train)
test_dataset = AudioTemporalDataset(X_test, y_test, lengths_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modèle LSTM pour la classification multilabel
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Model, self).__init__()
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, lengths):
        # Pack les séquences
        packed_input = pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.LSTM(packed_input)
        
        # Décoder le dernier état caché (hidden state)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Passer par le fully connected layer et appliquer sigmoid pour obtenir des probabilités entre 0 et 1
        output = self.fc(hidden[-1])
        return output

# Initialisation du modèle
input_size = X_padded.shape[2]
hidden_size = 164
output_size = 7  # Le nombre d'émotions
num_layers = 3

model = Model(input_size, hidden_size, output_size, num_layers)

# Critère et optimiseur
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Données prêtes avec `pack_padded_sequence`.")

# Fonction d'entraînement
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch in train_loader:
            inputs, labels, lengths = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Passer les longueurs dans un tenseur sur le même device
            lengths = torch.tensor(lengths, dtype=torch.long)

            # Réinitialiser les gradients
            optimizer.zero_grad()
            
            # Passage avant
            outputs = model(inputs, lengths)
            
            # Calcul de la perte
            loss = criterion(outputs, labels)
            
            # Passage arrière et optimisation
            loss.backward()
            optimizer.step()
            
            # Calcul des statistiques
            running_loss += loss.item() * inputs.size(0)
            total_samples += labels.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")





def evaluate_model(model, test_loader, criterion, device):
    model.eval()  # Mettre le modèle en mode évaluation
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # Désactiver le calcul des gradients pour l'évaluation
        for batch in test_loader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            lengths = batch[2]  # Longueurs des séquences

            # Passer les entrées et les longueurs à travers le modèle
            outputs = model(inputs, lengths)

            # Collecte des labels et des prédictions
            all_labels.extend(labels.cpu().numpy())  # Convertir en numpy pour les métriques
            all_predictions.extend(outputs.cpu().numpy())  # Convertir en numpy

    # Convertir les listes en numpy arrays pour calculer les métriques
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Calcul des métriques de régression
    mse = mean_squared_error(all_labels, all_predictions)
    mae = mean_absolute_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)


    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")

# Entraînement du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model(model, train_loader, criterion, optimizer, num_epochs=30, device=device)

# Évaluation du modèle
evaluate_model(model, test_loader, criterion, device=device)



torch.save(model.state_dict(),"emotion_recognition_lstm.pth")