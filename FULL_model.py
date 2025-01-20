import torch
import torch.nn as nn
import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Fonction pour extraire l'émotion depuis le nom du fichier
def extract_emotion_emodb(filename):
    emotions_map = {
        'W': 'anger',
        'L': 'boredom',
        'E': 'disgust',
        'A': 'fear',
        'F': 'happiness',
        'T': 'sadness',
        'N': 'neutral'
    }
    emotions_map_common = {
        'W': 'anger',
        'T': 'sadness',
        'E': 'disgust',
        'H': 'hapiness',
        'F': 'fear',
        'N': 'neutral'
    }
    return emotions_map_common.get(filename[5], 'unknown')

# Fonction pour extraire l'émotion depuis le nom du fichier
def extract_emotion_cremad(filename):
    emotions_map_common = {
        'ANG': 'anger',
        'SAD': 'sadness',
        'DIS': 'disgust',
        'FEA': 'fear',
        'HAP': 'hapiness',
        'NEU': 'neutral'
    }
    emotions_partial = {
        'ANG': 'anger',
        'SAD': 'sadness',
        'NEU': 'neutral'

    }
    if filename[13:15]=='HI':
        pass
    return emotions_map_common.get(filename[9:12], 'unknown')

# Charger les données avec les séquences temporelles
def load_data_with_temporal_features(path,extractor):
    data = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(root, file)
                emotion = extractor(file)
                
                if emotion!='unknown':
                    # Charger l'audio
                    y, sr = librosa.load(filepath, sr=None)
                    
                    # Extraire les MFCC
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=50)
                    
                    data.append({
                        'features': mfcc.T,  # Dimensions : (time_steps, n_mfcc)
                        'label': emotion
                    })
    return data


emodbdir = './EmoDB/wav/'
cremaddir = './CremaD/AudioWAV/'
data = load_data_with_temporal_features(emodbdir,extract_emotion_emodb)+load_data_with_temporal_features(cremaddir,extract_emotion_cremad)



# Encodage des labels
label_encoder = LabelEncoder()
labels = [item['label'] for item in data]
encoded_labels = label_encoder.fit_transform(labels)

# Récupération des caractéristiques et des étiquettes
X = [torch.tensor(item['features'], dtype=torch.float32) for item in data]
y = torch.tensor(encoded_labels, dtype=torch.long)

# Trier les séquences par longueur (requis pour pack_padded_sequence)
print('tri par longueur')
sequence_lengths = [seq.shape[0] for seq in X]
sorted_indices = np.argsort(-np.array(sequence_lengths))  # Tri décroissant
X = [X[i] for i in sorted_indices]
y = y[sorted_indices]
sequence_lengths = [sequence_lengths[i] for i in sorted_indices]

# Padding des séquences
print('padding')
max_seq_length = max(sequence_lengths)
X_padded = torch.stack([torch.cat([seq, torch.zeros(max_seq_length - seq.shape[0], seq.shape[1])]) for seq in X])

# Séparer les données en entraînement et test
X_train, X_test, y_train, y_test, lengths_train, lengths_test = train_test_split(
    X_padded, y, sequence_lengths, test_size=0.2, random_state=42
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

print('separation test train')
train_dataset = AudioTemporalDataset(X_train, y_train, lengths_train)
test_dataset = AudioTemporalDataset(X_test, y_test, lengths_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modèle LSTM
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Model, self).__init__()
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, lengths):
        # Pack les séquences
        packed_input = pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.LSTM(packed_input)
        
        # Décoder le dernier état caché (hidden state)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(hidden[-1])  # Utiliser le dernier état caché
        return output

# Initialisation du modèle
input_size = X_padded.shape[2]
hidden_size = 150
output_size = len(label_encoder.classes_)
num_layers = 4

model = Model(input_size, hidden_size, output_size, num_layers)

# Critère et optimiseur
criterion = nn.CrossEntropyLoss()
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
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")



def evaluate_model(model, test_loader, criterion, label_encoder, device='cpu'):
    model.eval()  # Passer en mode évaluation
    model.to(device)

    running_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_labels = []

    # Désactiver le calcul des gradients
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels, lengths = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Conserver lengths sur le CPU pour `pack_padded_sequence`
            lengths = torch.tensor(lengths, dtype=torch.long)
            
            # Passe avant
            outputs = model(inputs, lengths)
            
            # Calcul de la perte
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            # Prédictions
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_samples += labels.size(0)

    # Calcul de la perte moyenne
    test_loss = running_loss / total_samples

    # Conversion des étiquettes et des prédictions
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Rapport détaillé par classe
    class_names = label_encoder.classes_
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    print(f"\nTest Loss: {test_loss:.4f}")
    print("\nClassification Report:\n")
    print(report)

    # Matrice de confusion (optionnelle)
    cm = confusion_matrix(all_labels, all_predictions)
    print("\nConfusion Matrix:\n")
    print(cm)

    return test_loss, cm, report

# Entraînement du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model(model, train_loader, criterion, optimizer, num_epochs=30, device=device)

# Évaluation du modèle
evaluate_model(model, test_loader, criterion, label_encoder, device=device)
