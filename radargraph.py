import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Modèle LSTM pour la classification multilabel
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Model, self).__init__()
        # Définir la couche LSTM
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Couche fully connected pour la sortie (probabilités pour chaque émotion)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, lengths):
        # Pack les séquences pour traiter des longueurs variables
        packed_input = pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)
        # Passer par la couche LSTM
        packed_output, (hidden, _) = self.LSTM(packed_input)
        
        # Décoder les résultats
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Prendre le dernier état caché pour prédire la sortie
        output = self.fc(hidden[-1])  # hidden[-1] pour le dernier état caché
        return output

# Fonction pour obtenir les MFCC d'un fichier audio
def extract_mfcc_from_audio(file_path, duration=6, sr=22050):
    # Charger l'audio (en limitant à 'duration' secondes)
    y, sr = librosa.load(file_path, sr=sr, duration=duration)

    # Extraire les MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)

    return torch.tensor(mfcc.T, dtype=torch.float32)

# Fonction pour obtenir les probabilités d'appartenance aux émotions
def get_emotion_probabilities(model, audio_file, device='cpu', duration=6):
    # Extraire les MFCC du fichier audio
    mfcc_input = extract_mfcc_from_audio(audio_file, duration)
    
    # Ajouter un batch dimension
    mfcc_input = mfcc_input.unsqueeze(0).to(device)

    # Passer l'entrée à travers le modèle
    model.eval()  # Mettre le modèle en mode évaluation
    with torch.no_grad():
        output = model(mfcc_input, [mfcc_input.shape[1]])  # Longueur de la séquence

    # Appliquer la fonction sigmoïde pour obtenir des probabilités
    probabilities = output.squeeze().cpu().numpy()

    return probabilities

# Fonction pour créer le graphique radar
def plot_radar_chart(probabilities, labels):
    # Nombre de classes (émotions)
    num_vars = len(labels)

    # Convertir les probabilités en valeurs entre 0 et 1
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    probabilities = np.concatenate((probabilities, [probabilities[0]]))  # Fermer le graphique
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.fill(angles, probabilities, color='blue', alpha=0.25)
    ax.plot(angles, probabilities, color='blue', linewidth=2)  # Ajouter une ligne pour les probabilités

    # Étiquettes des classes (émotions)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)

    plt.title('Probabilités d\'appartenance aux émotions', size=14)
    plt.show()

# Fonction principale pour exécuter le modèle sur un échantillon de 6 secondes et afficher le graphique radar
def analyze_sample(model, audio_file, device='cpu'):
    emotions = ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise']
    
    # Obtenir les probabilités d'émotions
    probabilities = get_emotion_probabilities(model, audio_file, device=device)

    # Tracer le graphique radar
    plot_radar_chart(probabilities, emotions)

# Charger le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(input_size=128, hidden_size=200, output_size=7, num_layers=4)
model.load_state_dict(torch.load("lstm_ytb.pth", map_location=torch.device('cpu')))
model.to(device)

# Utiliser un fichier audio d'exemple pour l'analyse
audio_file = './08b10Wa.wav'

# Analyser l'échantillon et afficher le graphique radar
analyze_sample(model, audio_file, device=device)
