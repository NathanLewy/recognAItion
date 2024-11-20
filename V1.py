import librosa
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



    ### EXTRACTION DES CARACTERISTIQUES ###

chemin_dossier = r"C:\Users\Alex\Desktop\CS\2A\Projet IA\recognaition\DATASET\wav"
fichiers_audios = os.listdir(chemin_dossier)

# Création d'un dico qui attribue un nombre à chaque émotion 
Dico_Emo = {"W":1,"L":2,"E":3,"A":4,"F":5,"T":6,"N":7}


enregistrements = []
emotions = []


for f in fichiers_audios:
    f_path = os.path.join(chemin_dossier,f)
    data, sr = librosa.load(f_path)
    enregistrements.append(data)
    emotions.append(Dico_Emo[f[5]])


    """
    print(f"traitement du fichier : {f}")

    try :
        data, sr = librosa.load(f_path, sr = None)
        print(sr)
        print(f"La durer de l'enregistrement est de {len(data) /sr} secondes") 
    except Exception as exception:
        print(f"Erreur lors du traitement de {f} : {exception}")
   """



    ### DIVISION DES SIGNAUX EN FENETRES ###

# fenetre de T secondes 
T = 0.04
_, sr = librosa.load(os.path.join(chemin_dossier,fichiers_audios[0]), sr = None)
nb_donnees_fenetre = int(T * sr)
nb_chevauchement_fenetre = nb_donnees_fenetre   # gère le chevauchement entre les fenetres

enregistrements_decoupes = []
for i in range(len(enregistrements)):
    fenetres = librosa.util.frame(enregistrements[i], frame_length=nb_donnees_fenetre, hop_length=nb_chevauchement_fenetre)
    fenetres = np.transpose(fenetres)
    enregistrements_decoupes.append(fenetres)
    # fenetre est un array dont chaque ligne est une fenetre 



"""
def extraire_pitch(enregistrement_decoupe):
    pitch = []
    for i in range(len(enregistrement_decoupe)):
        pitch.append(librosa.pyin(enregistrement_decoupe[i], fmin=80, fmax=400)[0])
    print(pitch)
    return pitch

Pitch = [extraire_pitch(enregistrements_decoupes[i]) for i in range(len(enregistrements_decoupes))]
"""

#nombre de coefs à extraire
n_mfcc = 3

# fast fourrier transform 
n_fft = 512  

"""
def extraire_mfcc(enregistrement_decoupe):
    mfcc_features = np.array([[]])
    for j in range(len(enregistrement_decoupe)):
        mfcc_features = np.append(mfcc_features, np.transpose(librosa.feature.mfcc(y=enregistrement_decoupe[j], sr = sr, n_fft=n_fft, n_mfcc=n_mfcc)), axis=1)
    return mfcc_features
"""

def extraire_mfcc(enregistrement_decoupe):
    mfcc_features = np.empty((0, n_mfcc))  # Array initialisé avec 0 lignes, n_mfcc colonnes
    for j in range(len(enregistrement_decoupe)):
        mfcc_segment = np.transpose(librosa.feature.mfcc(y=enregistrement_decoupe[j], sr=sr, n_fft=n_fft, n_mfcc=n_mfcc))
        mfcc_features = np.vstack([mfcc_features, mfcc_segment])  # Concaténer verticalement
        #print(mfcc_segment)
    #print(mfcc_features)
    return mfcc_features


mfcc_features = [extraire_mfcc(enregistrements_decoupes[i]) for i in range(len(enregistrements_decoupes))] 
#ne peut pas être convertie en array à cause des différence de dimensions entre ses éléments

max = max([mfcc.shape[0] for mfcc in mfcc_features])
mfcc_features_adapte = np.zeros((len(mfcc_features),max,n_mfcc))
for i, mfcc in enumerate(mfcc_features):
    mfcc_features_adapte[i, :mfcc.shape[0], :] = mfcc
print("ok")




"""
    ### KNN ###

class KNN:
    def __init__(self, k=5):
        self.k = k
    
    def entrainement(self, X, Y):
        self.X_entrainement = list(X)
        self.Y_entrainement = list(Y)

    def prediction(self, x):
        distances = np.linalg.norm(self.X_entrainement - x, axis = 1 )    
        k_indices = np.argsort(distances)[:self.k].tolist  # Trouve les indices des k plus proches voisins
        k_labels_proches = [self.Y_entrainement[i] for i in k_indices]  # Obtient les étiquettes des données les plus proches
        most_common = Counter(k_labels_proches).most_common(1)  # Identifie l'étiquette la plus représentée
        return most_common[0][0]


    def predictions(self, X_test):
        predictions = [self.prediction(x) for x in X_test]
        return np.array(predictions)
"""

    ### ENTRAINEMENT ###

# Diviser les données en ensembles d'entraînement et de test
X_entrainement, X_test, Y_entrainement, Y_test = train_test_split(mfcc_features_adapte, emotions, test_size=0.2, random_state=42)
X_entrainement = list(X_entrainement)
X_test = list(X_test)
Y_entrainement = list(Y_entrainement)
Y_test = list(Y_test)

knn = KNN(k=3)
knn.entrainement(X_entrainement, Y_entrainement)    # Entrainement avec la classe KNN
Y_pred = knn.predictions(X_test)    # Prédictions


precision = accuracy_score(Y_test, Y_pred)
print(f"Précision du modèle KNN : {precision * 100:.2f}%")