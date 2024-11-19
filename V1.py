import librosa
import os
import numpy as np
import subprocess



    ### EXTRACTION DES CARACT2RISTIQUES ###

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
T = 0.02
_, sr = librosa.load(os.path.join(chemin_dossier,fichiers_audios[0]), sr = None)
nb_donnees_fenetre = int(T * sr)
nb_chevauchement_fenetre = nb_donnees_fenetre   # gère le chevauchement entre les fenetres

enregistrements_decoupes = []
for e in enregistrements:
    fenetres = librosa.util.frame(data, frame_length=nb_donnees_fenetre, hop_length=nb_chevauchement_fenetre)
    fenetres = np.transpose(fenetres)
    enregistrements_decoupes.append([fenetres,fenetres.shape])
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
n_mfcc = 13

def extraire_mfcc(enregistrement_decoupe):
    mfcc_features = []
    for j in range(enregistrement_decoupe[1][0]):
        mfcc_features.append(librosa.feature.mfcc(y=enregistrement_decoupe[0][j], sr = sr, n_mfcc=n_mfcc))
    return mfcc_features

mfcc_features = [extraire_mfcc(enregistrements_decoupes[i]) for i in range(len(enregistrements_decoupes))]




