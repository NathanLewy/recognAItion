import librosa
import os
import numpy as np



    ### EXTRACTION DES SIGNAUX ###

chemin_dossier = r"C:\Users\Alex\Desktop\CS\2A\Projet IA\recognaition\DATASET\wav"
fichiers_audios = os.listdir(chemin_dossier)

enregistrements = []

for f in fichiers_audios:
    f_path = os.path.join(chemin_dossier,f)
    data, sr = librosa.load(f_path)
    enregistrements.append(data)
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
    enregistrements_decoupes.append(fenetres)
    # fenetre est un array dont chaque ligne est une fenetre 

print(len(enregistrements) == len(enregistrements_decoupes))