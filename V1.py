import librosa
import os

chemin_dossier = r"C:\Users\Alex\Desktop\CS\2A\Projet IA\recognaition\DATASET\wav"
enregistrements = os.listdir(chemin_dossier)



for e in enregistrements:
    e_path = os.path.join(chemin_dossier,e)
    """
    print(f"traitement du fichier : {e}")

    try :
        data, sr = librosa.load(e_path, sr = None)
        print(sr)
        print(f"La durer de l'enregistrement est de {len(data) /sr} secondes") 
    except Exception as exception:
        print(f"Erreur lors du traitement de {e} : {exception}")
   """
    


### Nous allons découper chaque signal en fenêtre de durer T"

T = 0.02   # En secondes 
_, sr = librosa.load(os.path.join(chemin_dossier,enregistrements[0]), sr = None)
nb_donnees = T * sr
print(T, sr, nb_donnees)