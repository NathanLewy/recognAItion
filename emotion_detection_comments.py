import pandas as pd
from transformers import pipeline
import os

# Initialiser le pipeline d'analyse des émotions (GoEmotions pour une gamme variée)
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Fonction pour analyser les émotions
def analyze_emotions(comment):
    try:
        results = emotion_analyzer(comment)
        # Transformer les résultats en dictionnaire {emotion: score}
        emotion_scores = {res['label']: res['score'] for res in results[0]}
        return emotion_scores
    except Exception as e:
        return {}


#creer un fichier csv qui résume les emotions de la video
def create_emotion_summary(BASE_DIR,video_id):
    file_path = BASE_DIR + '\\'+ str(video_id)
    comments_data = pd.read_csv(file_path+'\comments.csv')
    # Appliquer l'analyse des émotions à chaque commentaire
    comments_data['emotion_scores'] = comments_data['comments'].apply(analyze_emotions)

    # Extraire les émotions principales et les scores
    emotions = ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']
    for emotion in emotions:
        comments_data[emotion] = comments_data['emotion_scores'].apply(lambda x: x.get(emotion, 0))

    # Calculer le bilan des émotions
    emotion_summary = comments_data[emotions].mean()
    emotion_summary = pd.DataFrame([emotion_summary.values], columns=emotions)
    emotion_summary["id"] = video_id
    emotion_summary = emotion_summary[["id"] + emotions]

    # Sauvegarder uniquement le résumé dans un fichier CSV
    output_file = BASE_DIR+'\\emotion_summary.csv'
    output_exists = os.path.exists(output_file)
    
    if not output_exists:
        emotion_summary.to_csv(output_file, index=False)
        print("Le fichier n'existait pas. Il a été créé avec les nouvelles données.")
    else:
        # Si le fichier existe déjà, ajouter les nouvelles données
        emotion_summary.to_csv(output_file, mode='a', header=False, index=False)
        print("Les nouvelles données ont été ajoutées au fichier existant.")


    # Afficher le bilan global des émotions
    print("Bilan global des émotions :")
    print(emotion_summary)

    print(f"Analyse terminée. Résultats détaillés enregistrés dans {output_file}.")
