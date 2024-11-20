import os
from pytubefix import YouTube
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import csv,re
from langdetect import detect, DetectorFactory
import textstat
import shutil
import emotion_detection_comments as edc

# Configuration
API_KEY = 'AIzaSyBVUFKGabDqsD3MS6hpsDijwWqIvicnG9Q'
VIDEO_ID = 'm3IqAolevVc'  # Remplacez par votre vidéo
BASE_DIR = 'E:\\travail\dataset_music_analysis'

# Fonction pour supprimer le répertoire existant
def clear_directory(base_dir):
    """Supprime le répertoire de base s'il existe."""
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        print(f"Directory {base_dir} cleared.")

# Appeler la fonction pour nettoyer le répertoire au début
clear_directory(BASE_DIR)

youtube = build('youtube', 'v3', developerKey=API_KEY)

def create_video_directory(video_id):
    """Crée un dossier pour la vidéo dans le répertoire de base."""
    video_dir = os.path.join(BASE_DIR, video_id)
    os.makedirs(video_dir, exist_ok=True)
    return video_dir



def get_all_comments(video_id):
    """Récupère tous les commentaires d'une vidéo YouTube."""
    comments = []
    try:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            maxResults=100
        )
        
        while request and len(comments) < 50:  # Continue jusqu'à ce que 100 commentaires soient récupérés
            response = request.execute()
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                # Filtrer les commentaires anglais de plus de 20 caractères
                cleaned_comment = clean_comment(comment)
                if len(cleaned_comment) > 40:
                    if is_english(cleaned_comment) and is_quality_comment(cleaned_comment):
                        comments.append(cleaned_comment)
            
            request = youtube.commentThreads().list_next(request, response)
        print(f"{len(comments)} appropriate comments found")

    except HttpError as e:
        print(f"An error occurred while retrieving comments: {e}")
    
    return comments

def clean_comment(comment):
    """Supprime les caractères non valides d'un commentaire."""
    # Conserve uniquement les lettres, les chiffres, les espaces, et les ponctuations courantes
    return re.sub(r'[^a-zA-Z0-9\s.,;!?\'"()\-:~]', '', comment)

def is_english(comment):
    """Vérifie si le commentaire est en anglais."""
    try:
        # Détecte la langue du commentaire
        return detect(comment) == 'en'
    except:
        return False  # Retourne False en cas d'erreur

def is_quality_comment(comment, min_readability_score=100):
    """Vérifie si le commentaire a un niveau de qualité acceptable."""
    return (textstat.flesch_reading_ease(comment) >= min_readability_score)

def save_comments(audio_folder, comments):
    """Enregistre les commentaires dans un fichier CSV dans le même dossier que l'audio."""
    filename = os.path.join(audio_folder, f'comments.csv')
    
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['comments'])
        for comment in comments: # Nettoyer le commentaire
            writer.writerow([comment])  # Écriture du commentaire nettoyé

def download_audio(video_id):
    """Télécharge l'audio de la vidéo YouTube et l'enregistre dans le dossier de la vidéo."""
    video_dir = create_video_directory(video_id)
    try:
        url = f'https://www.youtube.com/watch?v={video_id}'
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        if audio_stream is None:
            print("No audio stream found for this video.")
            return
        
        audio_file_path = os.path.join(video_dir, f"{video_id}.mp3")
        audio_stream.download(output_path=video_dir, filename=f"{video_id}.mp3")
        print(f"Audio downloaded as {audio_file_path}")
        
    except Exception as e:
        print(f"An error occurred while downloading audio: {e}")


def get_video_ids_from_playlist(playlist_id):
    """Récupère tous les identifiants de vidéo dans une playlist YouTube."""
    video_ids = []
    try:
        request = youtube.playlistItems().list(
            part='contentDetails',
            playlistId=playlist_id,
            maxResults=50
        )
        
        while request:
            response = request.execute()
            for item in response['items']:
                video_ids.append(item['contentDetails']['videoId'])
                
            request = youtube.playlistItems().list_next(request, response)
    
    except HttpError as e:
        print(f"An error occurred while retrieving playlist: {e}")
    
    return video_ids

def process_video(video_id):
    """Télécharge les commentaires et l'audio pour une vidéo spécifique."""
    comments = get_all_comments(video_id)
    if comments:
        download_audio(video_id)
        audio_folder = BASE_DIR + '\\'+ str(video_id)
        save_comments(audio_folder, comments)
        edc.create_emotion_summary(audio_folder)
        
    else:
        print(f"No comments retrieved for video {video_id}.")
    

def process_playlist(playlist_id):
    """Gère une playlist entière : télécharge les commentaires et l'audio pour chaque vidéo."""
    video_ids = get_video_ids_from_playlist(playlist_id)
    print(f"Found {len(video_ids)} videos in the playlist.")
    
    for video_id in video_ids:
        print(f"Processing video {video_id}")
        process_video(video_id)

# Exemple d'appel
PLAYLIST_ID = 'PLplXQ2cg9B_qrCVd1J_iId5SvP8Kf_BfS'
process_playlist(PLAYLIST_ID)






