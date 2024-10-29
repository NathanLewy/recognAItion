import os
from pytubefix import YouTube
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configuration
API_KEY = 'AIzaSyBVUFKGabDqsD3MS6hpsDijwWqIvicnG9Q'
VIDEO_ID = 'oTU25_wmWp4'  # Remplacez par votre vidéo
BASE_DIR = 'D:\\dataset_music_analysis'

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
        
        while request:
            response = request.execute()
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                # Filtrer les commentaires de plus de 20 caractères
                if len(comment) > 20:
                    comments.append(comment)
                
            request = youtube.commentThreads().list_next(request, response)
    
    except HttpError as e:
        print(f"An error occurred while retrieving comments: {e}")
    
    return comments

def save_comments(video_id, comments):
    """Sauvegarde les commentaires dans un fichier texte."""
    video_dir = create_video_directory(video_id)
    comments_file_path = os.path.join(video_dir, 'comments.txt')
    
    with open(comments_file_path, 'w', encoding='utf-8') as file:
        for comment in comments:
            file.write(comment + '\n')
    
    print(f"Comments saved to {comments_file_path}")

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

# Appel des fonctions
comments = get_all_comments(VIDEO_ID)
if comments:
    save_comments(VIDEO_ID, comments)
else:
    print("No comments retrieved.")

download_audio(VIDEO_ID)
