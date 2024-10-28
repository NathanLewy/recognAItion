from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Remplacez par votre clé API
API_KEY = 'key'  # Assure-toi d'utiliser une clé valide
VIDEO_ID = 'oTU25_wmWp4'  # ID de la vidéo YouTube

youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_comments(video_id):
    """Récupère les commentaires d'une vidéo YouTube."""
    comments = []
    
    try:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText'
        ).execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

    except HttpError as e:
        print(f"An error occurred while retrieving comments: {e}")
        # Gérer spécifiquement les erreurs liées aux commentaires désactivés
        if e.resp.status == 403:
            print(f"Comments are disabled for video: {video_id}")
        elif e.resp.status == 404:
            print(f"Video not found: {video_id}")

    return comments

# Appel de la fonction
comments = get_comments(VIDEO_ID)
if comments:
    print("Retrieved comments:")
    print(comments)
else:
    print("No comments retrieved.")
