import re
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, _errors
from langchain_community.document_loaders import YoutubeLoader
from langchain.schema import Document

def extract_video_id(url):
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/|youtube\.com\/e\/|youtube\.com\/watch\?.*v=|youtube\.com\/user\/.+\/[^#]+#)([^&\n?]+)',
        r'(?:youtube\.com\/shorts\/)([^&\n?]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_info(youtube_url):
    with yt_dlp.YoutubeDL() as ydl:
        info = ydl.extract_info(youtube_url, download=False)
    return info.get("title", "").replace("\u2060", ""), info.get("uploader", "")

def get_transcript(video_id, youtube_url):
    transcript_text = ""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        for item in transcript_list:
            transcript_text += item['text'] + " "
        return [Document(page_content=transcript_text)]
    except (_errors.TranscriptsDisabled, _errors.NoTranscriptFound):
        pass
    try:
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True, language=["en"])
        return loader.load()
    except:
        pass
    return None
