""" 
Description: Importing dependencies in the helper
"""

import re
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, _errors
from youtube_transcript_api.proxies import GenericProxyConfig
from langchain.schema import Document

""" 
Description: Function to extract the video id from the video.
"""

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


""" 
Description: Function to extract the video information from the video.
"""

def get_video_info(youtube_url):
    with yt_dlp.YoutubeDL() as ydl:
        info = ydl.extract_info(youtube_url, download=False)
    return info.get("title", "").replace("\u2060", ""), info.get("uploader", "")


""" 
Description: Function to extract the video transcript of any language like english, hindi, french etc from the video.
"""

def get_transcript_url(video_id):
    try:
        transcripts = YouTubeTranscriptApi(proxy_config=GenericProxyConfig(
        https_url="https://www.croxyproxy.com/")).list_transcripts(video_id)
        for t in transcripts:
            try:
                original = t.fetch()
                snippets = getattr(original, "snippets", original) 
                text = " ".join([s.text if hasattr(s, "text") else s["text"] for s in snippets])
                return [Document(page_content=text)]
            except:
                continue
        return None
    except (_errors.TranscriptsDisabled, _errors.NoTranscriptFound):
        pass
        
