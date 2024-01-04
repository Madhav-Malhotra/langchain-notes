import argparse
from langchain.document_loaders import YoutubeLoader

def get_transcript(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    return loader.load()

# Setup arg parser
parser = argparse.ArgumentParser(description='Load a transcript from a YouTube video.')
parser.add_argument('url', type=str, help='YouTube URL')
args = parser.parse_args()

# Load transcript
transcript = get_transcript(args.url)
print(transcript)