import whisper
from moviepy import VideoFileClip
import os

def extract_audio(video_path, audio_path):
    """Extract audio from video and save as .wav"""
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path, codec='pcm_s16le')  # ensures Whisper compatibility

def transcribe_audio(audio_path, model):
    """Transcribe audio using Whisper"""    
    result = model.transcribe(audio_path)
    return result['text']

def extract_and_transcribe_audio(video_file, model):
    audio_file = "temp_audio.wav"
    extract_audio(video_file, audio_file)
    transcript = transcribe_audio(audio_file, model)
    os.remove(audio_file)
    return transcript

def main(video_file):
    # Use .wav for best compatibility with Whisper
    audio_file = "temp_audio.wav"

    extract_audio(video_file, audio_file)
    transcript = transcribe_audio(audio_file)

    print("\n--- Transcription ---\n")
    print(transcript)

    os.remove(audio_file)

if __name__ == "__main__":
    model_size = "base"  # or "small", "medium", "large"
    model = whisper.load_model(model_size)
    video_file = "videos/sample1.mp4"
    if not os.path.isfile(video_file):
        raise FileNotFoundError(f"{video_file} not found.")
    main(video_file)
