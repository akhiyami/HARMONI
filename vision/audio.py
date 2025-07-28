"""
This module provides functionality to extract audio from video files and transcribe it using the Whisper model.
"""

#--------------------------------------- Imports ---------------------------------------#

import whisper
from moviepy import VideoFileClip
import os

#--------------------------------------- Functions ---------------------------------------#

def extract_audio(video_path, audio_path):
    """
    Extract audio from video and save as .wav
    Args:
        video_path (str): Path to the input video file.
        audio_path (str): Path where the extracted audio will be saved.
    """
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path, codec='pcm_s16le')  # ensures Whisper compatibility

def transcribe_audio(audio_path, model):
    """
    Transcribe audio using Whisper
    Args:
        audio_path (str): Path to the audio file.
        model: The Whisper model to use for transcription.
    Returns:
        str: Transcription of the audio.
    """
    result = model.transcribe(audio_path)
    return result['text']

def extract_and_transcribe_audio(video_file, model):
    """
    Extract audio from a video file and transcribe it using the Whisper model.
    Args:
        video_file (str): Path to the video file.
        model: The Whisper model to use for transcription.
    Returns:
        str: Transcription of the audio.
    """
    audio_file = "temp_audio.wav"
    extract_audio(video_file, audio_file)
    transcript = transcribe_audio(audio_file, model)
    os.remove(audio_file)
    return transcript


