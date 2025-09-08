"""
This module provides functionality to extract audio from video files and transcribe it using the Whisper model.
"""

#--------------------------------------- Imports ---------------------------------------#

import numpy as np
import whisper
from moviepy import VideoFileClip
import os
from datasets import Audio
from scipy.io import wavfile

import webrtcvad

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

def load_audio_wav(path, target_rate=16000):
    """Load a wav file as 16kHz mono PCM16 bytes + numpy array."""
    import soundfile as sf
    audio, sr = sf.read(path)
    if sr != target_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_rate)
    if audio.ndim > 1:  # stereo → mono
        audio = np.mean(audio, axis=1)
    # convert to PCM16
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes(), audio_int16, target_rate


def run_vad(audio_bytes, sample_rate=16000, frame_ms=30, mode=2):
    vad = webrtcvad.Vad(mode)  # aggressiveness: 0–3
    frame_len = int(sample_rate * frame_ms / 1000)  # samples per frame
    step = frame_len * 2  # 2 bytes per sample (16-bit PCM)
    flags = []
    for i in range(0, len(audio_bytes), step):
        frame = audio_bytes[i:i+step]
        if len(frame) < step:
            break
        flags.append(int(vad.is_speech(frame, sample_rate)))
    return np.array(flags)  # shape [T_audio]


def get_waveform(audio_path):
    sample_rate, data = wavfile.read(audio_path)

    if data.ndim > 1:
        data = data.mean(axis=1)

    # Normalize to [-1, 1] if int16
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0

    x, y = np.arange(len(data)) / sample_rate, data
    return x, y