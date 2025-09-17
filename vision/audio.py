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
import contextlib
import wave
from pydub import AudioSegment

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

def extract_mono_wav(video_path, wav_path="audio.wav", sample_rate=16000):
    audio = AudioSegment.from_file(video_path)
    audio = audio.set_channels(1)         # mono
    audio = audio.set_frame_rate(sample_rate)  # 16kHz (works well with VAD)
    audio.export(wav_path, format="wav")
    return wav_path

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



def get_vad_segments(audio_path, aggressiveness=2, frame_ms=30):
    vad = webrtcvad.Vad(aggressiveness)
    with contextlib.closing(wave.open(audio_path, "rb")) as wf:
        sample_rate = wf.getframerate()
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        assert nchannels == 1, "Audio must be mono for WebRTC VAD"
        assert sample_rate in (8000, 16000, 32000, 48000)

        frame_size = int(sample_rate * frame_ms / 1000)
        segments = []
        speech = False
        start = None
        t = 0.0
        step = frame_ms / 1000.0

        while True:
            frame = wf.readframes(frame_size)
            if len(frame) < frame_size * sampwidth:
                break
            is_speech = vad.is_speech(frame, sample_rate)
            if is_speech and not speech:
                start = t
                speech = True
            elif not is_speech and speech:
                segments.append((start, t))
                speech = False
            t += step

        if speech:
            segments.append((start, t))
    return segments

def vad_flags_for_frames(segments, num_frames, fps, frame_stride):
    """Return list of True/False for each processed frame"""
    flags = []
    effective_fps = fps / frame_stride  # since you skip frames
    for i in range(num_frames):
        t = i / effective_fps
        speaking = any(start <= t <= end for start, end in segments)
        flags.append(speaking)
    return flags

def get_waveform(audio_path):
    sample_rate, data = wavfile.read(audio_path)

    if data.ndim > 1:
        data = data.mean(axis=1)

    # Normalize to [-1, 1] if int16
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0

    x, y = np.arange(len(data)) / sample_rate, data
    return x, y