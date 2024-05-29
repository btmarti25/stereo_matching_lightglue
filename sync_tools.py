import numpy as np
from scipy.fft import fft, ifft
from moviepy.editor import VideoFileClip
import librosa
import cv2
# Function to extract audio from a video file
def extract_audio(video_path):
    with VideoFileClip(video_path) as video:
        audio = video.audio
        audio.write_audiofile(f"{video_path}_audio.wav")

def calculate_offset_fft(audio_path1, audio_path2):
    y1, sr1 = librosa.load(audio_path1, sr=None)  # Load audio without resampling
    y2, sr2 = librosa.load(audio_path2, sr=None)

    # Ensure that both signals are the same length
    if len(y1) > len(y2):
        y2 = np.pad(y2, (0, len(y1) - len(y2)), 'constant')
    elif len(y2) > len(y1):
        y1 = np.pad(y1, (0, len(y2) - len(y1)), 'constant')

    # Compute FFT of both signals
    Y1 = fft(y1)
    Y2 = fft(y2)

    correlation = ifft(Y1 * np.conj(Y2)).real
    max_index = np.argmax(correlation)

    if max_index > len(correlation) / 2:
        lag = max_index - len(correlation)
    else:
        lag = max_index

    time_offset = lag / sr1

    return time_offset # time offset in seconds
