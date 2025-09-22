import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Dict

class InsectAudioPreprocessor:
    """Preprocessor for insect sound data"""
    def __init__(self, target_sr: int = 16000, duration: float = 2.5, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128):
        self.target_sr = target_sr
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def load_and_preprocess(self, audio_path: Path) -> Dict:
        audio, sr = librosa.load(audio_path, sr=self.target_sr)
        target_length = int(self.target_sr * self.duration)
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        features = self.extract_features(audio)
        return {
            'waveform': audio,
            'spectrogram': features['spectrogram'],
            'mfcc': features['mfcc'],
            'chroma': features['chroma'],
            'spectral_centroid': features['spectral_centroid'],
            'zero_crossing_rate': features['zcr']
        }

    def extract_features(self, audio: np.ndarray) -> Dict:
        spec = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        spec_db = librosa.amplitude_to_db(np.abs(spec))
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.target_sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)
        mel_db = librosa.power_to_db(mel_spec)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.target_sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.target_sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.target_sr)
        zcr = librosa.feature.zero_crossing_rate(audio)
        return {
            'spectrogram': mel_db,
            'mfcc': mfcc,
            'chroma': chroma,
            'spectral_centroid': spectral_centroid,
            'zcr': zcr
        }
