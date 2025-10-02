import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Dict

class InsectAudioPreprocessor:
    """Enhanced preprocessor for insect sound data with richer audio features"""
    def __init__(self, target_sr: int = 22050, duration: float = 2.5, n_fft: int = 4096, hop_length: int = 256, n_mels: int = 256, n_mfcc: int = 40, use_enhanced: bool = True):
        """
        Enhanced audio preprocessor for insect classification

        Args:
            target_sr: Sample rate (22050 Hz for better frequency resolution - insect calls often have high-freq components)
            duration: Audio duration in seconds
            n_fft: FFT window size (4096 for better frequency resolution)
            hop_length: Hop length for STFT (256 for better temporal resolution)
            n_mels: Number of mel bands (256 for richer representation)
            n_mfcc: Number of MFCC coefficients (40 for detailed timbre)
            use_enhanced: Use enhanced features (delta features, harmonic-percussive separation)
        """
        self.target_sr = target_sr
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.use_enhanced = use_enhanced

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
        """
        Extract rich audio features for insect classification

        Enhanced features for 80% target accuracy:
        - 256 mel bins (was 128) - more frequency detail
        - 40 MFCCs (was 13) - richer timbre representation
        - Delta and delta-delta MFCCs - capture temporal dynamics
        - Higher sample rate (22050 Hz) - capture high-frequency insect calls
        - Larger FFT window (4096) - better frequency resolution
        """
        # Enhanced mel spectrogram: 256 mel bins for richer representation
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.target_sr,
            n_mels=self.n_mels,  # 256 bins
            n_fft=self.n_fft,    # 4096 for better freq resolution
            hop_length=self.hop_length,  # 256 for better temporal resolution
            fmin=20,   # Insects can produce very low frequencies
            fmax=self.target_sr // 2  # Up to Nyquist frequency
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        if self.use_enhanced:
            # Enhanced MFCCs: 40 coefficients for detailed timbre
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.target_sr,
                n_mfcc=self.n_mfcc,  # 40 coefficients
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )

            # Delta features capture temporal dynamics (important for insect calls)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

            # Stack: [MFCC (40) + Delta (40) + Delta-Delta (40)] = 120 total features
            # But we'll return mel_db as primary feature for CNN (visual representation)
            # and can optionally use MFCC stack for additional branches
            mfcc_enhanced = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        else:
            # Basic MFCC for backward compatibility
            mfcc = librosa.feature.mfcc(y=audio, sr=self.target_sr, n_mfcc=13)
            mfcc_enhanced = mfcc

        # Additional features (kept for compatibility, but mel_db is primary)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.target_sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.target_sr)
        zcr = librosa.feature.zero_crossing_rate(audio)

        return {
            'spectrogram': mel_db,  # [256, time] - PRIMARY FEATURE
            'mfcc': mfcc_enhanced,  # [120, time] with deltas or [13, time] basic
            'chroma': chroma,
            'spectral_centroid': spectral_centroid,
            'zcr': zcr
        }
