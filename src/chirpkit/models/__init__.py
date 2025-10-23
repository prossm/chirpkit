"""Model definitions for insect sound classification"""

from .cnn_lstm import CNNLSTMInsectClassifier
from .simple_cnn_lstm import SimpleCNNLSTMInsectClassifier
from .chirpkit_ensemble import ChirpKitEnsembleClassifier, DeepMLPClassifier

__all__ = [
    "CNNLSTMInsectClassifier",
    "SimpleCNNLSTMInsectClassifier",
    "ChirpKitEnsembleClassifier",
    "DeepMLPClassifier"
]