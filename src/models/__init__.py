"""Model definitions for insect sound classification"""

from .cnn_lstm import CNNLSTMInsectClassifier
from .simple_cnn_lstm import SimpleCNNLSTMInsectClassifier

__all__ = ["CNNLSTMInsectClassifier", "SimpleCNNLSTMInsectClassifier"]