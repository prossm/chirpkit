"""ChirpKit: Insect Sound Classification Package

A simple interface for insect sound classification inference.

Usage:
    from chirpkit import InsectClassifier
    
    classifier = InsectClassifier()
    if classifier.load_model():
        result = classifier.predict_audio("insect_sound.wav")
        print(f"Species: {result['species']}")
        print(f"Confidence: {result['confidence']:.2%}")
"""

from .classifier import InsectClassifier

__version__ = "0.1.0"

# Only expose the main inference interface
__all__ = ["InsectClassifier"]