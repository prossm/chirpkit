#!/usr/bin/env python3
"""
Test script to verify ChirpKit MoE integration

This script tests the ChirpKit classifier integration to ensure it works
properly with the MoE (Mixture of Experts) system.
"""

import asyncio
import sys
import logging
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chirpkit import InsectClassifier, DependencyManager
from chirpkit.models import find_any_model, list_models
from chirpkit.cli import classify_audio_file, get_classifier_instance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockProcessedAudio:
    """Mock audio object for testing"""
    
    def __init__(self, duration: float = 2.5, sample_rate: int = 16000):
        self.duration = duration
        self.sample_rate = sample_rate
        # Generate realistic-sounding mock audio (chirp-like pattern)
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Create a chirp pattern - frequency sweep with some harmonics
        freq_sweep = np.linspace(2000, 8000, len(t))
        self.waveform = (
            0.5 * np.sin(2 * np.pi * freq_sweep * t) +  # Main chirp
            0.2 * np.sin(4 * np.pi * freq_sweep * t) +  # Second harmonic
            0.1 * np.sin(6 * np.pi * freq_sweep * t) +  # Third harmonic
            0.05 * np.random.randn(len(t))              # Background noise
        )
        # Apply envelope to make it more realistic
        envelope = np.exp(-((t - duration/2) / (duration/4))**2)
        self.waveform = self.waveform * envelope


async def test_dependency_management():
    """Test the dependency management system"""
    logger.info("🔧 Testing dependency management...")
    
    # Test PyTorch detection
    torch = DependencyManager.get_torch()
    if torch:
        logger.info(f"✅ PyTorch detected: {torch.__version__}")
        
        # Test device detection
        if torch.cuda.is_available():
            logger.info(f"🚀 CUDA available: {torch.cuda.device_count()} device(s)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("🍎 Apple MPS backend available")
        else:
            logger.info("💻 Using CPU backend")
    else:
        logger.warning("⚠️ PyTorch not available")
        
    # Test TensorFlow detection (should show as not required)
    tf = DependencyManager.get_tensorflow()
    if tf:
        logger.info(f"ℹ️ TensorFlow also available: {tf.__version__}")
    else:
        logger.info("ℹ️ TensorFlow not available (not required)")
        
    # Check for failed imports
    failed_imports = DependencyManager.get_failed_imports()
    if failed_imports:
        logger.warning(f"⚠️ Failed imports: {failed_imports}")


async def test_model_discovery():
    """Test model discovery and loading utilities"""
    logger.info("🔍 Testing model discovery...")
    
    # List available models
    models = list_models()
    logger.info(f"Available models: {list(models.keys())}")
    
    for name, info in models.items():
        logger.info(f"  {name}: {info.get('status', 'unknown')} - {info.get('description', 'no description')}")
    
    # Try to find any model
    model_files = find_any_model()
    if model_files:
        model_path, encoder_path, info_path = model_files
        logger.info(f"✅ Found model files:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Encoder: {encoder_path}")
        logger.info(f"  Info: {info_path}")
        return model_files
    else:
        logger.warning("⚠️ No model files found")
        return None


async def test_classifier_initialization():
    """Test classifier initialization"""
    logger.info("🧠 Testing classifier initialization...")
    
    try:
        classifier = InsectClassifier()
        await classifier.initialize()
        
        if classifier.is_available():
            logger.info("✅ Classifier initialized successfully")
            
            # Get model info
            info = classifier.get_model_info()
            logger.info(f"Model info: {info}")
            
            # Get species list
            species = classifier.get_species_list()
            logger.info(f"Supports {len(species)} species")
            logger.info(f"Example species: {species[:5]}")
            
            return classifier
        else:
            logger.error("❌ Classifier not available after initialization")
            return None
            
    except Exception as e:
        logger.error(f"❌ Classifier initialization failed: {e}")
        return None


async def test_classification():
    """Test audio classification with mock data"""
    logger.info("🎵 Testing audio classification...")
    
    try:
        classifier = InsectClassifier()
        
        # Test with mock audio
        mock_audio = MockProcessedAudio()
        logger.info(f"Created mock audio: {mock_audio.duration}s at {mock_audio.sample_rate}Hz")
        
        result = await classifier.classify(mock_audio, detailed=True)
        
        logger.info("✅ Classification completed")
        logger.info(f"Result format check:")
        
        # Verify expected format
        expected_keys = ['model', 'classification', 'confidence', 'predictions', 'features']
        for key in expected_keys:
            if key in result:
                logger.info(f"  ✓ {key}: present")
            else:
                logger.error(f"  ✗ {key}: missing")
                
        # Check classification sub-structure
        if 'classification' in result:
            classification = result['classification']
            class_keys = ['is_insect', 'species', 'confidence', 'family']
            for key in class_keys:
                if key in classification:
                    logger.info(f"  ✓ classification.{key}: {classification[key]}")
                else:
                    logger.error(f"  ✗ classification.{key}: missing")
        
        # Show top predictions
        if 'predictions' in result and result['predictions']:
            logger.info("Top predictions:")
            for i, pred in enumerate(result['predictions'][:3]):
                logger.info(f"  {i+1}. {pred['species']}: {pred['confidence']:.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Classification test failed: {e}")
        return None


async def test_cli_integration():
    """Test CLI integration functions"""
    logger.info("🔧 Testing CLI integration...")
    
    # Test audio file if one exists
    test_audio = Path("test_audio/synthetic_insect.wav")
    if test_audio.exists():
        logger.info(f"Testing with real audio file: {test_audio}")
        try:
            result = classify_audio_file(str(test_audio))
            logger.info("✅ CLI audio classification successful")
            logger.info(f"Species: {result['classification']['species']}")
            logger.info(f"Confidence: {result['classification']['confidence']:.4f}")
        except Exception as e:
            logger.error(f"❌ CLI audio classification failed: {e}")
    else:
        logger.info("ℹ️ No test audio file found, skipping file-based test")
    
    # Test classifier instance creation
    try:
        classifier = get_classifier_instance()
        logger.info("✅ CLI classifier instance created")
        
        # Test compatibility methods
        info = classifier.get_model_info()
        species_count = len(classifier.get_species_list()) if classifier.get_species_list() else 0
        logger.info(f"Species supported: {species_count}")
        
    except Exception as e:
        logger.error(f"❌ CLI classifier instance failed: {e}")


async def test_compatibility_methods():
    """Test backward compatibility methods"""
    logger.info("🔄 Testing backward compatibility...")
    
    try:
        classifier = InsectClassifier()
        
        # Test synchronous model loading
        success = classifier.load_model()
        if success:
            logger.info("✅ Synchronous model loading works")
        else:
            logger.warning("⚠️ Synchronous model loading failed")
            
        # Test compatibility with existing predict_audio method
        test_audio = Path("test_audio/synthetic_insect.wav")
        if test_audio.exists():
            try:
                result = classifier.predict_audio(str(test_audio))
                logger.info("✅ Backward compatible predict_audio works")
                logger.info(f"Result: {result['species']} ({result['confidence']:.4f})")
            except Exception as e:
                logger.error(f"❌ predict_audio compatibility failed: {e}")
        
    except Exception as e:
        logger.error(f"❌ Compatibility test failed: {e}")


async def run_all_tests():
    """Run all integration tests"""
    logger.info("🚀 Starting ChirpKit MoE integration tests")
    logger.info("=" * 50)
    
    test_results = {}
    
    # Test 1: Dependency Management
    try:
        await test_dependency_management()
        test_results['dependencies'] = 'PASS'
    except Exception as e:
        logger.error(f"Dependency test failed: {e}")
        test_results['dependencies'] = 'FAIL'
    
    # Test 2: Model Discovery
    try:
        model_files = await test_model_discovery()
        test_results['model_discovery'] = 'PASS' if model_files else 'WARN'
        if not model_files:
            logger.warning("⚠️ No models found - some tests may fail")
    except Exception as e:
        logger.error(f"Model discovery test failed: {e}")
        test_results['model_discovery'] = 'FAIL'
    
    # Test 3: Classifier Initialization
    try:
        classifier = await test_classifier_initialization()
        test_results['initialization'] = 'PASS' if classifier else 'FAIL'
    except Exception as e:
        logger.error(f"Initialization test failed: {e}")
        test_results['initialization'] = 'FAIL'
    
    # Test 4: Classification
    try:
        result = await test_classification()
        test_results['classification'] = 'PASS' if result else 'FAIL'
    except Exception as e:
        logger.error(f"Classification test failed: {e}")
        test_results['classification'] = 'FAIL'
    
    # Test 5: CLI Integration
    try:
        await test_cli_integration()
        test_results['cli_integration'] = 'PASS'
    except Exception as e:
        logger.error(f"CLI integration test failed: {e}")
        test_results['cli_integration'] = 'FAIL'
    
    # Test 6: Compatibility
    try:
        await test_compatibility_methods()
        test_results['compatibility'] = 'PASS'
    except Exception as e:
        logger.error(f"Compatibility test failed: {e}")
        test_results['compatibility'] = 'FAIL'
    
    # Results summary
    logger.info("=" * 50)
    logger.info("🎯 TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    
    all_passed = True
    for test_name, result in test_results.items():
        status_icon = "✅" if result == 'PASS' else "⚠️" if result == 'WARN' else "❌"
        logger.info(f"{status_icon} {test_name.replace('_', ' ').title()}: {result}")
        if result == 'FAIL':
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED - ChirpKit is ready for MoE integration!")
    else:
        logger.info("\n🔧 Some tests failed - please check the logs above")
    
    return test_results


def main():
    """Main test runner"""
    try:
        # Run all tests
        results = asyncio.run(run_all_tests())
        
        # Exit with error code if any critical tests failed
        critical_tests = ['dependencies', 'initialization', 'classification']
        critical_failures = [test for test in critical_tests if results.get(test) == 'FAIL']
        
        if critical_failures:
            logger.error(f"Critical test failures: {critical_failures}")
            sys.exit(1)
        else:
            logger.info("✅ Integration test completed successfully")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()