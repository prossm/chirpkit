#!/usr/bin/env python3
"""
Test script to verify ChirpKit integration fixes
"""

import asyncio
import numpy as np
import logging
from pathlib import Path
import sys

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_chirpkit_integration():
    """Test the ChirpKit integration with the exact test case from the context"""
    
    print("🧪 Testing ChirpKit Integration Fixes")
    print("=" * 50)
    
    try:
        # Import ChirpKit
        print("1. Testing imports...")
        from src.chirpkit import InsectClassifier, find_any_model
        print("✅ Import successful")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    async def run_async_test():
        try:
            print("\n2. Testing classifier initialization...")
            classifier = InsectClassifier()
            await classifier.initialize()  # Should succeed
            print("✅ Initialization successful")
            
            print("\n3. Testing classifier availability...")
            assert classifier.is_available() == True
            print("✅ Classifier reports as available")
            
            print("\n4. Testing classification with mock audio...")
            # Mock processed audio
            class MockAudio:
                def __init__(self):
                    self.waveform = np.random.randn(16000)
                    self.sample_rate = 16000
                    self.duration = 1.0
            
            mock_audio = MockAudio()
            result = await classifier.classify(mock_audio, detailed=True)
            
            print("✅ Classification completed")
            
            print("\n5. Validating result format...")
            # Check required fields
            required_fields = ['classification', 'model', 'confidence', 'predictions', 'features']
            for field in required_fields:
                assert field in result, f"Missing field: {field}"
            
            # Check classification sub-fields
            classification = result['classification']
            required_classification_fields = ['is_insect', 'species', 'confidence', 'family']
            for field in required_classification_fields:
                assert field in classification, f"Missing classification field: {field}"
            
            # Check predictions format
            predictions = result['predictions']
            assert isinstance(predictions, list), "Predictions should be a list"
            assert len(predictions) > 0, "Should have at least one prediction"
            
            for pred in predictions:
                assert 'species' in pred, "Prediction missing species"
                assert 'confidence' in pred, "Prediction missing confidence"
                assert 'rank' in pred, "Prediction missing rank"
            
            print("✅ Result format validation passed")
            
            print("\n📊 Result Summary:")
            print(f"   Model: {result['model']}")
            print(f"   Top Species: {classification['species']}")
            print(f"   Confidence: {classification['confidence']:.3f}")
            print(f"   Is Insect: {classification['is_insect']}")
            print(f"   Family: {classification['family']}")
            print(f"   Total Predictions: {len(predictions)}")
            print(f"   Features: {list(result['features'].keys())}")
            
            print("\n6. Testing model discovery...")
            models = find_any_model()
            if models:
                model_path, encoder_path, info_path = models
                print(f"✅ Found models at: {model_path}")
            else:
                print("⚠️  No trained models found (expected for fresh install)")
            
            print("\n7. Testing cleanup...")
            classifier.cleanup()
            print("✅ Cleanup successful")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Run the async test
    return asyncio.run(run_async_test())

def test_error_conditions():
    """Test error handling and fallback conditions"""
    
    print("\n🔍 Testing Error Conditions")
    print("=" * 50)
    
    async def run_error_tests():
        try:
            # Test with non-existent model path
            print("1. Testing with non-existent model path...")
            from src.chirpkit import InsectClassifier
            
            classifier = InsectClassifier(model_path="/non/existent/path.pth")
            await classifier.initialize()
            
            # Should still work with fallback
            assert classifier.is_available() == True
            print("✅ Fallback initialization successful")
            
            return True
            
        except Exception as e:
            print(f"❌ Error test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return asyncio.run(run_error_tests())

def main():
    """Main test runner"""
    print("🚀 ChirpKit Integration Test Suite")
    print("=" * 60)
    
    # Test basic integration
    integration_success = test_chirpkit_integration()
    
    # Test error conditions
    error_handling_success = test_error_conditions()
    
    print("\n" + "=" * 60)
    print("📋 Test Results Summary:")
    print(f"   Integration Test: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    print(f"   Error Handling Test: {'✅ PASSED' if error_handling_success else '❌ FAILED'}")
    
    overall_success = integration_success and error_handling_success
    
    if overall_success:
        print("\n🎉 All tests passed! ChirpKit integration fixes are working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())