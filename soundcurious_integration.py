#!/usr/bin/env python3
"""
Simple integration example for soundcurious - showing the easiest way to use ChirpKit
with full Wikipedia enrichment
"""

from src.chirpkit.classifier import InsectClassifier

def simple_soundcurious_integration():
    """The simplest way for soundcurious to integrate ChirpKit with Wikipedia data"""
    
    # Step 1: Initialize classifier
    classifier = InsectClassifier()
    
    # Step 2: Load model (handles initialization automatically)
    if not classifier.load_model():
        raise RuntimeError("Failed to load ChirpKit model")
    
    # Step 3: Classify audio file - NOW WITH WIKIPEDIA DATA!
    audio_file = "path/to/insect_recording.wav"
    
    try:
        result = classifier.predict_audio(audio_file, top_k=5)
        
        # The result now automatically includes Wikipedia enrichment!
        if result['enriched']:
            print("=== ENRICHED RESULT (with Wikipedia) ===")
            print(f"Common Name: {result['species']}")
            print(f"Scientific Name: {result['scientific_name']}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Description: {result['description']}")
            print(f"Image: {result['image_url']}")
            print(f"Wikipedia: {result['wikipedia_url']}")
        else:
            print("=== BASIC RESULT (Wikipedia disabled) ===")
            print(f"Species: {result['species']}")
            print(f"Scientific Name: {result['scientific_name']}")
            print(f"Confidence: {result['confidence']:.1%}")
        
        print("\nAlternatives:")
        for i, (species, confidence) in enumerate(result['top_predictions'][:3]):
            print(f"{i+1}. {species}: {confidence:.1%}")
            
        return result
        
    except Exception as e:
        print(f"Classification error: {e}")
        return None

def soundcurious_api_response_example():
    """Example showing what soundcurious API would return to users"""
    
    classifier = InsectClassifier()
    if not classifier.load_model():
        return {"error": "Model not available"}
    
    # Sample audio classification
    try:
        result = classifier.predict_audio("sample_cricket.wav")
        
        if result['enriched']:
            # Return enriched API response
            return {
                "success": True,
                "identification": {
                    "common_name": result['species'],
                    "scientific_name": result['scientific_name'], 
                    "confidence": result['confidence'],
                    "confidence_level": "high" if result['confidence'] > 0.15 else "medium" if result['confidence'] > 0.08 else "low"
                },
                "information": {
                    "description": result['description'],
                    "image_url": result['image_url'],
                    "learn_more_url": result['wikipedia_url']
                },
                "alternatives": [
                    {
                        "species": species,
                        "confidence": conf
                    }
                    for species, conf in result['top_predictions'][:3]
                ],
                "metadata": {
                    "model": "ChirpKit v1.0",
                    "total_species": "471 species supported",
                    "data_source": "Wikipedia integration"
                }
            }
        else:
            # Basic response without enrichment
            return {
                "success": True,
                "identification": {
                    "species": result['species'],
                    "scientific_name": result['scientific_name'],
                    "confidence": result['confidence']
                },
                "alternatives": result['top_predictions'][:3]
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def performance_control_example():
    """Example showing how to control performance vs enrichment"""
    
    classifier = InsectClassifier()
    classifier.load_model()
    
    # Fast mode: Disable Wikipedia for quick responses
    print("=== FAST MODE (No Wikipedia lookups) ===")
    classifier.set_enrichment_enabled(False)
    
    import time
    start = time.time()
    result_fast = classifier.predict_audio("test_audio.wav")
    fast_time = time.time() - start
    
    print(f"Fast mode time: {fast_time:.2f}s")
    print(f"Enriched: {result_fast['enriched']}")
    
    # Full mode: Enable Wikipedia for rich responses  
    print("\n=== FULL MODE (With Wikipedia enrichment) ===")
    classifier.set_enrichment_enabled(True)
    
    start = time.time()
    result_enriched = classifier.predict_audio("test_audio.wav")
    enriched_time = time.time() - start
    
    print(f"Enriched mode time: {enriched_time:.2f}s")
    print(f"Enriched: {result_enriched['enriched']}")
    
    if result_enriched['enriched']:
        print(f"Got Wikipedia data: {result_enriched['description'][:50]}...")
    
    return result_fast, result_enriched

if __name__ == "__main__":
    print("🦗 ChirpKit Integration Example for soundcurious")
    print("=" * 50)
    
    # Run the simple integration
    try:
        result = simple_soundcurious_integration()
        if result:
            print("✅ Integration successful!")
        else:
            print("❌ Integration failed")
    except Exception as e:
        print(f"❌ Integration error: {e}")
        
    print("\n" + "=" * 50)
    print("🌐 API Response Example:")
    api_response = soundcurious_api_response_example()
    print(api_response)