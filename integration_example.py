#!/usr/bin/env python3
"""
Example integration of ChirpKit with Wikipedia enrichment for soundcurious
"""

import asyncio
from src.chirpkit.classifier import InsectClassifier

async def classify_with_enrichment_example():
    """Example showing how soundcurious would integrate ChirpKit with Wikipedia data"""
    
    # Initialize classifier
    classifier = InsectClassifier()
    await classifier.initialize()
    
    # Example audio classification
    audio_file = "path/to/insect_sound.wav"
    
    # Method 1: Use the async API (recommended for web apps)
    result = await classifier.classify(audio_file, detailed=True)
    
    print("=== Enhanced Classification Result ===")
    print(f"Model: {result['model']}")
    
    # Species information (now enriched!)
    species = result['classification']['species']
    print(f"Scientific Name: {species['scientific_name']}")
    print(f"Common Name: {species['common_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    # Wikipedia enrichment data
    if result['enrichment']:
        enrichment = result['enrichment']
        print(f"Description: {enrichment['description']}")
        print(f"Image URL: {enrichment['image_url']}")
        print(f"Wikipedia URL: {enrichment['wikipedia_url']}")
        print(f"Data cached: {enrichment['cached']}")
    
    # Alternative predictions with enrichment
    if 'enriched_predictions' in result:
        print("\n=== Top 3 Alternatives (with Wikipedia data) ===")
        for pred in result['enriched_predictions'][:3]:
            species_info = pred['species_info']
            print(f"{pred['rank']}. {species_info['common_name']} ({species_info['scientific_name']}) - {pred['confidence']:.2%}")
            if species_info['image_url']:
                print(f"   Image: {species_info['image_url']}")
    
    return result

def synchronous_integration_example():
    """Example for soundcurious using synchronous methods"""
    
    # Initialize classifier (synchronous)
    classifier = InsectClassifier()
    success = classifier.load_model()
    if not success:
        raise RuntimeError("Failed to load ChirpKit model")
    
    # Classify audio (synchronous) - now with enrichment!
    result = classifier.predict_audio("path/to/insect_sound.wav", top_k=5)
    
    print("=== Synchronous Classification Result ===")
    if result['enriched']:
        # New enriched response
        print(f"Common Name: {result['species']}")
        print(f"Scientific Name: {result['scientific_name']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Description: {result['description']}")
        print(f"Image URL: {result['image_url']}")
        print(f"Wikipedia URL: {result['wikipedia_url']}")
    else:
        # Legacy response (if enrichment disabled)
        print(f"Species: {result['species']}")
        print(f"Scientific Name: {result['scientific_name']}")
        print(f"Confidence: {result['confidence']:.2%}")
    
    print("\nTop Predictions:")
    for species, conf in result['top_predictions'][:3]:
        print(f"  {species}: {conf:.2%}")
    
    return result

def api_endpoint_example():
    """Example API endpoint that soundcurious might create"""
    
    async def classify_insect_endpoint(audio_file_path: str):
        """FastAPI endpoint example"""
        try:
            classifier = InsectClassifier()
            await classifier.initialize()
            
            # Classify with full Wikipedia enrichment
            result = await classifier.classify(audio_file_path, detailed=True)
            
            # Return API response with all enrichment data
            return {
                "success": True,
                "model": result['model'],
                "classification": {
                    "scientific_name": result['classification']['species']['scientific_name'],
                    "common_name": result['classification']['species']['common_name'],
                    "confidence": result['confidence'],
                    "family": result['classification']['family'],
                    "is_insect": result['classification']['is_insect']
                },
                "enrichment": result['enrichment'],  # Wikipedia data
                "alternatives": [
                    {
                        "scientific_name": pred['species_info']['scientific_name'],
                        "common_name": pred['species_info']['common_name'],
                        "confidence": pred['confidence'],
                        "image_url": pred['species_info']['image_url'],
                        "wikipedia_url": pred['species_info']['wikipedia_url']
                    }
                    for pred in result.get('enriched_predictions', [])[:5]
                ],
                "metadata": {
                    "total_species_supported": result['features']['total_species'],
                    "wikipedia_integration": result['features']['wikipedia_integration'],
                    "cached_result": result['enrichment']['cached'] if result['enrichment'] else False
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": "ChirpKit-Error"
            }
    
    return classify_insect_endpoint

def disable_enrichment_example():
    """Example showing how to disable Wikipedia enrichment for faster responses"""
    
    async def fast_classification():
        classifier = InsectClassifier()
        
        # Disable Wikipedia enrichment for faster responses
        classifier.set_enrichment_enabled(False)
        
        await classifier.initialize()
        result = await classifier.classify("audio_file.wav", detailed=True)
        
        # Will only return basic species ID, no Wikipedia data
        print("Fast mode (no Wikipedia):", result['enrichment'])  # Will be None
        
        # Re-enable enrichment
        classifier.set_enrichment_enabled(True)
        result_enriched = await classifier.classify("audio_file.wav", detailed=True)
        
        # Now includes Wikipedia data
        print("Enriched mode:", result_enriched['enrichment'])
        
        return result_enriched
    
    return fast_classification

if __name__ == "__main__":
    # Run async example
    print("Running async example...")
    asyncio.run(classify_with_enrichment_example())
    
    print("\n" + "="*50 + "\n")
    
    # Run synchronous example  
    print("Running synchronous example...")
    synchronous_integration_example()