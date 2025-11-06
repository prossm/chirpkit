#!/usr/bin/env python3
"""
Simplified, robust UI for testing the insect classifier
Uses BirdNET v6.0 ensemble model (79.73% accuracy)
"""
import gradio as gr
import torch
import numpy as np
import librosa
import joblib
import os
import traceback
import requests
import json
from pathlib import Path
import urllib.parse
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.chirpkit._version import __version__
from src.chirpkit.models.chirpkit_ensemble import ChirpKitEnsembleClassifier
from src.chirpkit.transfer_learning.birdnet_embeddings import BirdNETEmbeddingExtractor

# Global variables for model components
ensemble_classifier = None
birdnet_extractor = None
label_encoder = None
device = None
deployment_mode = "ensemble"  # Options: 'single', 'ensemble', 'ensemble_tta'

# Species info cache
species_cache_file = Path("species_cache.json")
species_cache = {}

def load_species_cache():
    """Load species information cache"""
    global species_cache
    if species_cache_file.exists():
        try:
            with open(species_cache_file, 'r') as f:
                species_cache = json.load(f)
        except:
            species_cache = {}

def save_species_cache():
    """Save species information cache"""
    try:
        with open(species_cache_file, 'w') as f:
            json.dump(species_cache, f, indent=2)
    except:
        pass

def get_species_info(scientific_name):
    """Get species common name and image from Wikipedia"""
    if scientific_name in species_cache:
        return species_cache[scientific_name]
    
    # Format scientific name for Wikipedia search
    search_name = scientific_name.replace('_', ' ')
    
    try:
        # Search Wikipedia for the species with proper User-Agent header
        search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(search_name)}"
        headers = {
            'User-Agent': f'ChirpKit/{__version__} (https://github.com/patrickmetzger/chirpkit; contact@chirpkit.ai) Wikipedia Integration'
        }
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            common_name = data.get('title', search_name)
            description = data.get('extract', '')
            image_url = data.get('thumbnail', {}).get('source', '')
            
            # Try to extract common name from description
            if description and ',' in description:
                # Often format is "Common name, scientific description..."
                potential_common = description.split(',')[0].strip()
                if len(potential_common) < 50 and not potential_common.startswith('The'):
                    common_name = potential_common
            
            species_info = {
                'common_name': common_name,
                'description': description[:200] + '...' if len(description) > 200 else description,
                'image_url': image_url,
                'wikipedia_url': f"https://en.wikipedia.org/wiki/{urllib.parse.quote(search_name)}"
            }
        else:
            # Fallback if Wikipedia page not found
            species_info = {
                'common_name': search_name,
                'description': f'No Wikipedia information found for {search_name}',
                'image_url': '',
                'wikipedia_url': ''
            }
            
    except Exception as e:
        print(f"Error fetching info for {scientific_name}: {e}")
        species_info = {
            'common_name': search_name,
            'description': f'Error fetching information for {search_name}',
            'image_url': '',
            'wikipedia_url': ''
        }
    
    # Cache the result
    species_cache[scientific_name] = species_info
    save_species_cache()
    
    return species_info

def load_model():
    """Load the ChirpKit v6.0 ensemble model and preprocessing components"""
    global ensemble_classifier, birdnet_extractor, label_encoder, device

    try:
        print("=" * 80)
        print("🚀 Loading ChirpKit v6.0 Ensemble Model")
        print("=" * 80)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load species cache
        load_species_cache()

        # Initialize BirdNET embedding extractor
        print("\n📊 Initializing BirdNET embedding extractor...")
        birdnet_extractor = BirdNETEmbeddingExtractor()
        print("   ✓ BirdNET ready")

        # Load ensemble classifier
        print(f"\n🎯 Loading ChirpKit ensemble ({deployment_mode} mode)...")
        ensemble_classifier = ChirpKitEnsembleClassifier(
            model_dir="models/trained/chirpkit-ensemble",
            mode=deployment_mode,
            tta_rounds=10,
            tta_noise_std=0.01,
            device=device
        )
        ensemble_classifier.load_models()

        # Get label encoder from ensemble
        label_encoder = ensemble_classifier.label_encoder
        n_classes = len(label_encoder.classes_)

        print(f"\n" + "=" * 80)
        print(f"✅ ChirpKit v6.0 Ensemble Loaded Successfully!")
        print(f"=" * 80)
        print(f"   Species: {n_classes}")
        print(f"   Mode: {deployment_mode}")
        print(f"   Device: {device}")
        if deployment_mode == 'single':
            print(f"   Expected accuracy: ~77%")
        elif deployment_mode == 'ensemble':
            print(f"   Expected accuracy: ~79.6%")
        else:  # ensemble_tta
            print(f"   Expected accuracy: ~79.7%")
        print("=" * 80 + "\n")

        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return False

def predict_species(audio_file):
    """Predict insect species from audio file using ChirpKit v6.0 ensemble"""
    if ensemble_classifier is None or label_encoder is None:
        return "❌ Model not loaded. Please restart the application."

    if audio_file is None:
        return "❌ Please upload an audio file first."

    try:
        print(f"Processing audio file: {audio_file}")

        # Extract BirdNET embedding from audio
        print("Extracting BirdNET embedding...")
        embedding = birdnet_extractor.extract_embeddings_from_audio(
            audio_file,
            aggregate='mean'
        )
        print(f"Embedding shape: {embedding.shape}")

        # Make prediction with ensemble
        print(f"Running ensemble prediction ({deployment_mode} mode)...")
        result = ensemble_classifier.predict(embedding, top_k=5)

        top_pred = result['top_prediction']
        predictions = result['predictions']

        # Get enriched species info
        predicted_species = top_pred['species']
        species_info = get_species_info(predicted_species)
        confidence = top_pred['confidence']

        # Format result with enhanced info and context
        output = f"🦗 **Predicted Species:** {species_info['common_name']}\n"
        output += f"🔬 **Scientific Name:** {top_pred['scientific_name']}\n"
        output += f"🎯 **Confidence:** {confidence:.2%}"

        # Add context for confidence interpretation (adjusted for 231 classes)
        if confidence > 0.10:  # 10%
            output += " (Very High) ⭐⭐⭐\n\n"
        elif confidence > 0.05:  # 5%
            output += " (High) ⭐⭐☆\n\n"
        elif confidence > 0.02:  # 2%
            output += " (Moderate) ⭐☆☆\n\n"
        else:
            output += " (Low - verify with expert) ☆☆☆\n\n"

        # Add model info
        output += f"🤖 **Model:** ChirpKit v6.0 ({result['mode']} mode, {result['num_models']} models"
        if result['tta_rounds'] > 0:
            output += f", {result['tta_rounds']} TTA rounds"
        output += ")\n\n"

        if species_info['description']:
            output += f"📖 **Description:** {species_info['description']}\n\n"

        if species_info['wikipedia_url']:
            output += f"🔗 **More Info:** [Wikipedia]({species_info['wikipedia_url']})\n\n"

        output += "📊 **Top 5 Predictions:**\n"

        for pred in predictions:
            pred_info = get_species_info(pred['species'])
            output += f"{pred['rank']}. {pred_info['common_name']} ({pred['scientific_name']}): {pred['confidence']:.2%}\n"

        print(f"Prediction completed: {predicted_species} ({confidence:.2%})")
        return output

    except Exception as e:
        error_msg = f"❌ Error processing audio: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg

def predict_and_display(audio_file):
    """Predict species and return both text results and image"""
    if ensemble_classifier is None or label_encoder is None:
        return "❌ Model not loaded. Please restart the application.", None

    if audio_file is None:
        return "❌ Please upload an audio file first.", None

    try:
        # Get prediction results
        result_text = predict_species(audio_file)

        # Extract the predicted species to get image info
        embedding = birdnet_extractor.extract_embeddings_from_audio(
            audio_file,
            aggregate='mean'
        )
        result = ensemble_classifier.predict(embedding, top_k=1)
        predicted_species = result['top_prediction']['species']
        species_info = get_species_info(predicted_species)

        # Download image with proper headers to avoid 403 errors
        image_url = species_info.get('image_url', '')

        if image_url:
            try:
                # Download image with proper User-Agent header
                headers = {
                    'User-Agent': f'ChirpKit/{__version__} (https://github.com/patrickmetzger/chirpkit; contact@chirpkit.ai) Wikipedia Integration'
                }
                response = requests.get(image_url, headers=headers, timeout=10, stream=True)

                if response.status_code == 200:
                    # Save to temporary file
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            tmp_file.write(chunk)
                        temp_image_path = tmp_file.name
                    return result_text, temp_image_path
                else:
                    print(f"⚠️  Failed to download image: HTTP {response.status_code}")
                    return result_text, None
            except Exception as img_error:
                print(f"⚠️  Error downloading image: {img_error}")
                return result_text, None

        return result_text, None

    except Exception as e:
        error_msg = f"❌ Error processing audio: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg, None

def search_species(search_term, all_species_info):
    """Filter species based on search term"""
    if not search_term.strip():
        return all_species_info
    
    search_term = search_term.lower().strip()
    filtered_results = []
    
    for species_info in all_species_info:
        # Search in both scientific and common names
        scientific = species_info.split('\n')[0].lower()
        common = species_info.split('\n')[1].lower() if '\n' in species_info else ""
        
        if search_term in scientific or search_term in common:
            filtered_results.append(species_info)
    
    if not filtered_results:
        return ["No species found matching your search."]
    
    return filtered_results

def create_interface():
    """Create the Gradio interface"""
    with gr.Blocks(title="🦗 ChirpKit - Insect Sound Classifier") as interface:
        gr.Markdown("# 🦗 ChirpKit Insect Sound Classifier")
        gr.Markdown("### v6.0 Ensemble Model (79.7% accuracy)")
        gr.Markdown("Record insect sounds live or upload audio files (.wav, .mp3) to identify species!")

        # Model status with clickable species count
        if label_encoder:
            with gr.Row():
                mode_label = {
                    'single': 'Single Model (~77%)',
                    'ensemble': 'Ensemble (~79.6%)',
                    'ensemble_tta': 'Ensemble + TTA (~79.7%)'
                }[deployment_mode]
                gr.Markdown(f"**Model:** ✅ ChirpKit v6.0 - {mode_label}")
                species_btn = gr.Button(
                    f"📋 {len(label_encoder.classes_)} species",
                    variant="secondary",
                    size="sm"
                )
        else:
            gr.Markdown("**Model Status:** ❌ Not loaded")
            species_btn = None  # Define species_btn for consistency
        
        with gr.Row():
            with gr.Column():
                # Audio input with recording capability
                audio_input = gr.Audio(
                    label="🎤 Record Audio or Upload File",
                    type="filepath",
                    sources=["microphone", "upload"]  # Enable both recording and upload
                )
                
                with gr.Row():
                    # Predict button
                    predict_btn = gr.Button("🔍 Identify Species", variant="primary", size="lg")
                    
                # Instructions
                gr.Markdown("""
                **🎤 Recording Tips:**
                - **Get close**: Position your device 1-3 feet from the insect
                - **Stay quiet**: Minimize background noise and movement
                - **Duration**: Record 2-5 seconds of clear sound
                - **Timing**: Many insects are most active at dawn/dusk
                - **Environment**: Outdoor recordings often work better than indoor
                
                **📁 Upload Tips:**
                - Supported formats: .wav, .mp3, .m4a, .flac
                - Best quality: Uncompressed formats like .wav
                - Length: 2-10 seconds optimal
                """)
            
            with gr.Column():
                # Results
                result_output = gr.Textbox(
                    label="🎯 Prediction Results",
                    lines=10,
                    max_lines=15,
                    placeholder="Upload an audio file and click 'Identify Species' to see results..."
                )
                
                # Species image
                species_image = gr.Image(
                    label="Species Photo",
                    show_label=True,
                    height=300
                )
        
        # Species modal (initially hidden)
        if label_encoder:
            # Quick species data - just format scientific names, no API calls
            all_species_data = []
            for i, species in enumerate(label_encoder.classes_):
                scientific_name = species.replace('_', ' ')
                # Use basic formatting - common names will be fetched only when needed for predictions
                formatted_info = f"{scientific_name}\n(Common name will be shown when identified)"
                all_species_data.append(formatted_info)
            
            # Modal components
            with gr.Column(visible=False) as species_modal:
                gr.Markdown("## 🔍 Species Browser")
                
                # Fixed search bar at top
                with gr.Row():
                    search_box = gr.Textbox(
                        label="Search Species",
                        placeholder="Enter scientific or common name...",
                        scale=4
                    )
                    search_btn = gr.Button("🔍 Search", scale=1)
                    close_btn = gr.Button("✖️ Close", variant="secondary", scale=1)
                
                # Scrollable species list
                species_list_text = "\n".join([f"{i+1:3d}. {species.replace('_', ' ')}" 
                                              for i, species in enumerate(label_encoder.classes_)])
                species_display = gr.Textbox(
                    value=species_list_text,
                    label=f"All {len(label_encoder.classes_)} Species",
                    lines=20,
                    max_lines=20,
                    interactive=False
                )
        
        # Connect the prediction function
        predict_btn.click(
            fn=predict_and_display,
            inputs=[audio_input],
            outputs=[result_output, species_image]
        )
        
        # Modal functionality
        if label_encoder:
            def show_modal():
                return gr.update(visible=True)
            
            def hide_modal():
                return gr.update(visible=False)
            
            def search_and_update(search_term):
                if not search_term.strip():
                    # Show all species
                    filtered_text = "\n".join([f"{i+1:3d}. {species.replace('_', ' ')}" 
                                             for i, species in enumerate(label_encoder.classes_)])
                else:
                    # Filter species by scientific name
                    search_term = search_term.lower().strip()
                    filtered_species = []
                    for i, species in enumerate(label_encoder.classes_):
                        scientific = species.replace('_', ' ').lower()
                        if search_term in scientific:
                            filtered_species.append(f"{i+1:3d}. {species.replace('_', ' ')}")
                    
                    if not filtered_species:
                        filtered_text = "No species found matching your search."
                    else:
                        filtered_text = "\n".join(filtered_species)
                
                return gr.update(value=filtered_text)
            
            # Connect modal events
            species_btn.click(show_modal, outputs=species_modal)
            close_btn.click(hide_modal, outputs=species_modal)
            search_btn.click(search_and_update, inputs=search_box, outputs=species_display)
            search_box.submit(search_and_update, inputs=search_box, outputs=species_display)
    
    return interface

def main():
    """Launch the app"""
    print("🚀 Starting Insect Classifier Web App...")
    
    # Load model
    if not load_model():
        print("❌ Cannot start app - model failed to load")
        return
    
    # Create interface
    interface = create_interface()
    
    print("📱 Access the app at: http://localhost:7860")
    print("🎵 Upload .wav or .mp3 files to test the classifier!")
    
    # Launch with error handling
    try:
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True,
            inbrowser=False  # Don't auto-open browser
        )
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()