#!/usr/bin/env python3
"""
Simplified, robust UI for testing the insect classifier
"""
import gradio as gr
import torch
import numpy as np
import librosa
import joblib
from src.models.simple_cnn_lstm import SimpleCNNLSTMInsectClassifier
import os
import traceback
import requests
import json
from pathlib import Path
import urllib.parse

# Global variables for model components
model = None
label_encoder = None
device = None

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
        # Search Wikipedia for the species
        search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(search_name)}"
        response = requests.get(search_url, timeout=5)
        
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
    """Load the trained model and preprocessing components"""
    global model, label_encoder, device
    
    try:
        print("Loading model components...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load species cache
        load_species_cache()
        
        # Load label encoder - automatically find the latest species count model
        import glob
        label_encoder_files = glob.glob('models/trained/*_label_encoder.joblib')
        if not label_encoder_files:
            print("No label encoder found")
            return False
        
        # Use the one with most species (highest number)
        latest_encoder = max(label_encoder_files, key=lambda x: int(x.split('_')[-3].replace('species', '')))
        print(f"Using label encoder: {latest_encoder}")
        
        label_encoder = joblib.load(latest_encoder)
        n_classes = len(label_encoder.classes_)
        print(f"Label encoder loaded: {n_classes} classes")
        
        # Load model - find corresponding model file
        model_name = latest_encoder.replace('_label_encoder.joblib', '.pth').split('/')[-1]
        model_path = f'models/trained/{model_name}'
        print(f"Looking for model: {model_path}")
        
        model = SimpleCNNLSTMInsectClassifier(n_classes=n_classes)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            print(f"Model loaded successfully on {device}")
            return True
        else:
            print(f"Model file not found: {model_path}")
            return False
            
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return False

def predict_species(audio_file):
    """Predict insect species from audio file"""
    if model is None or label_encoder is None:
        return "❌ Model not loaded. Please restart the application."
    
    if audio_file is None:
        return "❌ Please upload an audio file first."
    
    try:
        print(f"Processing audio file: {audio_file}")
        
        # Load audio file
        audio, sr = librosa.load(audio_file, sr=16000)
        print(f"Audio loaded: shape={audio.shape}, sr={sr}")
        
        # Ensure it's the right length (2.5 seconds)
        target_length = int(16000 * 2.5)
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        # Extract features (mel spectrogram)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=16000, 
            n_mels=128, 
            n_fft=2048, 
            hop_length=512
        )
        mel_db = librosa.power_to_db(mel_spec)
        print(f"Mel spectrogram shape: {mel_db.shape}")
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(device)
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        # Get species name and info
        predicted_species = label_encoder.classes_[predicted_idx]
        species_info = get_species_info(predicted_species)
        
        # Create confidence scores for top 5
        probs = probabilities[0].cpu().numpy()
        species_probs = [(label_encoder.classes_[i], float(probs[i])) for i in range(len(label_encoder.classes_))]
        species_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Format result with enhanced info and context
        result = f"🦗 **Predicted Species:** {species_info['common_name']}\n"
        result += f"🔬 **Scientific Name:** {predicted_species.replace('_', ' ')}\n"
        result += f"🎯 **Confidence:** {confidence:.2%}"
        
        # Add context for confidence interpretation
        if confidence > 0.15:  # 15%
            result += " (Very High) ⭐⭐⭐\n\n"
        elif confidence > 0.08:  # 8%
            result += " (High) ⭐⭐☆\n\n"
        elif confidence > 0.03:  # 3%
            result += " (Moderate) ⭐☆☆\n\n"
        else:
            result += " (Low - verify with expert) ☆☆☆\n\n"
        
        if species_info['description']:
            result += f"📖 **Description:** {species_info['description']}\n\n"
        
        if species_info['wikipedia_url']:
            result += f"🔗 **More Info:** [Wikipedia]({species_info['wikipedia_url']})\n\n"
        
        result += "📊 **Top 5 Predictions:**\n"
        
        for i, (species, conf) in enumerate(species_probs[:5]):
            species_common = get_species_info(species)['common_name']
            result += f"{i+1}. {species_common} ({species.replace('_', ' ')}): {conf:.2%}\n"
        
        print(f"Prediction completed: {predicted_species} ({confidence:.2%})")
        return result
        
    except Exception as e:
        error_msg = f"❌ Error processing audio: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg

def predict_and_display(audio_file):
    """Predict species and return both text results and image"""
    if model is None or label_encoder is None:
        return "❌ Model not loaded. Please restart the application.", None
    
    if audio_file is None:
        return "❌ Please upload an audio file first.", None
    
    try:
        # Get prediction results
        result_text = predict_species(audio_file)
        
        # Get the predicted species for image display
        # Extract species from the audio prediction process
        audio, sr = librosa.load(audio_file, sr=16000)
        target_length = int(16000 * 2.5)
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=16000, n_mels=128, n_fft=2048, hop_length=512
        )
        mel_db = librosa.power_to_db(mel_spec)
        
        input_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_idx = torch.argmax(outputs, dim=1).item()
        
        predicted_species = label_encoder.classes_[predicted_idx]
        species_info = get_species_info(predicted_species)
        
        # Return image URL for display
        image_url = species_info.get('image_url', '')
        
        return result_text, image_url if image_url else None
        
    except Exception as e:
        error_msg = f"❌ Error processing audio: {str(e)}"
        print(error_msg)
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
    with gr.Blocks(title="🦗 Insect Sound Classifier") as interface:
        gr.Markdown("# 🦗 Insect Sound Classifier")
        gr.Markdown("Record insect sounds live or upload audio files (.wav, .mp3) to identify species!")
        
        # Model status with clickable species count
        if label_encoder:
            with gr.Row():
                gr.Markdown(f"**Model Status:** ✅ Ready")
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