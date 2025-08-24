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

# Global variables for model components
model = None
label_encoder = None
device = None

def load_model():
    """Load the trained model and preprocessing components"""
    global model, label_encoder, device
    
    try:
        print("Loading model components...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        
        # Get species name
        predicted_species = label_encoder.classes_[predicted_idx]
        
        # Create confidence scores for top 5
        probs = probabilities[0].cpu().numpy()
        species_probs = [(label_encoder.classes_[i], float(probs[i])) for i in range(len(label_encoder.classes_))]
        species_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Format result
        result = f"🦗 **Predicted Species:** {predicted_species}\n"
        result += f"🎯 **Confidence:** {confidence:.2%}\n\n"
        result += "📊 **Top 5 Predictions:**\n"
        
        for i, (species, conf) in enumerate(species_probs[:5]):
            result += f"{i+1}. {species}: {conf:.2%}\n"
        
        print(f"Prediction completed: {predicted_species} ({confidence:.2%})")
        return result
        
    except Exception as e:
        error_msg = f"❌ Error processing audio: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg

def create_interface():
    """Create the Gradio interface"""
    with gr.Blocks(title="🦗 Insect Sound Classifier") as interface:
        gr.Markdown("# 🦗 Insect Sound Classifier")
        gr.Markdown("Upload an audio file (.wav, .mp3) to identify the insect species!")
        
        if label_encoder:
            gr.Markdown(f"**Model Status:** ✅ Ready ({len(label_encoder.classes_)} species)")
        else:
            gr.Markdown("**Model Status:** ❌ Not loaded")
        
        with gr.Row():
            with gr.Column():
                # Audio input
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath"
                )
                
                # Predict button
                predict_btn = gr.Button("🔍 Identify Species", variant="primary", size="lg")
            
            with gr.Column():
                # Results
                result_output = gr.Textbox(
                    label="🎯 Prediction Results",
                    lines=10,
                    max_lines=15,
                    placeholder="Upload an audio file and click 'Identify Species' to see results..."
                )
        
        # Species reference
        if label_encoder:
            with gr.Accordion("📚 Supported Species", open=False):
                species_list = "\n".join([f"{i+1}. {species}" for i, species in enumerate(label_encoder.classes_)])
                gr.Textbox(value=species_list, lines=10, label="All Supported Species")
        
        # Connect the prediction function
        predict_btn.click(
            fn=predict_species,
            inputs=[audio_input],
            outputs=[result_output]
        )
    
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