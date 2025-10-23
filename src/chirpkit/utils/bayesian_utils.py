import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
try:
    from ..models.bayesian_cnn_lstm import BayesianInsectClassifier
    from ..visualization.uncertainty_viz import UncertaintyVisualizer
except ImportError:
    from chirpkit.models.bayesian_cnn_lstm import BayesianInsectClassifier
    try:
        from chirpkit.visualization.uncertainty_viz import UncertaintyVisualizer
    except ImportError:
        # Skip visualization if not available
        UncertaintyVisualizer = None

class BayesianModelManager:
    """Utility class for managing Bayesian insect classifier with uncertainty quantification"""
    
    def __init__(self, model_path: str = None, n_classes: int = 471, device: str = 'auto'):
        """
        Initialize Bayesian model manager
        
        Args:
            model_path: Path to pre-trained model
            n_classes: Number of insect species classes
            device: Device to run model on ('auto', 'cpu', 'cuda')
        """
        self.device = self._setup_device(device)
        self.n_classes = n_classes
        self.model = None
        self.label_encoder = None
        self.species_list = None
        self.visualizer = UncertaintyVisualizer()
        
        if model_path:
            self.load_model(model_path)
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def load_model(self, model_path: str, label_encoder_path: str = None):
        """Load pre-trained Bayesian model and label encoder"""
        model_path = Path(model_path)
        
        # Initialize model
        self.model = BayesianInsectClassifier(n_classes=self.n_classes)
        
        # Load weights
        if model_path.suffix == '.pth':
            # Standard PyTorch model
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ Loaded model weights from: {model_path}")
        else:
            # Bayesian state with calibration
            self.model.load_bayesian_state(model_path, map_location=self.device)
            print(f"✅ Loaded Bayesian model state from: {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load label encoder
        if label_encoder_path is None:
            # Try to find label encoder automatically
            label_encoder_path = model_path.parent / f"{model_path.stem}_label_encoder.joblib"
        
        if Path(label_encoder_path).exists():
            import joblib
            self.label_encoder = joblib.load(label_encoder_path)
            self.species_list = self.label_encoder.classes_.tolist()
            print(f"✅ Loaded label encoder: {len(self.species_list)} species")
        else:
            print(f"⚠️ Label encoder not found at: {label_encoder_path}")
    
    def convert_standard_to_bayesian(self, standard_model_path: str, output_path: str = None):
        """Convert standard CNN-LSTM model to Bayesian version"""
        from ..models.simple_cnn_lstm import SimpleCNNLSTMInsectClassifier
        
        # Load standard model
        standard_model = SimpleCNNLSTMInsectClassifier(n_classes=self.n_classes)
        state_dict = torch.load(standard_model_path, map_location='cpu')
        standard_model.load_state_dict(state_dict)
        
        # Create Bayesian version
        bayesian_model = BayesianInsectClassifier(n_classes=self.n_classes)
        
        # Copy weights (architectures are compatible)
        bayesian_model.load_state_dict(state_dict, strict=False)
        
        if output_path is None:
            output_path = Path(standard_model_path).parent / f"bayesian_{Path(standard_model_path).name}"
        
        # Save Bayesian version
        bayesian_model.save_bayesian_state(output_path)
        print(f"✅ Converted to Bayesian model: {output_path}")
        
        return output_path
    
    def predict_with_uncertainty(self, audio_features: torch.Tensor, 
                                n_samples: int = 50, 
                                return_attention: bool = True) -> Dict:
        """
        Make predictions with uncertainty quantification
        
        Args:
            audio_features: Input mel spectrogram [batch, channels, freq, time]
            n_samples: Number of Monte Carlo samples
            return_attention: Whether to return attention analysis
            
        Returns:
            Dictionary with predictions, uncertainties, and interpretability data
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        
        with torch.no_grad():
            # Get uncertainty analysis
            if return_attention:
                uncertainty_data = self.model.get_attention_uncertainty_map(
                    audio_features.to(self.device), n_samples=n_samples
                )
            else:
                uncertainty_data = self.model.forward_with_uncertainty(
                    audio_features.to(self.device), n_samples=n_samples
                )
        
        # Add species information if available
        if self.label_encoder is not None:
            predictions = uncertainty_data['predictions']
            predicted_classes = torch.argmax(predictions, dim=-1)
            
            species_info = []
            for i in range(predictions.shape[0]):
                class_idx = predicted_classes[i].item()
                species_name = self.species_list[class_idx] if class_idx < len(self.species_list) else f"Unknown_{class_idx}"
                
                # Get top-k predictions
                top_k = min(5, len(self.species_list))
                probs = torch.softmax(predictions[i], dim=0)
                top_probs, top_indices = torch.topk(probs, top_k)
                
                top_species = [
                    {
                        'species': self.species_list[idx.item()],
                        'probability': prob.item(),
                        'confidence_level': self._classify_confidence(prob.item())
                    }
                    for idx, prob in zip(top_indices, top_probs)
                ]
                
                species_info.append({
                    'predicted_species': species_name,
                    'confidence': uncertainty_data['species_confidence']['confidence'][i].item(),
                    'uncertainty': uncertainty_data['prediction_uncertainty'][i].item(),
                    'top_predictions': top_species,
                    'uncertainty_level': self._classify_uncertainty(uncertainty_data['prediction_uncertainty'][i].item())
                })
            
            uncertainty_data['species_info'] = species_info
        
        return uncertainty_data
    
    def _classify_confidence(self, confidence: float) -> str:
        """Classify confidence level into human-readable categories"""
        if confidence > 0.8:
            return "Very High"
        elif confidence > 0.6:
            return "High"
        elif confidence > 0.4:
            return "Medium"
        elif confidence > 0.2:
            return "Low"
        else:
            return "Very Low"
    
    def _classify_uncertainty(self, uncertainty: float) -> str:
        """Classify uncertainty level into human-readable categories"""
        if uncertainty < 0.3:
            return "Low"
        elif uncertainty < 0.7:
            return "Medium"
        else:
            return "High"
    
    def calibrate_model(self, val_dataloader, save_path: str = None):
        """Calibrate model for better confidence estimation"""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        print("🌡️ Calibrating model temperature...")
        temperature = self.model.calibrate_temperature(val_dataloader, device=self.device)
        
        if save_path:
            self.model.save_bayesian_state(save_path)
            print(f"💾 Calibrated model saved to: {save_path}")
        
        return temperature
    
    def analyze_uncertainty_patterns(self, dataloader, max_batches: int = 10) -> Dict:
        """Analyze uncertainty patterns across dataset"""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        all_uncertainties = []
        all_predictions = []
        all_labels = []
        
        self.model.eval()
        batch_count = 0
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                if batch_count >= max_batches:
                    break
                
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                uncertainty_data = self.model.forward_with_uncertainty(X_batch, n_samples=20)
                
                all_uncertainties.append(uncertainty_data['total_uncertainty'].cpu())
                all_predictions.append(uncertainty_data['predictions'].cpu())
                all_labels.append(y_batch.cpu())
                
                batch_count += 1
        
        # Aggregate results
        uncertainties = torch.cat(all_uncertainties)
        predictions = torch.cat(all_predictions)
        labels = torch.cat(all_labels)
        
        # Compute analysis metrics
        predicted_classes = torch.argmax(predictions, dim=-1)
        correct_mask = (predicted_classes == labels)
        
        analysis = {
            'total_samples': len(uncertainties),
            'accuracy': correct_mask.float().mean().item(),
            'mean_uncertainty': uncertainties.mean().item(),
            'uncertainty_std': uncertainties.std().item(),
            'correct_predictions_uncertainty': uncertainties[correct_mask].mean().item() if correct_mask.any() else 0,
            'incorrect_predictions_uncertainty': uncertainties[~correct_mask].mean().item() if (~correct_mask).any() else 0,
        }
        
        # Per-class uncertainty analysis
        if self.species_list:
            class_uncertainties = {}
            for class_idx in range(min(len(self.species_list), predictions.shape[1])):
                class_mask = (predicted_classes == class_idx)
                if class_mask.any():
                    class_uncertainties[self.species_list[class_idx]] = uncertainties[class_mask].mean().item()
            
            analysis['class_uncertainties'] = class_uncertainties
        
        return analysis
    
    def create_uncertainty_visualization(self, audio_features: torch.Tensor, 
                                      output_dir: str = "uncertainty_analysis",
                                      save_plots: bool = True) -> Dict:
        """Create comprehensive uncertainty visualization"""
        # Get uncertainty analysis
        uncertainty_data = self.predict_with_uncertainty(audio_features, return_attention=True)
        
        if save_plots:
            # Save visualizations
            plots = self.visualizer.save_uncertainty_report(
                uncertainty_data, 
                audio_features[0, 0],  # First sample, first channel
                output_dir
            )
            
            return {
                'uncertainty_data': uncertainty_data,
                'plot_paths': plots
            }
        else:
            # Return matplotlib figures
            fig1 = self.visualizer.plot_uncertainty_map(uncertainty_data, audio_features[0, 0])
            fig2 = self.visualizer.create_uncertainty_summary_card(uncertainty_data)
            
            return {
                'uncertainty_data': uncertainty_data,
                'figures': {
                    'uncertainty_map': fig1,
                    'summary_card': fig2
                }
            }
    
    def get_model_summary(self) -> Dict:
        """Get summary of loaded Bayesian model"""
        if self.model is None:
            return {"status": "No model loaded"}
        
        return {
            "status": "Model loaded",
            "n_classes": self.n_classes,
            "n_species": len(self.species_list) if self.species_list else None,
            "uncertainty_method": self.model.uncertainty_method,
            "mc_samples": self.model.n_mc_samples,
            "is_calibrated": self.model.is_calibrated,
            "temperature": self.model.temperature.item() if self.model.is_calibrated else None,
            "device": str(self.device)
        }

def convert_existing_model_to_bayesian(model_path: str, output_dir: str = None) -> str:
    """Utility function to convert existing ChirpKit model to Bayesian version"""
    manager = BayesianModelManager()
    
    if output_dir is None:
        output_dir = Path(model_path).parent
    
    bayesian_path = manager.convert_standard_to_bayesian(
        model_path, 
        Path(output_dir) / f"bayesian_{Path(model_path).name}"
    )
    
    return str(bayesian_path)