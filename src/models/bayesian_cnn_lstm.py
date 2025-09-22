import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .simple_cnn_lstm import SimpleCNNLSTMInsectClassifier

class BayesianInsectClassifier(SimpleCNNLSTMInsectClassifier):
    """Bayesian extension of CNN-LSTM with uncertainty quantification and interrogatable state spaces"""
    
    def __init__(self, n_classes: int = 12, dropout: float = 0.3, uncertainty_method='monte_carlo'):
        super().__init__(n_classes, dropout)
        self.uncertainty_method = uncertainty_method
        self.n_mc_samples = 50  # For Monte Carlo Dropout
        self.dropout_rate = dropout
        
        # Enable dropout during inference for Monte Carlo sampling
        self.mc_dropout_enabled = True
        
        # Calibration parameters for confidence scoring
        self.temperature = nn.Parameter(torch.ones(1))
        self.is_calibrated = False
        
    def enable_mc_dropout(self):
        """Enable Monte Carlo dropout for uncertainty estimation"""
        self.mc_dropout_enabled = True
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()  # Keep dropout active during inference
                
    def disable_mc_dropout(self):
        """Disable Monte Carlo dropout for standard inference"""
        self.mc_dropout_enabled = False
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.eval()
    
    def forward_with_uncertainty(self, x, n_samples=None, return_attention=False):
        """
        Forward pass that returns predictions + uncertainty estimates
        
        Args:
            x: Input tensor [batch_size, channels, freq, time]
            n_samples: Number of Monte Carlo samples for uncertainty
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing predictions, uncertainties, and optional attention
        """
        if self.uncertainty_method == 'monte_carlo':
            return self._mc_forward_with_uncertainty(x, n_samples, return_attention)
        else:
            # Fallback to single forward pass
            if return_attention:
                outputs, attention = self.forward(x, return_attention=True)
                uncertainty = torch.zeros_like(outputs)
                return {
                    'predictions': outputs,
                    'uncertainty': uncertainty,
                    'attention_weights': attention
                }
            else:
                outputs = self.forward(x)
                uncertainty = torch.zeros_like(outputs)
                return {
                    'predictions': outputs,
                    'uncertainty': uncertainty
                }
    
    def _mc_forward_with_uncertainty(self, x, n_samples=None, return_attention=False):
        """Monte Carlo Dropout forward pass for uncertainty quantification"""
        n_samples = n_samples or self.n_mc_samples
        batch_size = x.size(0)
        
        # Store original training state
        was_training = self.training
        
        # Enable MC dropout
        self.enable_mc_dropout()
        
        # Collect predictions from multiple forward passes
        predictions = []
        attention_weights_list = []
        
        for i in range(n_samples):
            if return_attention:
                pred, attention = self.forward(x, return_attention=True)
                predictions.append(pred)
                attention_weights_list.append(attention)
            else:
                pred = self.forward(x)
                predictions.append(pred)
        
        # Stack predictions: [n_samples, batch_size, n_classes]
        predictions = torch.stack(predictions, dim=0)
        
        # Compute mean and uncertainty
        mean_predictions = predictions.mean(dim=0)  # [batch_size, n_classes]
        prediction_variance = predictions.var(dim=0)  # [batch_size, n_classes]
        
        # Different uncertainty measures
        epistemic_uncertainty = prediction_variance.mean(dim=1)  # Model uncertainty
        aleatoric_uncertainty = self._compute_aleatoric_uncertainty(mean_predictions)  # Data uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Predictive entropy (another uncertainty measure)
        pred_probs = F.softmax(predictions, dim=-1)
        mean_probs = pred_probs.mean(dim=0)
        predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
        
        # Restore original training state
        if not was_training:
            self.disable_mc_dropout()
            self.eval()
        
        result = {
            'predictions': mean_predictions,
            'prediction_samples': predictions,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty, 
            'total_uncertainty': total_uncertainty,
            'predictive_entropy': predictive_entropy,
            'prediction_variance': prediction_variance
        }
        
        if return_attention and attention_weights_list:
            # Stack attention weights and compute attention uncertainty
            attention_stack = torch.stack(attention_weights_list, dim=0)  # [n_samples, batch, heads, seq, seq]
            mean_attention = attention_stack.mean(dim=0)
            attention_variance = attention_stack.var(dim=0)
            
            result.update({
                'attention_weights': mean_attention,
                'attention_uncertainty': attention_variance,
                'attention_samples': attention_stack
            })
        
        return result
    
    def _compute_aleatoric_uncertainty(self, predictions):
        """Compute aleatoric (data) uncertainty from prediction confidence"""
        probs = F.softmax(predictions, dim=-1)
        # Higher entropy = more aleatoric uncertainty
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy
    
    def get_attention_uncertainty_map(self, x, n_samples=None):
        """
        Return attention patterns + uncertainty for interpretability
        
        Returns detailed attention analysis with uncertainty quantification
        """
        result = self.forward_with_uncertainty(x, n_samples, return_attention=True)
        
        if 'attention_weights' not in result:
            raise ValueError("Attention weights not available")
        
        attention = result['attention_weights']  # [batch, heads, seq, seq]
        attention_uncertainty = result.get('attention_uncertainty', torch.zeros_like(attention))
        
        # Compute attention statistics per head
        batch_size, num_heads, seq_len, _ = attention.shape
        
        # Average attention per position (where each head focuses)
        position_attention = attention.mean(dim=-1)  # [batch, heads, seq]
        
        # Attention diversity (how spread out attention is)
        attention_entropy = -torch.sum(attention * torch.log(attention + 1e-8), dim=-1).mean(dim=-1)
        
        # Head specialization (how different heads focus on different regions)
        head_similarities = []
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                similarity = F.cosine_similarity(
                    position_attention[:, i, :], 
                    position_attention[:, j, :], 
                    dim=1
                )
                head_similarities.append(similarity)
        
        avg_head_similarity = torch.stack(head_similarities).mean(dim=0) if head_similarities else torch.zeros(batch_size)
        
        return {
            'predictions': result['predictions'],
            'prediction_uncertainty': result['total_uncertainty'],
            'attention_map': attention,
            'attention_uncertainty': attention_uncertainty,
            'position_attention': position_attention,
            'attention_entropy': attention_entropy,
            'head_specialization': 1.0 - avg_head_similarity,  # Higher = more specialized
            'species_confidence': self.calibrated_confidence(result['predictions']),
            'uncertainty_breakdown': {
                'epistemic': result['epistemic_uncertainty'],
                'aleatoric': result['aleatoric_uncertainty'],
                'predictive_entropy': result['predictive_entropy']
            }
        }
    
    def calibrated_confidence(self, logits):
        """Apply temperature scaling for calibrated confidence scores"""
        if self.is_calibrated:
            scaled_logits = logits / self.temperature
        else:
            scaled_logits = logits
        
        probs = F.softmax(scaled_logits, dim=-1)
        confidence, predicted_class = torch.max(probs, dim=-1)
        
        return {
            'confidence': confidence,
            'predicted_class': predicted_class,
            'class_probabilities': probs
        }
    
    def calibrate_temperature(self, val_loader, device='cpu'):
        """
        Calibrate temperature parameter using validation set
        Implements Platt scaling for better confidence calibration
        """
        self.eval()
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                logits = self.forward(X_batch)
                logits_list.append(logits)
                labels_list.append(y_batch)
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        # Optimize temperature using NLL loss
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        self.is_calibrated = True
        
        print(f"🌡️ Temperature calibration complete: T = {self.temperature.item():.3f}")
        return self.temperature.item()
    
    def compute_uncertainty_metrics(self, predictions_dict, true_labels=None):
        """Compute comprehensive uncertainty metrics for analysis"""
        metrics = {}
        
        # Basic uncertainty statistics
        epistemic = predictions_dict['epistemic_uncertainty']
        aleatoric = predictions_dict['aleatoric_uncertainty']
        
        metrics.update({
            'mean_epistemic_uncertainty': epistemic.mean().item(),
            'std_epistemic_uncertainty': epistemic.std().item(),
            'mean_aleatoric_uncertainty': aleatoric.mean().item(),
            'std_aleatoric_uncertainty': aleatoric.std().item(),
            'mean_predictive_entropy': predictions_dict['predictive_entropy'].mean().item()
        })
        
        if true_labels is not None:
            # Uncertainty quality metrics
            predictions = predictions_dict['predictions']
            predicted_classes = torch.argmax(predictions, dim=-1)
            correct_mask = (predicted_classes == true_labels)
            
            # Average uncertainty for correct vs incorrect predictions
            correct_uncertainty = epistemic[correct_mask].mean() if correct_mask.any() else 0
            incorrect_uncertainty = epistemic[~correct_mask].mean() if (~correct_mask).any() else 0
            
            metrics.update({
                'uncertainty_separation': (incorrect_uncertainty - correct_uncertainty).item(),
                'accuracy': correct_mask.float().mean().item()
            })
        
        return metrics

    def save_bayesian_state(self, path):
        """Save model state including calibration parameters"""
        state = {
            'model_state_dict': self.state_dict(),
            'temperature': self.temperature.item(),
            'is_calibrated': self.is_calibrated,
            'uncertainty_method': self.uncertainty_method,
            'n_mc_samples': self.n_mc_samples,
            'n_classes': self.n_classes
        }
        torch.save(state, path)
        
    def load_bayesian_state(self, path, map_location='cpu'):
        """Load model state including calibration parameters"""
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state['model_state_dict'])
        self.temperature.data = torch.tensor([state['temperature']])
        self.is_calibrated = state['is_calibrated']
        self.uncertainty_method = state['uncertainty_method']
        self.n_mc_samples = state['n_mc_samples']