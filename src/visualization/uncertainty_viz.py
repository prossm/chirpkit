import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import io
import base64
from PIL import Image

class UncertaintyVisualizer:
    """Visualization tools for Bayesian model uncertainty and attention analysis"""
    
    def __init__(self, figsize=(15, 10), style='whitegrid'):
        self.figsize = figsize
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = figsize
        
    def plot_uncertainty_map(self, uncertainty_data, audio_features=None, save_path=None):
        """
        Create comprehensive uncertainty visualization
        
        Args:
            uncertainty_data: Output from get_attention_uncertainty_map()
            audio_features: Original mel spectrogram [freq, time] 
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🎲 Bayesian Uncertainty Analysis for Insect Classification', fontsize=16, fontweight='bold')
        
        # Extract data
        predictions = uncertainty_data['predictions'][0]  # First sample in batch
        pred_uncertainty = uncertainty_data['prediction_uncertainty'][0]
        attention_map = uncertainty_data['attention_map'][0]  # [heads, seq, seq]
        position_attention = uncertainty_data['position_attention'][0]  # [heads, seq]
        uncertainty_breakdown = uncertainty_data['uncertainty_breakdown']
        
        # 1. Prediction confidence with uncertainty
        ax1 = axes[0, 0]
        probs = torch.softmax(predictions, dim=0).cpu().numpy()
        top_k = 5
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        bars = ax1.bar(range(top_k), probs[top_indices])
        ax1.set_title(f'🏆 Top {top_k} Predictions\n(Uncertainty: {pred_uncertainty:.3f})')
        ax1.set_xlabel('Species Rank')
        ax1.set_ylabel('Probability')
        ax1.set_xticks(range(top_k))
        ax1.set_xticklabels([f'#{i+1}' for i in range(top_k)])
        
        # Color bars by confidence
        for i, bar in enumerate(bars):
            confidence = probs[top_indices[i]]
            bar.set_color(plt.cm.RdYlGn(confidence))
        
        # 2. Uncertainty breakdown
        ax2 = axes[0, 1]
        uncertainty_types = ['Epistemic\n(Model)', 'Aleatoric\n(Data)', 'Predictive\nEntropy']
        uncertainty_values = [
            uncertainty_breakdown['epistemic'][0].item(),
            uncertainty_breakdown['aleatoric'][0].item(), 
            uncertainty_breakdown['predictive_entropy'][0].item()
        ]
        
        bars = ax2.bar(uncertainty_types, uncertainty_values, 
                      color=['#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_title('🔍 Uncertainty Breakdown')
        ax2.set_ylabel('Uncertainty Value')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars, uncertainty_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # 3. Attention head specialization
        ax3 = axes[0, 2]
        num_heads = position_attention.shape[0]
        head_entropies = []
        
        for head in range(num_heads):
            head_att = position_attention[head].cpu().numpy()
            entropy = -np.sum(head_att * np.log(head_att + 1e-8))
            head_entropies.append(entropy)
        
        bars = ax3.bar(range(num_heads), head_entropies, color=plt.cm.viridis(np.linspace(0, 1, num_heads)))
        ax3.set_title('🧠 Attention Head Specialization')
        ax3.set_xlabel('Attention Head')
        ax3.set_ylabel('Entropy (Higher = More Spread)')
        ax3.set_xticks(range(num_heads))
        
        # 4. Audio spectrogram with attention overlay
        ax4 = axes[1, 0]
        if audio_features is not None:
            # Show mel spectrogram
            im = ax4.imshow(audio_features.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
            ax4.set_title('🎵 Mel Spectrogram')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Mel Frequency')
            plt.colorbar(im, ax=ax4, shrink=0.8)
        else:
            ax4.text(0.5, 0.5, 'Audio features\nnot provided', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('🎵 Audio Input')
        
        # 5. Attention heatmap (average across heads)
        ax5 = axes[1, 1]
        avg_attention = attention_map.mean(dim=0).cpu().numpy()  # Average across heads
        im = ax5.imshow(avg_attention, cmap='Blues', aspect='auto')
        ax5.set_title('👁️ Average Attention Pattern')
        ax5.set_xlabel('Key Position')
        ax5.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax5, shrink=0.8)
        
        # 6. Per-head attention patterns
        ax6 = axes[1, 2]
        # Show attention diversity across time for each head
        time_attention = position_attention.cpu().numpy()  # [heads, seq]
        
        for head in range(min(num_heads, 4)):  # Show up to 4 heads
            ax6.plot(time_attention[head], label=f'Head {head+1}', linewidth=2)
        
        ax6.set_title('⏰ Temporal Attention Patterns')
        ax6.set_xlabel('Time Step')
        ax6.set_ylabel('Attention Weight')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Uncertainty visualization saved to: {save_path}")
        
        return fig
    
    def plot_uncertainty_distribution(self, uncertainty_results, species_labels=None, save_path=None):
        """Plot distribution of uncertainties across predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('📊 Uncertainty Distribution Analysis', fontsize=16, fontweight='bold')
        
        epistemic = uncertainty_results['epistemic_uncertainty'].cpu().numpy()
        aleatoric = uncertainty_results['aleatoric_uncertainty'].cpu().numpy()
        total_uncertainty = uncertainty_results['total_uncertainty'].cpu().numpy()
        predictions = uncertainty_results['predictions'].cpu().numpy()
        
        # 1. Uncertainty distributions
        ax1 = axes[0, 0]
        ax1.hist(epistemic, bins=30, alpha=0.7, label='Epistemic', color='orange', density=True)
        ax1.hist(aleatoric, bins=30, alpha=0.7, label='Aleatoric', color='green', density=True)
        ax1.set_title('🔍 Uncertainty Type Distribution')
        ax1.set_xlabel('Uncertainty Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Prediction confidence vs uncertainty
        ax2 = axes[0, 1]
        max_probs = np.max(torch.softmax(torch.tensor(predictions), dim=1).numpy(), axis=1)
        scatter = ax2.scatter(max_probs, total_uncertainty, c=total_uncertainty, 
                            cmap='RdYlGn_r', alpha=0.6)
        ax2.set_title('🎯 Confidence vs Uncertainty')
        ax2.set_xlabel('Max Prediction Probability')
        ax2.set_ylabel('Total Uncertainty')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Uncertainty')
        
        # 3. Uncertainty by predicted class
        ax3 = axes[1, 0]
        predicted_classes = np.argmax(predictions, axis=1)
        
        if species_labels is not None and len(species_labels) > 0:
            unique_classes = np.unique(predicted_classes)
            class_uncertainties = []
            class_names = []
            
            for cls in unique_classes[:10]:  # Show top 10 classes
                mask = predicted_classes == cls
                if np.sum(mask) > 0:
                    class_uncertainties.append(total_uncertainty[mask].mean())
                    if cls < len(species_labels):
                        class_names.append(species_labels[cls][:15])  # Truncate long names
                    else:
                        class_names.append(f'Class {cls}')
            
            bars = ax3.bar(range(len(class_uncertainties)), class_uncertainties, 
                          color=plt.cm.viridis(np.linspace(0, 1, len(class_uncertainties))))
            ax3.set_title('🦗 Uncertainty by Species')
            ax3.set_xlabel('Species')
            ax3.set_ylabel('Average Uncertainty')
            ax3.set_xticks(range(len(class_names)))
            ax3.set_xticklabels(class_names, rotation=45, ha='right')
        else:
            ax3.text(0.5, 0.5, 'Species labels\nnot provided', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('🦗 Species Analysis')
        
        # 4. Epistemic vs Aleatoric scatter
        ax4 = axes[1, 1]
        scatter = ax4.scatter(epistemic, aleatoric, c=max_probs, cmap='RdYlGn', alpha=0.6)
        ax4.set_title('🧠 Epistemic vs Aleatoric Uncertainty')
        ax4.set_xlabel('Epistemic Uncertainty (Model)')
        ax4.set_ylabel('Aleatoric Uncertainty (Data)')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Confidence')
        
        # Add diagonal line
        max_val = max(epistemic.max(), aleatoric.max())
        ax4.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal uncertainty')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Uncertainty distribution plot saved to: {save_path}")
        
        return fig
    
    def create_uncertainty_summary_card(self, uncertainty_data, species_name=None):
        """Create a compact summary card for web UI display"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Extract key metrics
        pred_uncertainty = uncertainty_data['prediction_uncertainty'][0].item()
        species_confidence = uncertainty_data['species_confidence']
        confidence = species_confidence['confidence'][0].item()
        uncertainty_breakdown = uncertainty_data['uncertainty_breakdown']
        
        # Create summary text
        summary_text = f"""
🎲 Bayesian Uncertainty Analysis

🏆 Prediction Confidence: {confidence:.3f}
🔍 Total Uncertainty: {pred_uncertainty:.3f}

Uncertainty Breakdown:
🧠 Model Uncertainty: {uncertainty_breakdown['epistemic'][0]:.3f}
📊 Data Uncertainty: {uncertainty_breakdown['aleatoric'][0]:.3f}
📈 Predictive Entropy: {uncertainty_breakdown['predictive_entropy'][0]:.3f}

👁️ Attention Diversity: {uncertainty_data['head_specialization'][0]:.3f}
        """
        
        if species_name:
            summary_text = f"🦗 Species: {species_name}\n" + summary_text
        
        # Create visual elements
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Add uncertainty gauge
        uncertainty_level = "Low" if pred_uncertainty < 0.5 else "Medium" if pred_uncertainty < 1.0 else "High"
        gauge_color = "green" if pred_uncertainty < 0.5 else "orange" if pred_uncertainty < 1.0 else "red"
        
        ax.text(0.7, 0.3, f"Uncertainty Level:\n{uncertainty_level}", 
               transform=ax.transAxes, fontsize=14, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor=gauge_color, alpha=0.7))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('🔍 Model Uncertainty Summary', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string for web display"""
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{img_str}"
    
    def save_uncertainty_report(self, uncertainty_data, audio_features, output_dir, filename_prefix="uncertainty_analysis"):
        """Save comprehensive uncertainty analysis report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Main uncertainty map
        fig1 = self.plot_uncertainty_map(uncertainty_data, audio_features)
        fig1.savefig(output_dir / f"{filename_prefix}_map.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # Summary card
        fig2 = self.create_uncertainty_summary_card(uncertainty_data)
        fig2.savefig(output_dir / f"{filename_prefix}_summary.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        print(f"📊 Uncertainty analysis saved to: {output_dir}")
        
        return {
            'map_path': output_dir / f"{filename_prefix}_map.png",
            'summary_path': output_dir / f"{filename_prefix}_summary.png"
        }